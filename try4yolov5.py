"""
Kerem Berke YOLOv5 Plaka Modeli ile Plaka Okuma (Otomatik İndirmeli)
Tracking (IoU) + Firebase Entegrasyonu
try1.py
"""

from firebase_admin import credentials, db, storage
import tempfile
import cv2
import torch
import numpy as np
from datetime import datetime
import firebase_admin
import easyocr
import os
import time
# Yeni kütüphane
import yolov5

# ==================== YAPLANDIRMA ====================
FIREBASE_CRED_PATH = "firebase-credentials.json"
FIREBASE_DB_URL = "https://try1-cc8eb-default-rtdb.europe-west1.firebasedatabase.app/"
FIREBASE_STORAGE_BUCKET = "try1-cc8eb.firebasestorage.app"

VIDEO_PATH = "video2.mp4"
VIDEO_DURATION = 8
CONFIDENCE_THRESHOLD = 0.25      # Plaka tespiti için güven eşiği
OCR_CONFIDENCE_THRESHOLD = 0.4   # OCR okuma güven eşiği

# ==================== YARDIMCI MATEMATİK FONKSİYONLARI ====================
def calculate_iou(box1, box2):
    """Tracking için Kesişim/Birleşim hesabı"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# ==================== FIREBASE BAŞLATMA ====================
def initialize_firebase():
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CRED_PATH)
            firebase_admin.initialize_app(cred, {
                "databaseURL": FIREBASE_DB_URL,
                "storageBucket": FIREBASE_STORAGE_BUCKET,
            })
        print("✓ Firebase bağlantısı başarılı")
        return True
    except Exception as e:
        print(f"✗ Firebase bağlantı hatası: {e}")
        return False

def check_firebase_status():
    try:
        ref = db.reference('test/status')
        status = ref.get()
        return status == "yes"
    except Exception as e:
        return False

def download_video_from_storage(storage_path):
    try:
        print(f"Video indiriliyor: {storage_path}")
        bucket = storage.bucket()
        blob = bucket.blob(storage_path)
        temp_dir = tempfile.gettempdir()
        local_path = os.path.join(temp_dir, "temp_video.mp4")
        blob.download_to_filename(local_path)
        print(f"✓ Video indirildi: {local_path}")
        return local_path
    except Exception as e:
        print(f"✗ Video indirme hatası: {e}")
        return None

def send_plates_to_firebase(plates_data, video_meta=None):
    try:
        ref = db.reference('test/detected_plates')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        data = {
            timestamp: {
                'total_plates': len(plates_data),
                'detection_time': datetime.now().isoformat(),
                'video_meta': video_meta or {},
                'plates': plates_data
            }
        }

        ref.update(data)
        print(f"✓ {len(plates_data)} plaka Firebase'e gönderildi")
        db.reference('test/status').set("no")
        return True
    except Exception as e:
        print(f"✗ Firebase yazma hatası: {e}")
        return False


# ==================== MODEL VE OCR ====================
def load_plate_model():
    """Kerem Berke'nin Plaka Modelini Yükle"""
    try:
        print("⏳ Model indiriliyor/yükleniyor (Kerem Berke YOLOv5 License Plate)...")
        # Bu satır modeli otomatik indirir
        model = yolov5.load('keremberke/yolov5m-license-plate')
        
        # Model Ayarları
        model.conf = 0.25
        model.iou = 0.45
        model.agnostic = False
        model.multi_label = False
        model.max_det = 1000
        
        print("✓ Plaka Modeli Yüklendi!")
        return model
    except Exception as e:
        print(f"✗ Model yükleme hatası: {e}")
        print("Lütfen 'pip install yolov5' komutunu çalıştırdığınızdan emin olun.")
        return None

def initialize_ocr():
    try:
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print("✓ OCR başlatıldı")
        return reader
    except Exception as e:
        print(f"✗ OCR hatası: {e}")
        return None

def clean_plate_text(text):
    cleaned = ''.join(c for c in text if c.isalnum())
    # Plaka uzunluk kontrolü (TR plakalar genelde 6-9 karakter)
    if len(cleaned) >= 5 and len(cleaned) <= 9:
        return cleaned.upper()
    return None

# ==================== VİDEO İŞLEME ====================
def get_video_metadata(local_video_path, storage_video_path, requested_duration):
    """
    local_video_path   : temp_video.mp4 (yerel)
    storage_video_path : video2.mp4 (Firebase Storage'daki gerçek isim)
    """
    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        return {
            "video_name": os.path.basename(storage_video_path),
            "storage_path": storage_video_path,
            "requested_duration_sec": requested_duration,
            "video_duration_sec": None,
            "processed_duration_sec": requested_duration,
            "fps": None,
            "total_frames": None
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_duration_sec = (total_frames / fps) if fps > 0 else None

    processed_duration_sec = requested_duration
    if video_duration_sec is not None:
        processed_duration_sec = min(requested_duration, video_duration_sec)

    cap.release()

    return {
        "video_name": os.path.basename(storage_video_path),   # ✅ GERÇEK AD
        "storage_path": storage_video_path,                   # ✅ FIREBASE PATH
        "requested_duration_sec": requested_duration,
        "video_duration_sec": round(video_duration_sec, 2) if video_duration_sec else None,
        "processed_duration_sec": round(processed_duration_sec, 2),
        "fps": round(float(fps), 3) if fps else None,
        "total_frames": total_frames
    }


def process_video_and_detect_plates(model, ocr_reader, video_path, duration):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Video açılamadı: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(fps * duration)
    
    # Tracking Değişkenleri
    detected_objects = [] 
    object_counter = 0     
    IOU_THRESHOLD = 0.2    
    FRAME_DROPOUT = 30     
    
    frame_count = 0
    start_time = time.time()
    
    print(f"Video işleniyor... (~{total_frames} frame)")
    
    # DEBUG: İlk plakayı görüp görmediğini kontrol etmek için
    debug_saved = False
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time() - start_time
        
        # Her 5. kareyi işle
        if frame_count % 5 == 0:
            # YOLOv5 Tahmini
            results = model(frame)
            
            # Sonuçları al (x1, y1, x2, y2, conf, cls)
            # .xyxy[0] tensör döndürür, cpu().numpy() ile diziye çevirelim
            detections = results.xyxy[0].cpu().numpy()
            
            # --- DEBUG RESMİ KAYDETME (Sadece ilk tespit edilen kare) ---
            if len(detections) > 0 and not debug_saved:
                debug_frame = frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = map(int, det[:4])
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.imwrite("debug_yolov5_plaka.jpg", debug_frame)
                print("  [BİLGİ] 'debug_yolov5_plaka.jpg' kaydedildi. Plakaları doğru görüyor mu kontrol et!")
                debug_saved = True
            # ------------------------------------------------------------

            for det in detections:
                x1, y1, x2, y2 = map(int, det[:4])
                conf = float(det[4])
                
                # Sadece güven eşiğini geçenleri al (Model zaten filtreliyor ama garanti olsun)
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                    
                current_box = [x1, y1, x2, y2]
                
                # --- TRACKING (Aynı plaka mı?) ---
                matched_obj = None
                for obj in detected_objects:
                    if frame_count - obj['last_seen_frame'] > FRAME_DROPOUT: continue
                    
                    if calculate_iou(current_box, obj['last_box']) > IOU_THRESHOLD:
                        matched_obj = obj
                        obj['last_box'] = current_box
                        obj['last_seen_frame'] = frame_count
                        break
                
                if matched_obj is None:
                    object_counter += 1
                    matched_obj = {
                        'id': object_counter,
                        'best_plate': None,
                        'best_conf': 0.0,
                        'last_box': current_box,
                        'last_seen_frame': frame_count,
                        'detection_time': round(current_time, 2)
                    }
                    detected_objects.append(matched_obj)

                # --- OCR ---
                # Kenar kontrolü
                h, w, _ = frame.shape
                if x1 < 5 or y1 < 5 or x2 > w-5 or y2 > h-5: continue

                plate_roi = frame[y1:y2, x1:x2]
                if plate_roi.size == 0: continue
                
                # ROI'yi biraz büyütmek/geliştirmek OCR'ı artırabilir (Opsiyonel: Gray scale)
                # plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY) 

                try:
                    ocr_results = ocr_reader.readtext(plate_roi)
                    for detection in ocr_results:
                        text = detection[1]
                        ocr_conf = detection[2]
                        
                        if ocr_conf > OCR_CONFIDENCE_THRESHOLD:
                            cleaned_text = clean_plate_text(text)
                            
                            if cleaned_text:
                                if ocr_conf > matched_obj['best_conf']:
                                    prev = matched_obj['best_plate']
                                    matched_obj['best_plate'] = cleaned_text
                                    matched_obj['best_conf'] = ocr_conf
                                    
                                    if prev != cleaned_text:
                                        print(f"  [Araç {matched_obj['id']}] Plaka: {cleaned_text} (Güven: %{ocr_conf*100:.0f})")

                except Exception:
                    pass

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  İlerleme: %{(frame_count / total_frames) * 100:.1f}")
    
    cap.release()
    
    final_results = []
    print("\n" + "="*30)
    print("SONUÇLAR")
    for obj in detected_objects:
        if obj['best_plate']:
            res = {
                'plate': obj['best_plate'],
                'confidence': round(obj['best_conf'] * 100, 2),
                'detection_confidence': 99.0,
                'time_in_video': obj['detection_time'],
                'frame': obj['last_seen_frame'],
                'vehicle_id': obj['id']
            }
            final_results.append(res)
            print(f"Araç #{obj['id']} -> {res['plate']}")
    print("="*30)
    
    return final_results

# ==================== ANA PROGRAM ====================
def main():
    print("--- KEREM BERKE YOLOv5 PLAKA OKUMA ---")
    
    if not initialize_firebase(): return
    
    # Yeni model yükleyiciyi çağırıyoruz
    model = load_plate_model()
    if model is None: return
    
    ocr_reader = initialize_ocr()
    if ocr_reader is None: return
    
    print("\nSistem hazır. Firebase'den sinyal bekleniyor...")
    
    while True:
        try:
            if check_firebase_status():
                print("\n🔊 Sinyal Geldi!")
                local_video_path = download_video_from_storage(VIDEO_PATH)
                
                if local_video_path:
                    plates = process_video_and_detect_plates(
                        model, ocr_reader, local_video_path, VIDEO_DURATION
                    )
                    
                    try: os.remove(local_video_path)
                    except: pass
                    
                    video_meta = get_video_metadata(local_video_path=local_video_path,storage_video_path=VIDEO_PATH,requested_duration=VIDEO_DURATION)

                    if plates:
                        send_plates_to_firebase(plates, video_meta=video_meta)
                    else:
                        print("⚠ Plaka okunamadı.")
                        # Yine de video bilgisi yazmak istersen:
                        send_plates_to_firebase([], video_meta=video_meta)
                        db.reference('test/status').set("no")                
                print("Beklemede...")
            
            time.sleep(2)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Hata: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()