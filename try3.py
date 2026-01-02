"""
RF-DETR / YOLOv8 ile Plaka Okuma ve Firebase Entegrasyonu
Konum Takibi (IoU) ve En İyi Skoru Saklama (Best Confidence Retention) Özellikli
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
from pathlib import Path

# ==================== YAPLANDIRMA ====================
FIREBASE_CRED_PATH = "firebase-credentials.json"
FIREBASE_DB_URL = "https://try1-cc8eb-default-rtdb.europe-west1.firebasedatabase.app/"
FIREBASE_STORAGE_BUCKET = "try1-cc8eb.firebasestorage.app"

VIDEO_PATH = "video2.mp4"       # Trafik videonuzun yolu
VIDEO_DURATION = 8              # Video işleme süresi (saniye)
CONFIDENCE_THRESHOLD = 0.5      # Araç/Plaka tespit güven eşiği
OCR_CONFIDENCE_THRESHOLD = 0.5  # OCR okuma güven eşiği (0.5 = %50)

# ==================== YARDIMCI MATEMATİK FONKSİYONLARI ====================
def calculate_iou(box1, box2):
    """
    İki kutu arasındaki kesişim oranını (IoU) hesaplar.
    Bu, aynı aracı takip etmek için kullanılır.
    box: [x1, y1, x2, y2]
    """
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
        # print(f"Firebase status: {status}") # Sürekli log basmaması için kapattım
        return status == "yes"
    except Exception as e:
        print(f"✗ Firebase okuma hatası: {e}")
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

def send_plates_to_firebase(plates_data):
    try:
        ref = db.reference('test/detected_plates')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        data = {
            timestamp: {
                'total_plates': len(plates_data),
                'detection_time': datetime.now().isoformat(),
                'plates': plates_data
            }
        }
        
        ref.update(data)
        print(f"✓ {len(plates_data)} plaka Firebase'e gönderildi")
        
        status_ref = db.reference('test/status')
        status_ref.set("no")
        print("✓ Status 'no' olarak güncellendi")
        return True
    except Exception as e:
        print(f"✗ Firebase yazma hatası: {e}")
        return False

# ==================== MODEL VE OCR ====================
def load_rfdetr_model():
    try:
        from ultralytics import YOLO
        # Plaka tespiti için eğitilmiş modelinizi buraya yazın
        # Eğer genel bir modelse 'yolov8n.pt' aracı bulur, plakayı değil.
        # Plaka için özel eğitilmiş .pt dosyası önerilir.
        model = YOLO('yolov8n.pt') 
        print("✓ Model yüklendi")
        return model
    except Exception as e:
        print(f"✗ Model yükleme hatası: {e}")
        return None

def initialize_ocr():
    try:
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print("✓ OCR başlatıldı")
        return reader
    except Exception as e:
        print(f"✗ OCR başlatma hatası: {e}")
        return None

def clean_plate_text(text):
    cleaned = ''.join(c for c in text if c.isalnum())
    # Türk plaka standartlarına göre filtreleme (opsiyonel gevşetilebilir)
    if len(cleaned) >= 5 and len(cleaned) <= 9:
        return cleaned.upper()
    return None

# ==================== VİDEO İŞLEME (GÜNCELLENMİŞ TRACKING) ====================
def process_video_and_detect_plates(model, ocr_reader, video_path, duration):
    """DEBUG VERSİYONU: Detaylı loglama ve görsel kaydetme içerir"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"✗ Video açılamadı: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(fps * duration)
    
    # --- TRACKING DEĞİŞKENLERİ ---
    detected_objects = [] 
    object_counter = 0     
    IOU_THRESHOLD = 0.2    
    FRAME_DROPOUT = 30     
    
    frame_count = 0
    start_time = time.time()
    
    print(f"Video işleniyor... (~{total_frames} frame)")
    
    # DEBUG: İlk tespit edilen kareyi kaydetmek için bayrak
    debug_image_saved = False
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time() - start_time
        
        if frame_count % 5 == 0:
            # DEBUG: conf eşiğini biraz düşürdüm (0.25) ki her şeyi görelim
            results = model(frame, conf=0.25, verbose=False)
            
            for result in results:
                boxes = result.boxes
                
                # DEBUG: Hiç kutu bulundu mu?
                if len(boxes) > 0 and not debug_image_saved:
                    print(f"  [DEBUG] Frame {frame_count}'te {len(boxes)} nesne tespit edildi.")
                    # İlk tespit edilen karenin resmini kaydet (Neyi kutu içine alıyor görelim)
                    debug_frame = frame.copy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0]) # Sınıf ID'si (2=Araba, 0=Kişi vb.)
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(debug_frame, f"Class: {cls}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    
                    cv2.imwrite("debug_tespit.jpg", debug_frame)
                    print(f"  [DEBUG] 'debug_tespit.jpg' dosyası kaydedildi. Lütfen kontrol edin!")
                    debug_image_saved = True

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    current_box = [x1, y1, x2, y2]
                    
                    # --- TRACKING ---
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
                    # Kenar kontrolünü geçici olarak kapatalım (sorun bu mu diye)
                    # if x1 < 10 or y1 < 10 ... : continue

                    plate_roi = frame[y1:y2, x1:x2]
                    if plate_roi.size == 0: continue

                    try:
                        # DEBUG: OCR ne okuyor?
                        ocr_results = ocr_reader.readtext(plate_roi)
                        if len(ocr_results) > 0:
                            # OCR bir şeyler bulduysa yazdır
                            print(f"    [OCR Okudu] Ham metin: {ocr_results[0][1]} (Güven: {ocr_results[0][2]:.2f})")
                        
                        for detection in ocr_results:
                            text = detection[1]
                            ocr_conf = detection[2]
                            
                            if ocr_conf > OCR_CONFIDENCE_THRESHOLD:
                                cleaned_text = clean_plate_text(text)
                                
                                # DEBUG: Temizleme sonrası metin
                                # print(f"      [Temizlendi] {cleaned_text}")
                                
                                if cleaned_text:
                                    if ocr_conf > matched_obj['best_conf']:
                                        matched_obj['best_plate'] = cleaned_text
                                        matched_obj['best_conf'] = ocr_conf
                                        print(f"  ✓ [Araç #{matched_obj['id']}] PLAKA BULUNDU: {cleaned_text}")

                    except Exception as e:
                        print(f"OCR Hatası: {e}")
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  İlerleme: %{(frame_count / total_frames) * 100:.1f}")
    
    cap.release()
    
    final_results = []
    print("\nSONUÇ RAPORU")
    for obj in detected_objects:
        if obj['best_plate']:
            final_results.append({
                'plate': obj['best_plate'],
                'confidence': round(obj['best_conf'] * 100, 2),
                'detection_confidence': 99.0,
                'time_in_video': obj['detection_time'],
                'frame': obj['last_seen_frame'],
                'vehicle_id': obj['id']
            })
            
    return final_results

# ==================== ANA PROGRAM ====================
def main():
    print("=" * 60)
    print("RF-DETR PLAKA OKUMA SİSTEMİ (TRACKING + IOU)")
    print("=" * 60)
    
    if not initialize_firebase():
        return
    
    model = load_rfdetr_model()
    if model is None: return
    
    ocr_reader = initialize_ocr()
    if ocr_reader is None: return
    
    print("\nSistem hazır. Firebase'den sinyal bekleniyor...")
    
    while True:
        try:
            if check_firebase_status():
                print("\n🔊 SES ALGILANDI! İşlem başlıyor...")
                
                local_video_path = download_video_from_storage(VIDEO_PATH)
                
                if local_video_path is None:
                    print("✗ Video hatası, pas geçiliyor")
                    db.reference('test/status').set("yes") # Tekrar denesin diye veya hata durumu
                    time.sleep(5)
                    continue
                
                # İşlemi başlat
                plates_data = process_video_and_detect_plates(
                    model, 
                    ocr_reader, 
                    local_video_path, 
                    VIDEO_DURATION
                )
                
                # Geçici dosyayı sil
                try:
                    os.remove(local_video_path)
                except:
                    pass
                
                # Sonuçları gönder
                if plates_data:
                    send_plates_to_firebase(plates_data)
                else:
                    print("⚠ Hiç plaka tespit edilemedi. Status resetleniyor.")
                    db.reference('test/status').set("no")
                
                print("\nİşlem bitti. Beklemede...\n")
            
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\nÇıkış yapılıyor...")
            break
        except Exception as e:
            print(f"Genel Hata: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()