"""
Kerem Berke YOLOv5 Plaka Modeli ile Plaka Okuma (Otomatik İndirmeli)
Tracking (IoU) + Firebase Entegrasyonu
try1.py

Ek: Video özet tablosu + Gerçek tespit güveni (YOLO conf) ve OCR güveni
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
import yolov5

# ==================== YAPLANDIRMA ====================
FIREBASE_CRED_PATH = "firebase-credentials.json"
FIREBASE_DB_URL = "https://try1-cc8eb-default-rtdb.europe-west1.firebasedatabase.app/"
FIREBASE_STORAGE_BUCKET = "try1-cc8eb.firebasestorage.app"

# VIDEO_PATH = "video2.mp4"
VIDEO_PATH = "traffic_video.mp4"
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
    except Exception:
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
        model = yolov5.load('keremberke/yolov5m-license-plate')

        # Model Ayarları
        model.conf = CONFIDENCE_THRESHOLD
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
        print(f"✓ OCR başlatıldı (GPU: {torch.cuda.is_available()})")
        return reader
    except Exception as e:
        print(f"✗ OCR hatası: {e}")
        return None

def clean_plate_text(text):
    cleaned = ''.join(c for c in text if c.isalnum())
    if 5 <= len(cleaned) <= 9:
        return cleaned.upper()
    return None

# ==================== VİDEO META + TABLO ====================
def get_video_metadata(local_video_path, storage_video_path, requested_duration):
    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        return {
            "video_name": os.path.basename(storage_video_path),
            "storage_path": storage_video_path,
            "requested_duration_sec": requested_duration,
            "video_duration_sec": None,
            "processed_duration_sec": requested_duration,
            "fps": None,
            "total_frames": None,
            "resolution": None,
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    resolution = f"{width}x{height}" if width > 0 and height > 0 else None

    video_duration_sec = (total_frames / fps) if fps > 0 else None

    processed_duration_sec = requested_duration
    if video_duration_sec is not None:
        processed_duration_sec = min(requested_duration, video_duration_sec)

    planned_frames = int(float(fps) * float(processed_duration_sec)) if fps and processed_duration_sec else None

    cap.release()

    return {
        "video_name": os.path.basename(storage_video_path),
        "storage_path": storage_video_path,
        "requested_duration_sec": requested_duration,
        "video_duration_sec": round(video_duration_sec, 2) if video_duration_sec else None,
        "processed_duration_sec": round(processed_duration_sec, 2),
        "fps": round(float(fps), 3) if fps else None,
        "total_frames": total_frames,
        "resolution": resolution,
        "planned_frames": planned_frames,
    }

def print_video_summary_table(video_meta: dict, stats: dict):
    """
    stats:
      - plates_count
      - max_det_conf_pct
      - avg_det_conf_pct
      - max_ocr_conf_pct
      - avg_ocr_conf_pct
    """
    headers = [
        "Video", "Süre (sn)", "Çözünürlük", "Plaka Sayısı",
        "Max YOLO(%)", "Ort YOLO(%)", "Max OCR(%)", "Ort OCR(%)"
    ]

    row = [
        video_meta.get("video_name") or "-",
        str(video_meta.get("processed_duration_sec") if video_meta.get("processed_duration_sec") is not None else "-"),
        video_meta.get("resolution") or "-",
        str(stats.get("plates_count", 0)),
        f"{stats.get('max_det_conf_pct', 0.0):.2f}",
        f"{stats.get('avg_det_conf_pct', 0.0):.2f}",
        f"{stats.get('max_ocr_conf_pct', 0.0):.2f}",
        f"{stats.get('avg_ocr_conf_pct', 0.0):.2f}",
    ]

    col_widths = [max(len(headers[i]), len(str(row[i]))) for i in range(len(headers))]

    def fmt_line(values):
        return " | ".join(str(values[i]).ljust(col_widths[i]) for i in range(len(values)))

    line_len = sum(col_widths) + 3 * (len(headers) - 1)
    print("\n" + "=" * line_len)
    print(fmt_line(headers))
    print("-" * line_len)
    print(fmt_line(row))
    print("=" * line_len + "\n")

# ==================== VİDEO İŞLEME ====================
def process_video_and_detect_plates(model, ocr_reader, video_path, duration):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Video açılamadı: {video_path}")
        return [], {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        fps = 25.0

    total_frames = int(fps * duration)

    detected_objects = []
    object_counter = 0
    IOU_THRESHOLD = 0.2
    FRAME_DROPOUT = 30

    frame_count = 0
    start_time = time.time()

    print(f"Video işleniyor... (~{total_frames} frame)")

    debug_saved = False

    # İstatistik için: tüm tespit conf'ları (YOLO) ve OCR conf'ları
    all_det_confs = []
    all_ocr_confs = []

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time() - start_time

        if frame_count % 5 == 0:
            results = model(frame)
            detections = results.xyxy[0].cpu().numpy()

            if len(detections) > 0 and not debug_saved:
                debug_frame = frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = map(int, det[:4])
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.imwrite("debug_yolov5_plaka.jpg", debug_frame)
                print("[BİLGİ] 'debug_yolov5_plaka.jpg' kaydedildi. Plakaları doğru görüyor mu kontrol et!")
                debug_saved = True

            for det in detections:
                x1, y1, x2, y2 = map(int, det[:4])
                det_conf = float(det[4])  # ✅ YOLO'nun gerçek confidence değeri

                if det_conf < CONFIDENCE_THRESHOLD:
                    continue

                all_det_confs.append(det_conf)

                current_box = [x1, y1, x2, y2]

                # --- TRACKING ---
                matched_obj = None
                for obj in detected_objects:
                    if frame_count - obj['last_seen_frame'] > FRAME_DROPOUT:
                        continue

                    if calculate_iou(current_box, obj['last_box']) > IOU_THRESHOLD:
                        matched_obj = obj
                        obj['last_box'] = current_box
                        obj['last_seen_frame'] = frame_count
                        # ✅ aynı obje ise, daha yüksek YOLO conf gördüysek güncelle
                        if det_conf > obj['best_det_conf']:
                            obj['best_det_conf'] = det_conf
                        break

                if matched_obj is None:
                    object_counter += 1
                    matched_obj = {
                        'id': object_counter,
                        'best_plate': None,
                        'best_conf': 0.0,        # OCR conf
                        'best_det_conf': det_conf,  # ✅ YOLO conf
                        'last_box': current_box,
                        'last_seen_frame': frame_count,
                        'detection_time': round(current_time, 2)
                    }
                    detected_objects.append(matched_obj)

                # --- OCR ---
                h, w, _ = frame.shape
                if x1 < 5 or y1 < 5 or x2 > w - 5 or y2 > h - 5:
                    continue

                plate_roi = frame[y1:y2, x1:x2]
                if plate_roi.size == 0:
                    continue

                try:
                    ocr_results = ocr_reader.readtext(plate_roi)
                    for detection in ocr_results:
                        text = detection[1]
                        ocr_conf = detection[2]

                        if ocr_conf > OCR_CONFIDENCE_THRESHOLD:
                            all_ocr_confs.append(ocr_conf)

                            cleaned_text = clean_plate_text(text)
                            if cleaned_text and ocr_conf > matched_obj['best_conf']:
                                prev = matched_obj['best_plate']
                                matched_obj['best_plate'] = cleaned_text
                                matched_obj['best_conf'] = ocr_conf

                                if prev != cleaned_text:
                                    print(f"  [Araç {matched_obj['id']}] Plaka: {cleaned_text} (OCR: %{ocr_conf*100:.0f}, YOLO: %{matched_obj['best_det_conf']*100:.0f})")
                except Exception:
                    pass

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  İlerleme: %{(frame_count / total_frames) * 100:.1f}")

    cap.release()

    final_results = []
    print("\n" + "=" * 30)
    print("SONUÇLAR")
    for obj in detected_objects:
        if obj['best_plate']:
            res = {
                'plate': obj['best_plate'],
                'confidence': round(obj['best_conf'] * 100, 2),                  # OCR %
                'detection_confidence': round(obj['best_det_conf'] * 100, 2),    # ✅ YOLO %
                'time_in_video': obj['detection_time'],
                'frame': obj['last_seen_frame'],
                'vehicle_id': obj['id']
            }
            final_results.append(res)
            print(f"Araç #{obj['id']} -> {res['plate']} (YOLO: %{res['detection_confidence']:.0f}, OCR: %{res['confidence']:.0f})")
    print("=" * 30)

    # ✅ tablo için istatistik
    def safe_avg(arr):
        return float(np.mean(arr)) if arr else 0.0

    stats = {
        "plates_count": len(final_results),
        "max_det_conf_pct": (max(all_det_confs) * 100.0) if all_det_confs else 0.0,
        "avg_det_conf_pct": (safe_avg(all_det_confs) * 100.0) if all_det_confs else 0.0,
        "max_ocr_conf_pct": (max(all_ocr_confs) * 100.0) if all_ocr_confs else 0.0,
        "avg_ocr_conf_pct": (safe_avg(all_ocr_confs) * 100.0) if all_ocr_confs else 0.0,
    }

    return final_results, stats

# ==================== ANA PROGRAM ====================
def main():
    print("--- KEREM BERKE YOLOv5 PLAKA OKUMA ---")

    if not initialize_firebase():
        return

    model = load_plate_model()
    if model is None:
        return

    ocr_reader = initialize_ocr()
    if ocr_reader is None:
        return

    print("\nSistem hazır. Firebase'den sinyal bekleniyor...")

    while True:
        try:
            if check_firebase_status():
                print("\n🔊 Sinyal Geldi!")
                local_video_path = download_video_from_storage(VIDEO_PATH)

                if local_video_path:
                    video_meta = get_video_metadata(
                        local_video_path=local_video_path,
                        storage_video_path=VIDEO_PATH,
                        requested_duration=VIDEO_DURATION
                    )

                    plates, stats = process_video_and_detect_plates(
                        model, ocr_reader, local_video_path, VIDEO_DURATION
                    )

                    # ✅ Tabloyu iş bittikten sonra basıyoruz (conf'lar artık elimizde)
                    print_video_summary_table(video_meta, stats)

                    try:
                        os.remove(local_video_path)
                    except:
                        pass

                    if plates:
                        send_plates_to_firebase(plates, video_meta=video_meta)
                    else:
                        print("⚠ Plaka okunamadı.")
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
