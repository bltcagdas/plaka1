"""
RT-DETRv2 ile Plaka Tespiti + Tracking (IoU) + EasyOCR + Firebase
YOLOv5 yerine RT-DETRv2 kullanır.

ÖNEMLİ:
- Plaka için eğitilmiş .pth checkpoint gereklidir (license plate class).
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
import sys

# ==================== YAPLANDIRMA ====================
FIREBASE_CRED_PATH = "firebase-credentials.json"
FIREBASE_DB_URL = "https://try1-cc8eb-default-rtdb.europe-west1.firebasedatabase.app/"
FIREBASE_STORAGE_BUCKET = "try1-cc8eb.firebasestorage.app"

VIDEO_PATH = "traffic_video.mp4"   # Firebase Storage path
VIDEO_DURATION = 8

CONFIDENCE_THRESHOLD = 0.25      # Plaka tespiti için güven eşiği
OCR_CONFIDENCE_THRESHOLD = 0.4   # OCR okuma güven eşiği

# RT-DETRv2 ayarları
RTDETR_REPO_DIR = "./RT-DETR"  # sende bu klasör var
RTDETR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Plaka için eğitilmiş ağırlık dosyan (örnek):
# weights/rtdetrv2_plate.pth gibi
RTDETR_CHECKPOINT_PATH = "./weights/rtdetrv2_plate.pth"

# Eğer model tek sınıf (plate) ise class id genelde 0 olur.
PLATE_CLASS_ID = 0


# ==================== YARDIMCI MATEMATİK FONKSİYONLARI ====================
def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / float(box1_area + box2_area - intersection_area)


# ==================== FIREBASE ====================
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


# ==================== OCR ====================
def initialize_ocr():
    try:
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print(f"✓ OCR başlatıldı (GPU: {torch.cuda.is_available()})")
        return reader
    except Exception as e:
        print(f"✗ OCR hatası: {e}")
        return None

def clean_plate_text(text: str):
    cleaned = ''.join(c for c in text if c.isalnum())
    if 5 <= len(cleaned) <= 9:
        return cleaned.upper()
    return None


# ==================== RT-DETRv2 MODEL ====================
def load_rtdetrv2_model():
    """
    Local repo'daki hubconf.py üzerinden RT-DETRv2'yi yüklemeye çalışır.
    Repo entrypoint isimleri farklı olabilir. Bu yüzden önce liste bastırıyoruz.
    """
    if not os.path.isdir(RTDETR_REPO_DIR):
        raise FileNotFoundError(f"RT-DETR klasörü bulunamadı: {RTDETR_REPO_DIR}")

    # torch.hub local source kullanabilmek için
    print("⏳ RT-DETRv2 model hazırlanıyor...")

    # Mevcut entrypoint'leri listele (ilk çalıştırmada çok faydalı)
    try:
        eps = torch.hub.list(RTDETR_REPO_DIR, source="local")
        print("✓ RT-DETR repo hub entrypoints:", eps)
    except Exception as e:
        print("⚠ torch.hub.list çalışmadı. hubconf.py uyumsuz olabilir:", e)

    # Burada repo'daki gerçek entrypoint adına göre değiştirmen gerekebilir.
    # Örnek isimler: 'rtdetrv2_r18vd', 'rtdetrv2_r50vd' gibi olabilir.
    # list() çıktısında gördüğün birini yaz.
    entrypoint_name = None

    # Basit otomatik seçim: list içinden rtdetrv2 geçen ilk şeyi al
    try:
        eps = torch.hub.list(RTDETR_REPO_DIR, source="local")
        for n in eps:
            if "rtdetrv2" in n.lower():
                entrypoint_name = n
                break
    except Exception:
        pass

    if entrypoint_name is None:
        # fallback: kullanıcı elle güncellesin
        entrypoint_name = "rtdetrv2_r50vd"  # BUNU list() çıktına göre düzelt

    print(f"→ Seçilen entrypoint: {entrypoint_name}")

    try:
        model = torch.hub.load(
            RTDETR_REPO_DIR,
            entrypoint_name,
            source="local",
            pretrained=False  # plaka checkpoint yükleyeceğiz
        )
    except Exception as e:
        raise RuntimeError(
            f"RT-DETRv2 torch.hub.load başarısız: {e}\n"
            f"Çözüm: torch.hub.list({RTDETR_REPO_DIR}) çıktısından doğru entrypoint'i seç."
        )

    # checkpoint yükle (plaka eğitimi)
    if not os.path.isfile(RTDETR_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Plaka checkpoint bulunamadı: {RTDETR_CHECKPOINT_PATH}\n"
            f"weights klasörüne .pth koy ve RTDETR_CHECKPOINT_PATH'i güncelle."
        )

    ckpt = torch.load(RTDETR_CHECKPOINT_PATH, map_location="cpu")
    # Bazı checkpoint'ler {'model': state_dict} taşır
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)

    model.to(RTDETR_DEVICE)
    model.eval()
    print(f"✓ RT-DETRv2 yüklendi (device={RTDETR_DEVICE})")
    return model


def rtdetrv2_predict_boxes(model, frame_bgr, conf_thres=0.25):
    """
    RT-DETR tarzı çıktılardan bbox çıkaran genel bir postprocess.
    Repo'ya göre output isimleri değişebilir; en yaygın DETR formatlarına göre yazıldı.

    Çıktı: list of (x1,y1,x2,y2,score)
    """
    h, w = frame_bgr.shape[:2]

    # BGR->RGB, normalize (yaygın DETR pratikleri)
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    # tensor: (1,3,H,W)
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(RTDETR_DEVICE)

    with torch.no_grad():
        out = model(x)

    # out bazı repolarda dict olur, bazılarında tuple/list
    # Yaygın: out['pred_logits'], out['pred_boxes']
    if isinstance(out, (list, tuple)) and len(out) == 1:
        out = out[0]

    if isinstance(out, dict):
        pred_logits = out.get("pred_logits", None)
        pred_boxes = out.get("pred_boxes", None)
    else:
        # repo çok farklıysa burada patlar; debug için print ekleyebilirsin
        raise RuntimeError(f"Beklenmeyen model çıktısı tipi: {type(out)}")

    if pred_logits is None or pred_boxes is None:
        raise RuntimeError("Model çıktısında pred_logits/pred_boxes yok. Repo postprocess farklı olabilir.")

    # pred_logits: (1, N, C+1) (son sınıf genelde 'no-object')
    # pred_boxes: (1, N, 4) genelde normalized cxcywh
    logits = pred_logits[0]
    boxes = pred_boxes[0]

    probs = torch.softmax(logits, dim=-1)

    # no-object class'ı genelde son index
    scores, labels = probs[:, :-1].max(dim=-1)

    # sadece plaka class
    keep = (labels == PLATE_CLASS_ID) & (scores >= conf_thres)
    scores = scores[keep]
    boxes = boxes[keep]

    results = []

    # cxcywh (0-1) -> xyxy pixel
    for b, s in zip(boxes, scores):
        cx, cy, bw, bh = b.tolist()
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h

        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2)))
        y2 = int(max(0, min(h - 1, y2)))

        if x2 <= x1 or y2 <= y1:
            continue

        results.append((x1, y1, x2, y2, float(s)))

    return results


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
    headers = [
        "Video", "Süre (sn)", "Çözünürlük", "Plaka Sayısı",
        "Max DET(%)", "Ort DET(%)", "Max OCR(%)", "Ort OCR(%)"
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

    all_det_confs = []
    all_ocr_confs = []

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time() - start_time

        if frame_count % 5 == 0:
            # ===== RT-DETRv2 DETECT =====
            detections = rtdetrv2_predict_boxes(model, frame, conf_thres=CONFIDENCE_THRESHOLD)

            if len(detections) > 0 and not debug_saved:
                debug_frame = frame.copy()
                for (x1, y1, x2, y2, s) in detections:
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(debug_frame, f"{s:.2f}", (x1, max(0, y1-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imwrite("debug_rtdetrv2_plaka.jpg", debug_frame)
                print("[BİLGİ] 'debug_rtdetrv2_plaka.jpg' kaydedildi. Plakaları doğru görüyor mu kontrol et!")
                debug_saved = True

            for (x1, y1, x2, y2, det_conf) in detections:
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
                        if det_conf > obj['best_det_conf']:
                            obj['best_det_conf'] = det_conf
                        break

                if matched_obj is None:
                    object_counter += 1
                    matched_obj = {
                        'id': object_counter,
                        'best_plate': None,
                        'best_conf': 0.0,          # OCR conf
                        'best_det_conf': det_conf, # DET conf
                        'last_box': current_box,
                        'last_seen_frame': frame_count,
                        'detection_time': round(current_time, 2)
                    }
                    detected_objects.append(matched_obj)

                # --- OCR ---
                h, w = frame.shape[:2]
                if x1 < 2 or y1 < 2 or x2 > w - 2 or y2 > h - 2:
                    continue

                plate_roi = frame[y1:y2, x1:x2]
                if plate_roi.size == 0:
                    continue

                try:
                    ocr_results = ocr_reader.readtext(plate_roi)
                    for detection in ocr_results:
                        text = detection[1]
                        ocr_conf = float(detection[2])

                        if ocr_conf > OCR_CONFIDENCE_THRESHOLD:
                            all_ocr_confs.append(ocr_conf)
                            cleaned_text = clean_plate_text(text)
                            if cleaned_text and ocr_conf > matched_obj['best_conf']:
                                prev = matched_obj['best_plate']
                                matched_obj['best_plate'] = cleaned_text
                                matched_obj['best_conf'] = ocr_conf

                                if prev != cleaned_text:
                                    print(f"  [Araç {matched_obj['id']}] Plaka: {cleaned_text} "
                                          f"(OCR: %{ocr_conf*100:.0f}, DET: %{matched_obj['best_det_conf']*100:.0f})")
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
                'confidence': round(obj['best_conf'] * 100, 2),               # OCR %
                'detection_confidence': round(obj['best_det_conf'] * 100, 2), # DET %
                'time_in_video': obj['detection_time'],
                'frame': obj['last_seen_frame'],
                'vehicle_id': obj['id']
            }
            final_results.append(res)
            print(f"Araç #{obj['id']} -> {res['plate']} (DET: %{res['detection_confidence']:.0f}, OCR: %{res['confidence']:.0f})")

    print("=" * 30)

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


# ==================== ANA ====================
def main():
    print("--- RT-DETRv2 PLAKA OKUMA ---")

    if not initialize_firebase():
        return

    try:
        model = load_rtdetrv2_model()
    except Exception as e:
        print("✗ RT-DETRv2 yüklenemedi:", e)
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
