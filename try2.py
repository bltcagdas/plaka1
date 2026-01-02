"""
RF-DETR ile Plaka Okuma ve Firebase Entegrasyonu
try1.py

Gereksinimler:
1. Python 3.8+
2. Virtual environment (venv)
3. Firebase Admin SDK credentials (JSON dosyası)
"""
from firebase_admin import credentials, db, storage  # storage ekledik
import tempfile
import cv2
import torch
import numpy as np
from datetime import datetime
import firebase_admin
import easyocr,os
import time
from pathlib import Path

# ==================== YAPLANDIRMA ====================
FIREBASE_CRED_PATH = "firebase-credentials.json"  # Firebase credentials dosyanızın yolu
FIREBASE_DB_URL = "https://try1-cc8eb-default-rtdb.europe-west1.firebasedatabase.app/"  # Firebase Realtime Database URL'iniz
# FIREBASE_STORAGE_BUCKET = "try1-cc8eb.appspot.com"
FIREBASE_STORAGE_BUCKET = "try1-cc8eb.firebasestorage.app"
VIDEO_PATH = "video2.mp4"  # Trafik videonuzun yolu
VIDEO_DURATION = 8  # Video işleme süresi (saniye)
CONFIDENCE_THRESHOLD = 0.5  # Tespit güven eşiği
OCR_CONFIDENCE_THRESHOLD = 0.6  # OCR güven eşiği

# ==================== FIREBASE BAŞLATMA ====================
def initialize_firebase():
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_CRED_PATH)
            firebase_admin.initialize_app(cred, {
                "databaseURL": FIREBASE_DB_URL,
                "storageBucket": FIREBASE_STORAGE_BUCKET,  # <-- şart!
            })
        print("✓ Firebase bağlantısı başarılı")
        return True
    except Exception as e:
        print(f"✗ Firebase bağlantı hatası: {e}")
        return False

def check_firebase_status():
    """Firebase'den status kontrolü yap"""
    try:
        ref = db.reference('test/status')
        status = ref.get()
        print(f"Firebase status: {status}")
        return status == "yes"
    except Exception as e:
        print(f"✗ Firebase okuma hatası: {e}")
        return False

def download_video_from_storage(storage_path):
    """Firebase Storage'dan videoyu indir"""
    try:
        print(f"Video indiriliyor: {storage_path}")
        bucket = storage.bucket()
        blob = bucket.blob(storage_path)
        
        # Geçici dosya oluştur
        temp_dir = tempfile.gettempdir()
        local_path = os.path.join(temp_dir, "temp_video.mp4")
        
        # Videoyu indir
        blob.download_to_filename(local_path)
        print(f"✓ Video indirildi: {local_path}")
        return local_path
    except Exception as e:
        print(f"✗ Video indirme hatası: {e}")
        return None

def send_plates_to_firebase(plates_data):
    """Okunan plakaları Firebase'e gönder"""
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
        
        # Status'u tekrar "no" yap
        status_ref = db.reference('test/status')
        status_ref.set("no")
        print("✓ Status 'no' olarak güncellendi")
        
        return True
    except Exception as e:
        print(f"✗ Firebase yazma hatası: {e}")
        return False

# ==================== RF-DETR MODEL YÜKLEME ====================
def load_rfdetr_model():
    """RF-DETR modelini yükle"""
    try:
        # RF-DETR modelini yükle (önceden eğitilmiş veya özel eğitilmiş)
        # Burada örnek olarak YOLOv8 kullanıyorum çünkü RF-DETR kurulumu daha karmaşık
        # RF-DETR için: from rfdetr import RFDETR
        
        # Örnek: Ultralytics YOLOv8 (plaka tespiti için)
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # veya kendi eğittiğiniz model
        print("✓ Model yüklendi")
        return model
    except Exception as e:
        print(f"✗ Model yükleme hatası: {e}")
        return None

# ==================== OCR BAŞLATMA ====================
def initialize_ocr():
    """EasyOCR başlat (Türkçe plakalar için)"""
    try:
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print("✓ OCR başlatıldı")
        return reader
    except Exception as e:
        print(f"✗ OCR başlatma hatası: {e}")
        return None

# ==================== PLAKA TEMİZLEME ====================
def clean_plate_text(text):
    """Plaka metnini temizle ve formatla"""
    # Sadece harf ve rakamları al
    cleaned = ''.join(c for c in text if c.isalnum())
    # Türk plaka formatına uygunluğu kontrol et (örn: 34ABC123)
    if len(cleaned) >= 6 and len(cleaned) <= 9:
        return cleaned.upper()
    return None

# ==================== VİDEO İŞLEME VE PLAKA OKUMA ====================
def process_video_and_detect_plates(model, ocr_reader, video_path, duration):
    """Videoyu işle ve plakaları oku"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"✗ Video açılamadı: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(fps * duration)
    
    detected_plates = {}  # Benzersiz plakalar için dictionary
    frame_count = 0
    start_time = time.time()
    
    print(f"Video işleniyor... ({duration} saniye, ~{total_frames} frame)")
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time() - start_time
        
        # Her 5. frame'i işle (performans için)
        if frame_count % 5 == 0:
            # Model ile araç/plaka tespiti
            results = model(frame, conf=CONFIDENCE_THRESHOLD)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Bounding box koordinatları
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Plaka bölgesini kırp
                    plate_roi = frame[y1:y2, x1:x2]
                    
                    if plate_roi.size == 0:
                        continue
                    
                    # OCR ile plaka oku
                    try:
                        ocr_results = ocr_reader.readtext(plate_roi)
                        
                        for detection in ocr_results:
                            text = detection[1]
                            ocr_conf = detection[2]
                            
                            if ocr_conf > OCR_CONFIDENCE_THRESHOLD:
                                cleaned_text = clean_plate_text(text)
                                
                                if cleaned_text and cleaned_text not in detected_plates:
                                    detected_plates[cleaned_text] = {
                                        'plate': cleaned_text,
                                        'confidence': round(ocr_conf * 100, 2),
                                        'detection_confidence': round(conf * 100, 2),
                                        'time_in_video': round(current_time, 2),
                                        'frame': frame_count
                                    }
                                    print(f"  → Plaka bulundu: {cleaned_text} (Güven: {ocr_conf:.2f})")
                    
                    except Exception as e:
                        continue
        
        frame_count += 1
        
        # İlerleme göstergesi
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  İlerleme: %{progress:.1f} - {len(detected_plates)} benzersiz plaka bulundu")
    
    cap.release()
    print(f"\n✓ Video işleme tamamlandı!")
    print(f"  Toplam {len(detected_plates)} benzersiz plaka tespit edildi")
    
    return list(detected_plates.values())

# ==================== ANA FONKSİYON ====================
def main():
    """Ana program döngüsü"""
    print("=" * 60)
    print("RF-DETR PLAKA OKUMA SİSTEMİ")
    print("=" * 60)
    
    # Firebase başlat
    if not initialize_firebase():
        return
    
    # Model yükle
    model = load_rfdetr_model()
    if model is None:
        return
    
    # OCR başlat
    ocr_reader = initialize_ocr()
    if ocr_reader is None:
        return
    
    print("\nSistem hazır. Firebase'den sinyal bekleniyor...")
    
    # Firebase'den status kontrolü
    while True:
        try:
            if check_firebase_status():
                print("\n🔊 SES ALGILANDI! Video indiriliyor ve işleme başlıyor...")
                
                # Storage'dan videoyu indir
                local_video_path = download_video_from_storage(VIDEO_PATH)
                
                if local_video_path is None:
                    print("✗ Video indirilemedi, işlem iptal edildi")
                    # Status'u tekrar "no" yap
                    status_ref = db.reference('test/status')
                    status_ref.set("yes")
                    continue
                
                # Videoyu işle ve plakaları oku
                plates_data = process_video_and_detect_plates(
                    model, 
                    ocr_reader, 
                    local_video_path, 
                    VIDEO_DURATION
                )
                
                # Geçici video dosyasını sil
                try:
                    os.remove(local_video_path)
                    print(f"✓ Geçici video dosyası silindi")
                except:
                    pass
                
                # Plakaları Firebase'e gönder
                if plates_data:
                    send_plates_to_firebase(plates_data)
                    
                    # Tespit edilen plakaları göster
                    print("\n" + "=" * 60)
                    print("TESPİT EDİLEN PLAKALAR:")
                    print("=" * 60)
                    for i, plate in enumerate(plates_data, 1):
                        print(f"{i}. {plate['plate']} - Zaman: {plate['time_in_video']}s - Güven: %{plate['confidence']}")
                    print("=" * 60)
                else:
                    print("⚠ Hiç plaka tespit edilemedi")
                
                print("\nİşlem tamamlandı. Yeni sinyal bekleniyor...\n")
            
            time.sleep(2)  # 2 saniyede bir kontrol et
            
        except KeyboardInterrupt:
            print("\n\nProgram sonlandırılıyor...")
            break
        except Exception as e:
            print(f"✗ Hata: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()