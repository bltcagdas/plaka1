import torch
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase

# 📌 Model ve sınıf listesi
model_path = "checkpoint_best_regular.pth"
CLASS_NAMES = ["License_Plate"]

# ⚠ RFDETRBase init args
model = RFDETRBase(
    pretrain_weights=model_path,
    num_classes=len(CLASS_NAMES),
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 📌 Görüntüyü oku
image_path = "example_licenseplate2.jpg"
image = Image.open(image_path).convert("RGB")

# ⚡ Inference
detections = model.predict(image, threshold=0.35)

# 🏷 Label listesi
labels = [
    f"{CLASS_NAMES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

print("Detected:", labels)

# 📌 Annotate
annotator_boxes = sv.BoxAnnotator()
annotator_labels = sv.LabelAnnotator()

# Swap detections to supervision format if needed
annotated_image = image.copy()
annotated_image = annotator_boxes.annotate(annotated_image, detections)
annotated_image = annotator_labels.annotate(annotated_image, detections, labels)

# 📤 Çıktı kaydet
output_path = "output_1.jpg"
annotated_image.save(output_path)
print("Annotated image saved at:", output_path)
