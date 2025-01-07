import cv2
from ultralytics import YOLO
import os




# OpenVINO modelini yükle
model = YOLO('/kaynak/best.pt')
model.export(format="openvino") 
model= YOLO('best_openvino_model')

# Test görüntüleri için klasör yolu
img_folder = 'val' 

# Çıktıları kaydetmek için klasör yolu
output_folder = 'val_output'

for img_name in os.listdir(img_folder):
    # Görüntüyü oku
    img_path = os.path.join(img_folder, img_name)
    img = cv2.imread(img_path)

    # Tespit işlemini gerçekleştir
    results = model(img)

    # Sonuçları görselleştir
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Sınırlayıcı kutu koordinatlarını al
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Güven skorunu al
            confidence = box.conf[0]

            # Etiket adını al
            cls = int(box.cls[0])
            label = model.names[cls]  # model.names etiket adlarını içerir

            # Sınırlayıcı kutuyu çiz
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Güven skorunu ve etiket adını yazdır
            text = f'{label} {confidence:.2f}'
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Çıktı görüntüsünü kaydet
    output_path = os.path.join(output_folder, img_name)
    cv2.imwrite(output_path, img)