import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yolov8m.pt"  )  
image_path = "3.png"  
image = cv2.imread(image_path)

# Détection des objets
results = model(image , conf = 0.5 )

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  
        label = model.names[int(box.cls[0])]  
        confidence = float(box.conf[0]) 

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.axis("off")  # Hide axes
plt.show()

