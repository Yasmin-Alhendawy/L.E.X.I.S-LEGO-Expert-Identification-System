# -*- coding: utf-8 -*-
"""
Created on Sun May 25 18:06:11 2025

@author: Jasmi
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === Configuration ===
MODEL_PATH = r"C:\Users\Jasmi\Desktop\new testing code\mobilenetv2_classifier_20250517_213548.h5"
IMAGE_SIZE = (96, 96)
CONFIDENCE_THRESHOLD = 0.5

# === Class Labels ===
CLASS_LABELS = [
    "(11214) Technic Axle Pin 3L with Friction Ridges Lengthwise and 1L Axle",
    "(15068) 2×2 Curved",
    "(15573) 1×2 Jumper",
    "(2445) 2×12 Plate",
    "(3004) 1×2 Brick",
    "(3005) 1×1 Brick",
    "(3008) 1×8 Brick",
    "(3022) 2×2 Plate",
    "(30414) Brick Special 1 x 4 with 4 Studs on One Side",
    "(32062) Technic Axle 2 Notched",
    "(32607) Plant, Plate 1 x 1 Round with 3 Leaves",
    "(3460) 1×8 Plate",
    "(3623) 1×3 Plate",
    "(3832) 2×10 Plate",
    "(4032) 2×2 Plate, Round w Axle",
    "(4286) Slope 33 3 x 1",
    "(44728) 2×2 Bracket",
    "(4477) 1×10 Plate",
    "(49668) Plate Special 1 x 1 with Tooth",
    "(60478) 1×2 Plate w Handle, End",
    "(87079) 2×4 Tile",
    "(92280) 1×2 Plate w Clip, Top",
    "(98138) 1×1 Tile, Round"
]

# === Load Model ===
model = load_model(MODEL_PATH)

# === Helper ===
def classify_frame(frame):
    img = cv2.resize(frame, IMAGE_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]
    class_id = np.argmax(preds)
    confidence = preds[class_id]
    if confidence < CONFIDENCE_THRESHOLD:
        return "Unknown", confidence
    return CLASS_LABELS[class_id], confidence

# === Webcam Loop ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    label, conf = classify_frame(frame)
    text = f"{label} ({conf * 100:.1f}%)"

    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("MobileNetV2 Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
