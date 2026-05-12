# -*- coding: utf-8 -*-
"""
Created on Sun May 25 22:54:32 2025

@author: Jasmi
"""

from inference import get_model
import supervision as sv
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# === Configuration ===
IMAGE_SIZE = (96, 96)
FRAME_RESOLUTION = (640, 640)  # You can try (416, 416) for speed/scale test
MOBILENET_MODEL_PATH = r"C:\Users\Jasmi\Desktop\CNN_LEGO_weights.h5"
CONFIDENCE_THRESHOLD = 0.5
YOLO_CONFIDENCE = 0.3
BBOX_PADDING = 5  # pixels

# === Load Models ===
yolo_model = get_model(model_id="ldc-lego-yolo-5wyps/1", api_key="w5RGdO0YzK6pvmC3EoRx")
mobilenet_model = load_model(MOBILENET_MODEL_PATH)

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

def preprocess_image(img):
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_patch(img):
    preprocessed = preprocess_image(img)
    preds = mobilenet_model.predict(preprocessed, verbose=0)
    class_id = np.argmax(preds[0])
    confidence = preds[0][class_id]
    return ("Unknown", confidence) if confidence < CONFIDENCE_THRESHOLD else (CLASS_LABELS[class_id], confidence)

def detect_color(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    avg_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
    h, s, v = avg_hsv
    if s < 50 and v > 200: return "WHITE"
    elif v < 50: return "BLACK"
    elif s < 50: return "GRAY"
    elif h < 10 or h > 160: return "RED"
    elif h < 21: return "ORANGE"
    elif h < 31: return "YELLOW"
    elif h < 86: return "GREEN"
    elif h < 96: return "CYAN"
    elif h < 126: return "BLUE"
    elif h < 146: return "VIOLET"
    elif h < 160: return "PINK"
    return "UNDEFINED"

# === Main Loop ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.resize(frame, FRAME_RESOLUTION)

    # Run YOLO
    results = yolo_model.infer(frame, confidence=YOLO_CONFIDENCE)[0]
    detections = sv.Detections.from_inference(results)

    for xyxy in detections.xyxy:
        x1, y1, x2, y2 = map(int, xyxy)

        # Pad box
        x1 = max(0, x1 - BBOX_PADDING)
        y1 = max(0, y1 - BBOX_PADDING)
        x2 = min(frame.shape[1], x2 + BBOX_PADDING)
        y2 = min(frame.shape[0], y2 + BBOX_PADDING)

        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            continue

        label, conf = classify_patch(patch)
        color = detect_color(patch)

        text = f"{label} ({conf*100:.1f}%) [{color}]"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("YOLO + MobileNet + Color", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
