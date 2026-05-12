# -*- coding: utf-8 -*-
"""
Created on Tue May 20 12:31:10 2025

@author: Jasmi
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import tensorflow as tf
from datetime import datetime
import csv
import os
from collections import defaultdict

# === Configuration ===
IMAGE_SIZE = (96, 96)
YOLO_MODEL_PATH = r"C:\Users\Jasmi\Desktop\Machine project\Useful codes\Yolocodes\best.pt"
MOBILENET_MODEL_PATH = r"C:\Users\Jasmi\Desktop\new testing code\mobilenetv2_classifier_20250517_213548.h5"
CONFIDENCE_THRESHOLD = 0.5
EXPORT_CSV = "lego_detections.csv"

# === Load Models ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model.to(device)
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

# === Helpers ===
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
    if confidence < CONFIDENCE_THRESHOLD:
        return "Unknown", confidence
    return CLASS_LABELS[class_id], confidence

def detect_color(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    avg_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
    h, s, v = avg_hsv

    if s < 50 and v > 200:
        return "WHITE"
    elif v < 50:
        return "BLACK"
    elif s < 50:
        return "GRAY"
    elif h < 10 or h > 160:
        return "RED"
    elif h < 21:
        return "ORANGE"
    elif h < 31:
        return "YELLOW"
    elif h < 86:
        return "GREEN"
    elif h < 96:
        return "CYAN"
    elif h < 126:
        return "BLUE"
    elif h < 146:
        return "VIOLET"
    elif h < 160:
        return "PINK"
    return "UNDEFINED"

def save_to_csv(row):
    file_exists = os.path.isfile(EXPORT_CSV)
    with open(EXPORT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Class", "Confidence", "Color", "x1", "y1", "x2", "y2"])
        writer.writerow(row)

# === Main Loop ===
def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not detected!")
        return

    print("Press 'q' to quit.")
    total_counts = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.predict(source=frame, device=device, conf=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

        frame_counts = defaultdict(int)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            label, conf = classify_patch(patch)
            color = detect_color(patch)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Update counts
            frame_counts[label] += 1
            total_counts[label] += 1

            # Save to CSV
            save_to_csv([timestamp, label, f"{conf:.4f}", color, x1, y1, x2, y2])

            # Draw on frame
            text = f"{label} ({conf*100:.1f}%) [{color}]"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display class-wise counts
        y_offset = 20
        for label, count in frame_counts.items():
            cv2.putText(frame, f"{label}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20

        # Display total count
        cv2.putText(frame, f"Total: {len(boxes)}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("LEGO Detector + Classifier + Color", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Session complete. CSV saved to:", EXPORT_CSV)

if __name__ == "__main__":
    run()
