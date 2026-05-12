# -*- coding: utf-8 -*-
"""
Created on Sun May 25 18:43:36 2025

@author: Jasmi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 20 12:13:57 2025

@author: Jasmi
"""

import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import tensorflow as tf
import threading

# === Configuration ===
IMAGE_SIZE = (96, 96)
MOBILENET_MODEL_PATH = r"C:\Users\Jasmi\Desktop\Machine project\Results\Weights\CNN_LEGO_weights_23Classes.h5"
YOLO_MODEL_PATH = r"C:\Users\Jasmi\Desktop\Machine project\Main codes\YOLO11_Augmentation_Pictures\YOLO11_optimizd_pic_mode.pt"
CONFIDENCE_THRESHOLD = 0.2

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

# === Helper Functions ===
def preprocess_image(img):
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_patch(img):
    resized = cv2.resize(img, IMAGE_SIZE)
    preprocessed = preprocess_image(resized)
    preds = mobilenet_model.predict(preprocessed)
    class_id = np.argmax(preds[0])
    confidence = preds[0][class_id]
    if confidence < CONFIDENCE_THRESHOLD:
        return "Unknown", confidence
    return CLASS_LABELS[class_id], confidence

def browse_and_classify_only():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if filepath:
        img = cv2.imread(filepath)
        if img is None:
            messagebox.showerror("Error", "Image could not be loaded.")
            return

        label, conf = classify_patch(img)
        color = detect_color(img)

        img_resized = cv2.resize(img, IMAGE_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        panel.config(image=img_tk)
        panel.image = img_tk

        result_text.set(f"Label: {label}\nConfidence: {conf*100:.1f}%\nColor: {color}")

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

def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.predict(source=frame, device=device, conf=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            patch = frame[y1:y2, x1:x2]

            if patch.size == 0:
                continue

            label, conf = classify_patch(patch)
            color = detect_color(patch)

            text = f"{label} ({conf*100:.1f}%) [{color}]"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Webcam Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === GUI Setup ===
root = tk.Tk()
root.title("Brick Classifier GUI")

btn_image = tk.Button(root, text="Classify Image (MobileNet Only)", command=browse_and_classify_only)
btn_image.pack(pady=10)

btn_webcam = tk.Button(root, text="Start Webcam", command=lambda: threading.Thread(target=run_webcam).start())
btn_webcam.pack(pady=10)

panel = tk.Label(root)
panel.pack()

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Arial", 12), justify="left")
result_label.pack(pady=5)

root.mainloop()