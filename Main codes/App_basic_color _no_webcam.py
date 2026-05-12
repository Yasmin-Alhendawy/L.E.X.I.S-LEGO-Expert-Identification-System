# -*- coding: utf-8 -*-
"""
Created on Sat May 17 22:27:09 2025

@author: Jasmi
"""

import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
from tensorflow.keras.models import load_model

# === Configuration ===
MODEL_PATH = r"C:\Users\Jasmi\Desktop\new testing code\mobilenetv2_classifier_20250517_213548.h5"  # Change to your saved model file
IMAGE_SIZE = (96, 96)
CONFIDENCE_THRESHOLD = 0.5

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

# === Load Trained Model ===
model = load_model(MODEL_PATH)

def preprocess_image_pil(pil_img):
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(IMAGE_SIZE)
    img = np.array(pil_img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image_from_pil(pil_img):
    preprocessed = preprocess_image_pil(pil_img)
    preds = model.predict(preprocessed)
    class_id = np.argmax(preds[0])
    confidence = preds[0][class_id]
    return ("Unknown", confidence) if confidence < CONFIDENCE_THRESHOLD else (CLASS_LABELS[class_id], confidence)

def bgr_to_hex(b, g, r):
    return f"#{r:02X}{g:02X}{b:02X}"

def detect_color(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width, _ = frame.shape
    cx, cy = width // 2, height // 2
    hue, sat, val = hsv_frame[cy, cx]

    color = "Undefined"
    if sat < 50 and val > 200:
        color = "WHITE"
    elif val < 50:
        color = "BLACK"
    elif sat < 50:
        color = "GRAY"
    elif hue < 10 or hue > 160:
        color = "RED"
    elif hue < 21:
        color = "ORANGE"
    elif hue < 31:
        color = "YELLOW"
    elif hue < 86:
        color = "GREEN"
    elif hue < 96:
        color = "CYAN"
    elif hue < 126:
        color = "BLUE"
    elif hue < 146:
        color = "VIOLET"
    elif hue < 160:
        color = "PINK"

    b, g, r = frame[cy, cx]
    return color, (b, g, r), (cx, cy)

class PredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LEGO Classifier + Color Picker")
        Label(root, text="Choose an option below", font=("Arial", 16)).pack(pady=10)

        self.image_label = Label(root)
        self.image_label.pack()
        self.result_label = Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)
        self.color_label = Label(root, text="", font=("Arial", 14))
        self.color_label.pack(pady=10)

        Button(root, text="Open Webcam", command=self.open_webcam, width=20).pack(pady=5)
        Button(root, text="Test Image File", command=self.load_image, width=20).pack(pady=5)

    def open_webcam(self):
        def run_camera():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.result_label.config(text="Webcam not detected!")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                label, conf = predict_image_from_pil(pil_img)
                color_name, (b, g, r), (cx, cy) = detect_color(frame)
                hex_color = bgr_to_hex(b, g, r)

                text = f"{label} ({conf*100:.1f}%)"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if label != "Unknown" else (0, 0, 255), 2)
                cv2.putText(frame, f"Color: {color_name} | {hex_color}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (b, g, r), 2)
                cv2.circle(frame, (cx, cy), 5, (25, 25, 25), 2)

                cv2.imshow("Webcam - LEGO Classifier + Color Picker", frame)
                if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                    break

            cap.release()
            cv2.destroyAllWindows()

        threading.Thread(target=run_camera).start()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"), ("All Files", "*.*")])
        if not path:
            return
        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception as e:
            self.result_label.config(text=f"Failed to load image: {e}")
            return

        label, conf = predict_image_from_pil(pil_img)
        result_text = f"Prediction: {label} ({conf*100:.1f}%)"
        self.result_label.config(text=result_text)

        frame = np.array(pil_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        color_name, (b, g, r), _ = detect_color(frame)
        hex_color = bgr_to_hex(b, g, r)
        self.color_label.config(text=f"Color: {color_name} | {hex_color}", fg=hex_color)

        display = pil_img.resize((192, 192))
        photo = ImageTk.PhotoImage(display)
        self.image_label.config(image=photo)
        self.image_label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorApp(root)
    root.mainloop()
