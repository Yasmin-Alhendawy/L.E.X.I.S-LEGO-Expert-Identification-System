
# -*- coding: utf-8 -*-
"""
LEGO Classifier GUI with YOLO + CNN23 + CNN7 + Real-Time Boosts + Session Summary + TensorFlow Lite
Limited YOLO detection to realistic LEGO size range.
"""

import cv2
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import filedialog, simpledialog
import pandas as pd
from collections import deque, Counter
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image, ImageTk
import os

# === Configuration ===
IMAGE_SIZE = (96, 96)
FRAME_RESOLUTION = (480, 480)
CONFIDENCE_THRESHOLD = 0.4
YOLO_CONFIDENCE = 0.4
ENSEMBLE_ALPHA = 0.5
YOLO_SKIP_FRAMES = 1
SAVE_INTERVAL = 30
SMOOTHING_WINDOW = 30  # Number of frames for smoothing

TFLITE_MODEL_PATH = r"C:\Users\Jasmi\Desktop\Machine project\Results\Weights\CNN_LEGO_weights_23Classes.tflite"
YOLO_MODEL_PATH = r"C:\Users\Jasmi\Desktop\Machine project\Main codes\YOLO11_Augmentation_Pictures\YOLO11_optimizd_pic_mode.pt"
MODEL_7CLASS_PATH = r"C:\Users\Jasmi\Desktop\Machine project\Results\Weights\CNN_Synthatic_Data_7Classes.h5"

frame_area = FRAME_RESOLUTION[0] * FRAME_RESOLUTION[1]
MIN_AREA = 7000     # Fixed name used in rest of code
MAX_AREA = 100000   # Adjust to your LEGO size range
SKIP_YOLO_CLASSES = ['(3008) 1×8 Brick', '(3460) 1×8 Plate', '(3832) 2×10 Plate', '(4477) 1×10 Plate']


# === Smoothing buffer for classification stability ===
from collections import defaultdict, deque
prediction_buffer = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))

def get_smoothed_label(obj_id, new_label):
    buffer = prediction_buffer[obj_id]
    buffer.append(new_label)
    most_common = max(set(buffer), key=buffer.count)
    count = buffer.count(most_common)
    if count >= (SMOOTHING_WINDOW // 2 + 1):
        return most_common
    return "Unknown"


# === Load Models ===
yolo_model = YOLO(YOLO_MODEL_PATH)
cnn7_model = tf.keras.models.load_model(MODEL_7CLASS_PATH)

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("[INFO] YOLO and TFLite MobileNet models loaded successfully.")

results_list = []
piece_counter = Counter()
session_start_time = time.time()

CLASS_LABELS = [
    "(11214) Technic Axle Pin 3L with Friction Ridges Lengthwise and 1L Axle",
    "(15068) 2×2 Curved", "(15573) 1×2 Jumper", "(2445) 2×12 Plate", "(3004) 1×2 Brick",
    "(3005) 1×1 Brick", "(3008) 1×8 Brick", "(3022) 2×2 Plate",
    "(30414) Brick Special 1 x 4 with 4 Studs on One Side",
    "(32062) Technic Axle 2 Notched", "(32607) Plant, Plate 1 x 1 Round with 3 Leaves",
    "(3460) 1×8 Plate", "(3623) 1×3 Plate", "(3832) 2×10 Plate",
    "(4032) 2×2 Plate, Round w Axle", "(4286) Slope 33 3 x 1",
    "(44728) 2×2 Bracket", "(4477) 1×10 Plate",
    "(49668) Plate Special 1 x 1 with Tooth", "(60478) 1×2 Plate w Handle, End",
    "(87079) 2×4 Tile", "(92280) 1×2 Plate w Clip, Top", "(98138) 1×1 Tile, Round"
]

ENSEMBLE_CLASSES = [
    "(3004) 1×2 Brick", "(3005) 1×1 Brick", "(3022) 2×2 Plate",
    "(3623) 1×3 Plate", "(4286) Slope 33 3 x 1",
    "(11214) Technic Axle Pin 3L with Friction Ridges Lengthwise and 1L Axle",
    "(15573) 1×2 Jumper"
]

def apply_clahe_to_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def find_closest_lego_color(bgr_tuple):
    lego_color_hex = {
        'White': 0xffffff, 'Bright Red': 0xff0000, 'Bright Blue': 0x0000ff,
        'Bright Yellow': 0xffff00, 'Black': 0x000000, 'Dark Green': 0x009900,
        'Bright Green': 0x00cc00, 'Dark Orange': 0xA83D15, 'Bright Orange': 0xff6600
    }
    def hex_to_bgr(hex_val):
        return (hex_val & 0xFF, (hex_val >> 8) & 0xFF, (hex_val >> 16) & 0xFF)
    def color_distance(c1, c2):
        return sum((a - b) ** 2 for a, b in zip(c1, c2))
    return min(lego_color_hex, key=lambda name: color_distance(hex_to_bgr(lego_color_hex[name]), bgr_tuple))

def run_tflite_inference(image):
    interpreter.set_tensor(input_details[0]['index'], image.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def classify_patch(img):
    if img is None or img.size == 0 or np.mean(img) < 10 or np.std(img) < 20:
        return "Unknown", 0.0
    img = cv2.resize(img, IMAGE_SIZE).astype("float32") / 255.0
    preprocessed = np.expand_dims(img, axis=0)
    preds_23 = run_tflite_inference(preprocessed)[0]
    class_id_23 = np.argmax(preds_23)
    label_23 = CLASS_LABELS[class_id_23]
    conf_23 = preds_23[class_id_23]
    if label_23 in ENSEMBLE_CLASSES:
        preds_7 = cnn7_model.predict(preprocessed, verbose=0)[0]
        preds_7_padded = np.zeros_like(preds_23)
        preds_7_padded[:7] = preds_7
        ensemble_preds = ENSEMBLE_ALPHA * preds_23 + (1 - ENSEMBLE_ALPHA) * preds_7_padded
        class_id = np.argmax(ensemble_preds)
        confidence = ensemble_preds[class_id]
        label = CLASS_LABELS[class_id] if confidence >= CONFIDENCE_THRESHOLD else "Unknown"
        return label, confidence
    else:
        return label_23 if conf_23 >= CONFIDENCE_THRESHOLD else "Unknown", conf_23

def update_counter_display():
    elapsed = int(time.time() - session_start_time)
    mins, secs = divmod(elapsed, 60)
    session_time = f"Session Time: {mins:02}:{secs:02}"
    count_text = "\n".join([f"{label}: {count}" for label, count in piece_counter.items()])
    summary = f"{session_time}\nTotal Pieces: {sum(piece_counter.values())}\n{count_text}"
    counter_text.set(summary)

def save_to_excel(auto=False):
    df = pd.DataFrame(results_list)
    df = df[df['Confidence'] > CONFIDENCE_THRESHOLD]
    count_df = pd.DataFrame(piece_counter.items(), columns=['Label', 'Count'])
    elapsed = int(time.time() - session_start_time)
    summary_df = pd.DataFrame({
        'Total Pieces': [sum(piece_counter.values())],
        'Session Duration (s)': [elapsed]
    })
    file_name = "lego_autosave.xlsx" if auto else "lego_classification_summary.xlsx"
    with pd.ExcelWriter(file_name) as writer:
        df.to_excel(writer, sheet_name="Detailed Results", index=False)
        count_df.to_excel(writer, sheet_name="Piece Count Summary", index=False)
        summary_df.to_excel(writer, sheet_name="Session Summary", index=False)


# (rest of your webcam and GUI code continues here — unchanged)
# === GUI Setup ===
root = tk.Tk()
root.title("Brick Classifier GUI")

panel = tk.Label(root)
panel.pack()

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Arial", 12), justify="left")
result_label.pack(pady=5)

counter_text = tk.StringVar()
counter_label = tk.Label(root, textvariable=counter_text, font=("Arial", 10), justify="left")
counter_label.pack(pady=5)

def run_video_capture(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Failed to open video source.")
        return

    frame_count = 0
    last_save_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, FRAME_RESOLUTION)
        frame = apply_clahe_to_frame(frame)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display_frame = frame.copy()

        results = yolo_model.predict(source=frame, conf=YOLO_CONFIDENCE, verbose=False)
        yolo_output = results[0]
        boxes = yolo_output.boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                score = float(box.conf[0])
                class_id = int(box.cls[0])
                area = (x2 - x1) * (y2 - y1)
                if score < YOLO_CONFIDENCE or not (MIN_AREA <= area <= MAX_AREA):
                    continue
                patch = frame[y1:y2, x1:x2]
                label = CLASS_LABELS[class_id]
                label, conf = classify_patch(patch)
                print(f"[YOLO] {label} | YOLO: {score:.2f} | CNN: {conf:.2f}")
                if label != "Unknown" and conf > CONFIDENCE_THRESHOLD:
                    b, g, r = patch[patch.shape[0] // 2, patch.shape[1] // 2]
                    lego_color = find_closest_lego_color((b, g, r))
                    results_list.append({"Label": label, "Confidence": conf, "Color": lego_color})
                    piece_counter[label] += 1
                    update_counter_display()
                    text = f"{label} ({conf*100:.1f}%) [{lego_color}]"
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        else:
            h, w = frame.shape[:2]
            center_crop = frame[h//4:3*h//4, w//4:3*w//4]
            label, conf = classify_patch(center_crop)
            print(f"[Fallback] {label} | CNN: {conf:.2f}")
            if label != "Unknown" and conf > CONFIDENCE_THRESHOLD:
                b, g, r = center_crop[center_crop.shape[0] // 2, center_crop.shape[1] // 2]
                lego_color = find_closest_lego_color((b, g, r))
                results_list.append({"Label": label, "Confidence": conf, "Color": lego_color})
                piece_counter[label] += 1
                update_counter_display()
                text = f"{label} ({conf*100:.1f}%) [{lego_color}]"
                cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if time.time() - last_save_time > SAVE_INTERVAL:
            save_to_excel(auto=True)
            last_save_time = time.time()

        display_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("LEGO Detector", display_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def run_webcam():
    threading.Thread(target=run_video_capture, args=(0,), daemon=True).start()

def run_phone_cam():
    ip = simpledialog.askstring("Phone Camera URL", "Enter phone stream URL:")
    if ip:
        threading.Thread(target=run_video_capture, args=(ip,), daemon=True).start()

def browse_and_classify_only():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    img = cv2.imread(file_path)
    label, confidence = classify_patch(img)
    if label != "Unknown" and confidence > CONFIDENCE_THRESHOLD:
        b, g, r = img[img.shape[0] // 2, img.shape[1] // 2]
        lego_color = find_closest_lego_color((b, g, r))
        results_list.append({"Label": label, "Confidence": confidence, "Color": lego_color})
        piece_counter[label] += 1
        update_counter_display()
    result_text.set(f"Prediction: {label}\nConfidence: {confidence*100:.2f}%")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil.resize((320, 240)))
    panel.config(image=img_tk)
    panel.image = img_tk

def batch_classify_folder():
    folder_path = filedialog.askdirectory()
    if not folder_path:
        return

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    
    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        label, confidence = classify_patch(img)
        if label != "Unknown" and confidence > CONFIDENCE_THRESHOLD:
            b, g, r = img[img.shape[0] // 2, img.shape[1] // 2]
            lego_color = find_closest_lego_color((b, g, r))
            results_list.append({"Label": label, "Confidence": confidence, "Color": lego_color, "File": img_name})
            piece_counter[label] += 1
            update_counter_display()

    result_text.set(f"Batch completed.\nTotal images processed: {len(image_files)}")
    save_to_excel(auto=False)

# === GUI Buttons ===
tk.Button(root, text="Classify Image", command=browse_and_classify_only).pack(pady=10)
tk.Button(root, text="Start Webcam", command=run_webcam).pack(pady=10)
tk.Button(root, text="Use Phone Camera", command=run_phone_cam).pack(pady=10)
tk.Button(root, text="Batch Classify Folder", command=batch_classify_folder).pack(pady=10)
tk.Button(root, text="Save Results to Excel", command=lambda: save_to_excel(auto=False)).pack(pady=10)

root.mainloop()
