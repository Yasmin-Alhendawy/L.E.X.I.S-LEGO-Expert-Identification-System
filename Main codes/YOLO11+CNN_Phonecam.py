
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import time
from ultralytics import YOLO

# === Configuration ===
IMAGE_SIZE = (96, 96)
FRAME_RESOLUTION = (640, 640)
MOBILENET_MODEL_PATH = r"C:\Users\Jasmi\Desktop\Machine project\Results\Weights\CNN_LEGO_weights_23Classes.h5"
CONFIDENCE_THRESHOLD = 0.3
YOLO_CONFIDENCE = 0.5
BBOX_PADDING = 15
MIN_AREA = 6000
MAX_AREA = 100000
SKIP_YOLO_CLASSES = ['(3008) 1×8 Brick', '(3460) 1×8 Plate', '(3832) 2×10 Plate', '(4477) 1×10 Plate']

# === Load Models ===
yolo_model = YOLO(r"C:\Users\Jasmi\Desktop\Machine project\Main codes\YOLO11_Augmentation_Pictures\YOLO11_optimizd_pic_mode.pt")
mobilenet_model = load_model(MOBILENET_MODEL_PATH)

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

def find_closest_lego_color(bgr_tuple):
    lego_color_hex = {
        'White': 0xffffff, 'Brick Yellow': 0xD9BB7B, 'Nougat': 0xD67240, 'Bright Red': 0xff0000,
        'Bright Blue': 0x0000ff, 'Bright Yellow': 0xffff00, 'Black': 0x000000, 'Dark Green': 0x009900,
        'Bright Green': 0x00cc00, 'Dark Orange': 0xA83D15, 'Medium Blue': 0x478CC6, 'Bright Orange': 0xff6600,
        'Bright Bluish Green': 0x059D9E, 'Bright Yellowish-Green': 0x95B90B, 'Bright Reddish Violet': 0x990066,
        'Sand Blue': 0x5E748C, 'Sand Yellow': 0x8D7452, 'Earth Blue': 0x002541, 'Earth Green': 0x003300,
        'Sand Green': 0x5F8265,
    }
    def hex_to_bgr(hex_val):
        return (hex_val & 0xFF, (hex_val >> 8) & 0xFF, (hex_val >> 16) & 0xFF)
    def color_distance(c1, c2):
        return sum((a - b) ** 2 for a, b in zip(c1, c2))
    return min(lego_color_hex, key=lambda name: color_distance(hex_to_bgr(lego_color_hex[name]), bgr_tuple))

def preprocess_image(img):
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_patch(img):
    try:
        if img is None or img.size == 0 or np.mean(img) < 10:
            return "Unknown", 0.0
        preprocessed = preprocess_image(img)
        preds = mobilenet_model.predict(preprocessed, verbose=0)
        if preds.shape[-1] != len(CLASS_LABELS):
            return "Unknown", 0.0
        class_id = np.argmax(preds[0])
        confidence = preds[0][class_id]
        if class_id >= len(CLASS_LABELS):
            return "Unknown", confidence
        return ("Unknown", confidence) if confidence < CONFIDENCE_THRESHOLD else (CLASS_LABELS[class_id], confidence)
    except Exception:
        return "Unknown", 0.0

# === Main Loop ===
cap = cv2.VideoCapture("http://192.168.0.105:8080/video")
if not cap.isOpened():
    exit()

while True:
    fps_start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, FRAME_RESOLUTION)
    results = yolo_model.predict(source=frame, conf=YOLO_CONFIDENCE, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []

    filtered_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)
        if MIN_AREA <= area <= MAX_AREA:
            filtered_boxes.append((x1, y1, x2, y2))

    if len(filtered_boxes) == 0:
        label, conf = classify_patch(frame)
        if label == "Unknown":
            continue
        b, g, r = frame[frame.shape[0] // 2, frame.shape[1] // 2]
        lego_color = find_closest_lego_color((b, g, r))
        text = f"{label} ({conf*100:.1f}%) [{lego_color}]"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        for x1, y1, x2, y2 in filtered_boxes:
            x1 = max(0, x1 - BBOX_PADDING)
            y1 = max(0, y1 - BBOX_PADDING)
            x2 = min(frame.shape[1], x2 + BBOX_PADDING)
            y2 = min(frame.shape[0], y2 + BBOX_PADDING)
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            label, conf = classify_patch(patch)
            if label in SKIP_YOLO_CLASSES or label == "Unknown":
                continue
            b, g, r = patch[patch.shape[0] // 2, patch.shape[1] // 2]
            lego_color = find_closest_lego_color((b, g, r))
            text = f"{label} ({conf*100:.1f}%) [{lego_color}]"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    fps_end_time = time.time()
    fps = 1.0 / (fps_end_time - fps_start_time + 1e-5)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    cv2.imshow("LEGO Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
