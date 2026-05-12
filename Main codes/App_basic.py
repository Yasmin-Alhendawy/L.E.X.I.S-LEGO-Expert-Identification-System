import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import os
import csv
import datetime
from tensorflow.keras.models import load_model

# === Configuration ===
MODEL_PATH = r"C:\Users\Jasmi\Desktop\new testing code\mobilenetv2_classifier_20250517_213548.h5"
IMAGE_SIZE = (96, 96)
CONFIDENCE_THRESHOLD = 0.0

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
print(f"Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# === Image Preprocessing ===
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

    if confidence < CONFIDENCE_THRESHOLD:
        return "Unknown", confidence
    else:
        return CLASS_LABELS[class_id], confidence

# === GUI Application ===
class PredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LEGO Part Classifier")
        self.label = Label(root, text="Choose an option below", font=("Arial", 16))
        self.label.pack(pady=10)

        self.image_label = Label(root)
        self.image_label.pack()

        self.result_label = Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

        Button(root, text="Open Webcam", command=self.open_webcam, width=20).pack(pady=5)
        Button(root, text="Take Prediction", command=self.capture_prediction, width=20).pack(pady=5)
        Button(root, text="Test Image File", command=self.load_image, width=20).pack(pady=5)

        self.cap = None
        self.session_dir = None
        self.log_path = None
        self.csv_path = None
        self.predict_count = 0

    def open_webcam(self):
        def run_camera():
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 96)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 96)

            if not self.cap.isOpened():
                self.result_label.config(text="Webcam not detected!")
                return

            session_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.session_dir = os.path.join("sessions", f"session_{session_time}")
            os.makedirs(self.session_dir, exist_ok=True)

            self.log_path = os.path.join(self.session_dir, "results_log.txt")
            self.csv_path = os.path.join(self.session_dir, "predictions.csv")

            with open(self.log_path, "w") as log_file, open(self.csv_path, "w", newline='') as csv_file:
                log_file.write(f"--- Session Started: {session_time} ---\n")
                writer = csv.writer(csv_file)
                writer.writerow(["Frame #", "Timestamp", "Label", "Confidence (%)"])

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.frame = frame.copy()
                display_frame = frame.copy()
                cv2.imshow("Webcam Feed", display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.cap.release()
            self.cap = None
            cv2.destroyAllWindows()

            with open(self.log_path, "a") as log_file:
                log_file.write(f"--- Session Ended: {datetime.datetime.now()} ---\n")

        threading.Thread(target=run_camera).start()

    def capture_prediction(self):
        if self.cap is None or not hasattr(self, 'frame'):
            self.result_label.config(text="Webcam not running!")
            return

        frame = self.frame
        resized_frame = cv2.resize(frame, IMAGE_SIZE)
        pil_img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        label, conf = predict_image_from_pil(pil_img)
        result_text = f"Prediction: {label} ({conf * 100:.1f}%)"
        self.result_label.config(text=result_text)

        self.predict_count += 1
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        frame_filename = os.path.join(self.session_dir, f"frame_{self.predict_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        with open(self.log_path, "a") as log_file:
            log_file.write(f"{timestamp} - {label} ({conf * 100:.1f}%)\n")

        with open(self.csv_path, "a", newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([self.predict_count, timestamp, label, f"{conf * 100:.1f}"])

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            pil_image = Image.open(file_path)
        except Exception as e:
            self.result_label.config(text=f"Failed to load image: {str(e)}")
            return

        label, conf = predict_image_from_pil(pil_image)
        result_text = f"Prediction: {label} ({conf*100:.1f}%)"
        self.result_label.config(text=result_text)

        display_img = pil_image.convert("RGB").resize((192, 192))
        photo = ImageTk.PhotoImage(display_img)
        self.image_label.config(image=photo)
        self.image_label.image = photo

# === Start GUI App ===
if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorApp(root)
    root.mainloop()
