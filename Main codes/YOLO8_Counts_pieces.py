from ultralytics import YOLO
import torch
import cv2

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ✅ Make sure the path to your model is correct here:
model = YOLO(r"C:\Users\Jasmi\Desktop\Machine project\Useful codes\Yolocodes\best.pt")
model.to(device)

def run_realtime_with_count():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not detected!")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, device=device, conf=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
        count = len(boxes)

        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f"Count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("LEGO Detection + Count", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_with_count()
