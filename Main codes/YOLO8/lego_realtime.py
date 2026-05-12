from ultralytics import YOLO
import torch

# Set device (RTX 4060 will use CUDA)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the best model
model = YOLO("best.pt")
model.to(device)

def run_realtime():
    # 0 = default webcam, change if you have multiple cameras
    model.predict(source=0, show=True, device=device)

if __name__ == "__main__":
    run_realtime()
