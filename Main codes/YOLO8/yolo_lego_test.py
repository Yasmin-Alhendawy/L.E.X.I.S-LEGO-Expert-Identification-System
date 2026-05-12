from ultralytics import YOLO
import torch

# Set device (RTX 4060 will use CUDA)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the best model
model = YOLO("runs/detect/train18/weights/best.pt")  # Update if your path is different
model.to(device)

def test_image(image_path: str):
    results = model(image_path)
    results[0].show()  # Opens window with prediction

if __name__ == "__main__":
    test_image("D:/Desktop/YOLO_Lego/Test_images/2x2 top2.jpeg")
