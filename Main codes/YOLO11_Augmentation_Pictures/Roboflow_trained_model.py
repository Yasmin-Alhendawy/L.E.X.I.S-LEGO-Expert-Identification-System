from inference import get_model
import supervision as sv
import cv2

# Load a pre-trained YOLOv8n model
model = get_model(model_id="ldc-lego-yolo-5wyps/1", api_key="w5RGdO0YzK6pvmC3EoRx")


# Create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Start webcam capture (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference
    results = model.infer(frame)[0]

    # Load the results into the supervision Detections API
    detections = sv.Detections.from_inference(results)

    # Annotate the frame
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    # Display the frame
    cv2.imshow("Webcam YOLOv8 Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
