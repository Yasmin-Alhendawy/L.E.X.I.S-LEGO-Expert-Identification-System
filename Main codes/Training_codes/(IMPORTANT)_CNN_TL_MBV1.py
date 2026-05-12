# -*- coding: utf-8 -*-
"""
Created on Sat May 17 14:10:40 2025

@author: Jasmi
"""

import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# === Configuration ===
IMAGE_SIZE = (96, 96)  # Upsampled for MobileNet compatibility
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = r"C:\Users\Jasmi\Desktop\Machine project\Meow\Data5"

# === Load Data ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === Load MobileNet Base ===
base_model = MobileNet(
    weights='imagenet',
    include_top=False,
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)
base_model.trainable = False

# === Add Custom Head ===
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# === Compile Model ===
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Train Model ===
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# === Save Model with Timestamp ===
timestamp = time.strftime("%Y%m%d_%H%M%S")
model_filename = f"mobilenet_classifier_{timestamp}.h5"
model.save(model_filename)
print(f"Model saved as {model_filename}")

# === Plot Accuracy and Loss ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# === Evaluate Model ===
val_generator.reset()
y_true = val_generator.classes
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
class_labels = list(train_generator.class_indices.keys())

report = classification_report(y_true, y_pred_classes, target_names=class_labels)
print("Classification Report:\n", report)

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# === Load Model for Prediction ===
model = tf.keras.models.load_model(model_filename)

def preprocess_image(img):
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(img):
    preprocessed = preprocess_image(img)
    preds = model.predict(preprocessed)
    class_id = np.argmax(preds[0])
    confidence = preds[0][class_id]
    return class_labels[class_id], confidence

# === Webcam Prediction ===
def test_with_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not detected!")
        return

    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, conf = predict_image(frame)
        display_text = f"{label} ({conf*100:.1f}%)"

        cv2.putText(frame, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Real-Time Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Test with Single Image ===
def test_with_image_file(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return
    label, conf = predict_image(image)
    print(f"Predicted: {label} ({conf*100:.2f}%)")
    cv2.putText(image, f"{label} ({conf*100:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Image Prediction", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === Example Usage ===
# test_with_image_file("test.jpg")
# test_with_webcam()
# === Enhanced Webcam Prediction with Performance Metrics ===
def test_with_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not detected!")
        return

    # Performance tracking variables
    frame_count = 0
    total_fps = 0
    total_inference_time = 0
    start_time = time.time()
    
    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Start timing for FPS calculation
        frame_start = time.time()
        
        # Make prediction
        inference_start = time.time()
        label, conf = predict_image(frame)
        inference_time = time.time() - inference_start
        
        # Calculate FPS
        frame_time = time.time() - frame_start
        fps = 1 / frame_time if frame_time > 0 else 0
        
        # Update performance metrics
        frame_count += 1
        total_fps += fps
        total_inference_time += inference_time
        
        # Prepare display text
        display_text = f"{label} ({conf*100:.1f}%)"
        fps_text = f"FPS: {fps:.1f}"
        inference_text = f"Inference: {inference_time*1000:.1f}ms"
        
        # Draw on frame
        cv2.putText(frame, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, fps_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, inference_text, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Real-Time Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate and display final performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    avg_fps = total_fps / frame_count if frame_count > 0 else 0
    avg_inference = total_inference_time / frame_count * 1000 if frame_count > 0 else 0
    
    print("\n=== Performance Summary ===")
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average inference time: {avg_inference:.2f}ms")
    print(f"Total inference time: {total_inference_time:.2f} seconds")

    cap.release()
    cv2.destroyAllWindows()

# === Batch Test with Directory ===
def test_with_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    correct = 0
    total = 0
    inference_times = []
    
    for root, dirs, files in os.walk(directory_path):
        for dir_name in dirs:
            true_label = dir_name
            class_dir = os.path.join(root, dir_name)
            
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_dir, file)
                    image = cv2.imread(image_path)
                    
                    if image is not None:
                        start_time = time.time()
                        pred_label, conf = predict_image(image)
                        inference_time = time.time() - start_time
                        inference_times.append(inference_time)
                        
                        total += 1
                        if pred_label == true_label:
                            correct += 1
                        
                        print(f"Image: {file}")
                        print(f"True: {true_label}, Predicted: {pred_label} ({conf*100:.1f}%)")
                        print(f"Inference time: {inference_time*1000:.2f}ms\n")
    
    if total > 0:
        accuracy = correct / total * 100
        avg_inference = sum(inference_times) / len(inference_times) * 1000
        print("\n=== Batch Test Results ===")
        print(f"Total images tested: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Average inference time: {avg_inference:.2f}ms")
    else:
        print("No valid images found in the directory.")
