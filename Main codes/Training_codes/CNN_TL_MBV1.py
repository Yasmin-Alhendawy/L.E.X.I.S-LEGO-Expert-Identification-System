# -*- coding: utf-8 -*-
"""
Created on Sat May 17 14:10:40 2025

@author: Jasmi
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# === Configuration ===
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = r"C:\Users\Jasmi\Desktop\new testing code\data"

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
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# === Load MobileNet Base ===
base_model = MobileNet(
    weights='imagenet',
    include_top=False,
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)
base_model.trainable = False  # Freeze the base model

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

# === Save Model ===
model.save("mobilenet_classifier.h5")
print("Model saved as mobilenet_classifier.h5")

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


import cv2
import numpy as np

# === Load the trained model ===
model = tf.keras.models.load_model("mobilenet_classifier.h5")
class_labels = list(train_generator.class_indices.keys())

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

# === Real-time Webcam Prediction ===
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

# === Test with Image File ===
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

# === Run testing ===
# Uncomment one of the following to test

#test_with_image_file("test.jpg")  # Replace with your image file path
test_with_webcam()                # Use webcam for real-time prediction

