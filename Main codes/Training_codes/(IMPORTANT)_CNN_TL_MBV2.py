# -*- coding: utf-8 -*-
"""
Updated script with base_model unfreezing for fine-tuning.
Includes ReduceLROnPlateau and EarlyStopping callbacks.
Enhanced data augmentation.
Now uses MobileNetV2 as the feature extractor.
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
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# === Configuration ===
IMAGE_SIZE = (96, 96)  # Upsampled for MobileNetV2 compatibility
BATCH_SIZE = 32
EPOCHS = 30
DATASET_PATH = r"C:\Users\Jasmi\Desktop\Machine project\Meow\data6"

# === Load Data ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
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

# === Load MobileNetV2 Base ===
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)
base_model.trainable = True  # Unfreeze for fine-tuning

# === Add Custom Head ===
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# === Compile Model ===
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Callbacks ===
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)

# === Compute Class Weights ===
from sklearn.utils.class_weight import compute_class_weight

class_indices = train_generator.class_indices
class_labels_array = np.array(list(class_indices.values()))
class_weight = compute_class_weight(class_weight='balanced', classes=class_labels_array, y=train_generator.classes)
class_weight_dict = dict(enumerate(class_weight))

# === Train Model ===
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[lr_schedule, early_stop],
    class_weight=class_weight_dict
)

# === Save Model with Timestamp ===
timestamp = time.strftime("%Y%m%d_%H%M%S")
model_filename = f"mobilenetv2_classifier_{timestamp}.h5"
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
    
if __name__ == "__main__":
    test_with_webcam()

