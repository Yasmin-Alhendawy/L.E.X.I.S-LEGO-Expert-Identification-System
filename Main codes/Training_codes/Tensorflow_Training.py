# -*- coding: utf-8 -*-
"""
Created on Sat May 17 12:15:01 2025

@author: Jasmi
"""

import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -------------------------
# STEP 1: Split Dataset
# -------------------------
SOURCE_DIR = r"C:\Users\Jasmi\Desktop\new testing code\data"
DEST_DIR = r"C:\Users\Jasmi\Desktop\new testing code\fixed"
SPLIT_RATIOS = (0.7, 0.15, 0.15)  # Train, val, test
SEED = 42

random.seed(SEED)
sets = ['train', 'val', 'test']

for split in sets:
    os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)

classes = next(os.walk(SOURCE_DIR))[1]
for class_name in classes:
    src_class_dir = os.path.join(SOURCE_DIR, class_name)
    images = [img for img in os.listdir(src_class_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(SPLIT_RATIOS[0] * n_total)
    n_val = int(SPLIT_RATIOS[1] * n_total)
    n_test = n_total - n_train - n_val

    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train + n_val],
        'test': images[n_train + n_val:]
    }

    for split in sets:
        split_dir = os.path.join(DEST_DIR, split, class_name)
        os.makedirs(split_dir, exist_ok=True)
        for img in splits[split]:
            src = os.path.join(src_class_dir, img)
            dst = os.path.join(split_dir, img)
            shutil.copyfile(src, dst)

    print(f"Class '{class_name}': {n_train} train, {n_val} val, {n_test} test images.")

# -------------------------
# STEP 2: Train CNN Model
# -------------------------
print("\n=== Training Model ===")
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
tf.keras.mixed_precision.set_global_policy('mixed_float16')

BASE_DIR = DEST_DIR
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
TEST_DIR = os.path.join(BASE_DIR, 'test')

subfolders = next(os.walk(TRAIN_DIR))[1]
num_classes = len(subfolders)
target_size = (64, 64)

train_gen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.5, brightness_range=[0.5, 1.0])
valid_gen = ImageDataGenerator(rescale=1. / 255)
test_gen = ImageDataGenerator(rescale=1. / 255)

train_batches = train_gen.flow_from_directory(
    TRAIN_DIR, target_size=target_size, class_mode='categorical',
    batch_size=32, shuffle=True, color_mode="grayscale", classes=subfolders
)

valid_batches = valid_gen.flow_from_directory(
    VAL_DIR, target_size=target_size, class_mode='categorical',
    batch_size=32, shuffle=False, color_mode="grayscale", classes=subfolders
)

test_batches = test_gen.flow_from_directory(
    TEST_DIR, target_size=target_size, class_mode='categorical',
    batch_size=32, shuffle=False, color_mode="grayscale", classes=subfolders
)

model = Sequential([
    Conv2D(64, 3, input_shape=(64, 64, 1), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, 3, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

history = model.fit(
    train_batches, validation_data=valid_batches,
    epochs=30, callbacks=[early_stopping, reduce_lr],
    steps_per_epoch=len(train_batches), validation_steps=len(valid_batches)
)

test_loss, test_acc = model.evaluate(test_batches)
print('\nTest accuracy:', test_acc)

# -------------------------
# STEP 2.1: Evaluation Metrics
# -------------------------
y_true = test_batches.classes
y_pred_proba = model.predict(test_batches)
y_pred = np.argmax(y_pred_proba, axis=1)
class_labels = list(test_batches.class_indices.keys())

# Classification Report
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("\nClassification Report:\n", report_df)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ROC AUC Score
try:
    auc = roc_auc_score(tf.keras.utils.to_categorical(y_true, num_classes),
                        y_pred_proba, multi_class='ovr', average='macro')
    print(f"\nMacro-Averaged ROC AUC Score: {auc:.4f}")
except Exception as e:
    print("\nROC AUC not computed. Reason:", str(e))

model.save("lego_model_local.h5")

# -------------------------
# STEP 3: Plot Training Graphs
# -------------------------
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.grid(True)
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.grid(True)
plt.legend()
plt.title('Accuracy')

plt.tight_layout()
plt.show()

# -------------------------
# STEP 4: Single Image Prediction
# -------------------------
def preprocess_image(image_path):
    img = load_img(image_path, target_size=target_size, color_mode="grayscale")
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Try predicting from a test image (optional)
image_path = os.path.join(BASE_DIR, 'test_sample.jpeg')  # Replace with a real image path

if os.path.exists(image_path):
    image_array = preprocess_image(image_path)
    predictions = model.predict(image_array)
    top_indices = np.argsort(predictions[0])[-3:][::-1]

    print("\nTop 3 Predictions:")
    for idx in top_indices:
        print(f"Class: {subfolders[idx]}, Probability: {predictions[0][idx]:.4f}")
else:
    print("\nNote: No test image found at:", image_path)
