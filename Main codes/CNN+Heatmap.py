# -*- coding: utf-8 -*-
"""
Created on Sun May 25 23:10:00 2025

@author: Jasmi
"""

# gradcam_gui.py

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button
import matplotlib.cm as cm

# === Configuration ===
IMAGE_SIZE = (96, 96)
MODEL_PATH = r"C:\Users\Jasmi\Desktop\Machine project\[Final]\Weights\CNN_LEGO_weights_23Classes.h5"# Change if needed

# === Load Model ===
model = load_model(MODEL_PATH)

# === Get Last Convolutional Layer Name ===
last_conv_layer_name = None
for layer in reversed(model.layers):
    try:
        if 'conv' in layer.name and len(layer.output.shape) == 4:
            last_conv_layer_name = layer.name
            break
    except AttributeError:
        continue


# === Class Labels ===
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

# === Grad-CAM Utility ===
def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# === GUI Callback ===
def open_image_and_run_gradcam():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if not path:
        return

    original_img = Image.open(path).convert("RGB")
    display_img = original_img.resize((192, 192))
    img_array = np.array(original_img.resize(IMAGE_SIZE)).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    pred_idx = np.argmax(preds)
    label = CLASS_LABELS[pred_idx]
    confidence = preds[pred_idx]

    heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_idx)
    heatmap = cv2.resize(heatmap, original_img.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cm.jet(heatmap)[:, :, :3]
    superimposed = np.uint8(heatmap_color * 255 * 0.5 + np.array(original_img) * 0.5)

    display_heatmap = Image.fromarray(superimposed.astype("uint8"))
    photo = ImageTk.PhotoImage(display_img)
    heat_photo = ImageTk.PhotoImage(display_heatmap)

    panel_original.config(image=photo)
    panel_original.image = photo
    panel_gradcam.config(image=heat_photo)
    panel_gradcam.image = heat_photo
    result_text.set(f"Prediction: {label}\nConfidence: {confidence*100:.2f}%")

# === Tkinter GUI Setup ===
root = tk.Tk()
root.title("LEGO Grad-CAM Visualizer")

Button(root, text="Select Image", command=open_image_and_run_gradcam, width=30).pack(pady=10)

panel_original = Label(root)
panel_original.pack(side="left", padx=10)

panel_gradcam = Label(root)
panel_gradcam.pack(side="right", padx=10)

result_text = tk.StringVar()
Label(root, textvariable=result_text, font=("Arial", 12)).pack(pady=10)

root.mainloop()
