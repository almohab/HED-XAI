"""Grad-CAM for the VGG16-20 benchmark (target layer block5_conv3). Edit image_path to explain another slice.

Extracted from the executed notebook `notebooks/12_GradCAM_VGG16_explanations.ipynb` (cell 0,1,2,3,4,5,6,7); the model definition and the
training/evaluation code are unchanged. Only hard-coded Windows paths were replaced by the
environment variables below, and figures are written to disk instead of being displayed.

    BT_DATA_DIR    folder with train/, val/ and test/ sub-folders, each containing glioma/,
                   meningioma/ and pituitary/ (default: data/converted_images_split)
    BT_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/gradcam)

Run from the `brain_tumor/` folder:  python src/explainability/gradcam_vgg16.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("BT_DATA_DIR", "data/converted_images_split"))
OUT_DIR = os.path.abspath(os.environ.get("BT_OUTPUT_DIR", "outputs/gradcam"))
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)  # relative file names used below (models, figures) are written here

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_fig_no = [0]
def _save_instead_of_show(*args, **kwargs):
    _fig_no[0] += 1
    plt.savefig(f"figure_{_fig_no[0]:02d}.png", dpi=200, bbox_inches="tight")
    plt.close("all")
plt.show = _save_instead_of_show

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

model_path = os.environ.get("BT_VGG16_MODEL", os.path.join(os.path.dirname(OUT_DIR), "vgg16_20", "VGG16_TB_Classification.h5"))

model = tf.keras.models.load_model(model_path)

print("Model loaded successfully.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

print("\nModel layers:")
for i, layer in enumerate(model.layers):
    print(i, "->", layer.name, "->", layer.__class__.__name__)

target_layer = model.get_layer("block5_conv3")

print("Grad-CAM target layer:", target_layer.name)
print("Layer type:", target_layer.__class__.__name__)
print("Number of filters:", target_layer.filters)

# Check the actual output tensor shape
print("Output tensor shape:", target_layer.output.shape)

image_path = os.path.join(DATA_DIR, "test", "glioma", "891.jpg")

def preprocess_image(path):

    img = tf.keras.preprocessing.image.load_img(
        path,
        target_size=(224, 224)
    )

    img = tf.keras.preprocessing.image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = img / 255.0

    return img


img_array = preprocess_image(image_path)

print("Image prepared successfully.")
print("Image shape:", img_array.shape)
print("Minimum pixel value:", img_array.min())
print("Maximum pixel value:", img_array.max())

prediction = model.predict(img_array, verbose=0)

print("Raw prediction:", prediction)
print("Prediction shape:", prediction.shape)

predicted_index = np.argmax(prediction[0])
confidence = float(prediction[0][predicted_index])

print("Predicted class index:", predicted_index)
print("Confidence:", confidence)

import os

test_dir = r"C:\Users\usER\Downloads\1512427\converted_images_split\test"

class_names = sorted([
    folder for folder in os.listdir(test_dir)
    if os.path.isdir(os.path.join(test_dir, folder))
])

print("Class names in alphabetical order:")
for i, class_name in enumerate(class_names):
    print(i, "->", class_name)

print("\nModel predicted index:", predicted_index)
print("Corresponding class:", class_names[predicted_index])

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        target_layer.output,
        model.output
    ]
)

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(
        img_array,
        training=False
    )

    # Target the predicted class: pituitary (index 2)
    target = predictions[:, predicted_index]

grads = tape.gradient(
    target,
    conv_outputs
)

print("Target layer output shape:", conv_outputs.shape)
print("Prediction shape:", predictions.shape)
print("Target class index:", predicted_index)
print("Target class:", class_names[predicted_index])
print("Target probability:", float(target[0]))

print("\nGradient shape:", grads.shape)
print("Gradient minimum:", float(tf.reduce_min(grads)))
print("Gradient maximum:", float(tf.reduce_max(grads)))
print("Gradient mean:", float(tf.reduce_mean(grads)))
print("Gradient absolute mean:", float(tf.reduce_mean(tf.abs(grads))))

# Global average pooling of gradients
pooled_grads = tf.reduce_mean(
    grads,
    axis=(0, 1, 2)
)

# Remove batch dimension from feature maps
conv_outputs = conv_outputs[0]

# Weighted combination of feature maps
heatmap = tf.reduce_sum(
    conv_outputs * pooled_grads,
    axis=-1
)

# Keep positive contributions only
heatmap = tf.maximum(
    heatmap,
    0
)

# Normalize heatmap to [0, 1]
max_val = tf.reduce_max(heatmap)

heatmap = tf.where(
    max_val > 0,
    heatmap / max_val,
    tf.zeros_like(heatmap)
)

heatmap = heatmap.numpy()

print("Grad-CAM generated successfully.")
print("Heatmap shape:", heatmap.shape)
print("Heatmap minimum:", heatmap.min())
print("Heatmap maximum:", heatmap.max())
print("Heatmap mean:", heatmap.mean())

def show_gradcam(image_path, heatmap, predicted_class, confidence):

    # Load original MRI
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (224, 224))

    # Resize Grad-CAM heatmap
    heatmap_resized = cv2.resize(
        heatmap,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    # Convert heatmap to 8-bit
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply JET colormap
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # Create overlay
    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    # Create figure
    plt.figure(figsize=(15, 5))

    # Original MRI
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original MRI", fontsize=14)
    plt.axis("off")

    # Grad-CAM Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_resized, cmap="jet")
    plt.title("Grad-CAM Heatmap", fontsize=14)
    plt.axis("off")

    # Grad-CAM Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(
        f"Grad-CAM Overlay\n"
        f"Predicted: {predicted_class} ({confidence:.2%})",
        fontsize=14
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()


show_gradcam(
    image_path,
    heatmap,
    class_names[predicted_index],
    confidence
)
