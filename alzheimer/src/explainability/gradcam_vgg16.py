"""Grad-CAM explanations for the VGG16 benchmark (target layer block5_conv3).

Extracted from the executed notebook `03_GradCAM_VGG16_explanations.ipynb` (cell 16-25); the model definition and the
training/evaluation code are unchanged. Only hard-coded Windows paths were replaced by the
environment variables below, and figures are written to disk instead of being displayed.

    AD_DATA_DIR    folder containing "Non Demented/" and "Very mild Dementia/" (default: data/balanced_data)
    AD_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/gradcam)

Run from the `alzheimer/` folder:  python src/explainability/gradcam_vgg16.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("AD_DATA_DIR", "data/balanced_data"))
OUT_DIR = os.path.abspath(os.environ.get("AD_OUTPUT_DIR", "outputs/gradcam"))
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
import sys, glob
MODEL_FILE = os.environ.get("AD_VGG16_MODEL", os.path.join(os.path.dirname(OUT_DIR), "vgg16_50", "vgg16_model.h5"))

print("TensorFlow version:", tf.__version__)
print("Libraries loaded successfully.")

model = tf.keras.models.load_model(MODEL_FILE)

print("Model loaded successfully.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

base_model = model.layers[0]

print("Base model:", base_model.name)
print("Base model output shape:", base_model.output_shape)

def preprocess_image(path):
    img = tf.keras.preprocessing.image.load_img(
        path,
        target_size=(224, 224)
    )

    img = tf.keras.preprocessing.image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = img / 255.0

    return img

def gradcam(img_array):

    target_layer = base_model.get_layer("block5_conv3")

    # Model that returns the target convolutional activations
    # and the VGG16 feature output
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[target_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, base_output = grad_model(
            img_array,
            training=False
        )

        # Pass VGG16 features through the classifier
        x = base_output

        for layer in model.layers[1:]:
            x = layer(x, training=False)

        predictions = x[:, 0]

        # Explain the predicted class
        if float(predictions[0]) >= 0.5:
            target = predictions
        else:
            target = 1.0 - predictions

    # Gradient of target class with respect to feature maps
    grads = tape.gradient(target, conv_outputs)

    # Global average pooling
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # Weighted combination of feature maps
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(heatmap, 0)

    # Normalize ONLY for visualization
    max_value = tf.reduce_max(heatmap)

    heatmap = heatmap / (
        max_value + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()

def show_gradcam(img_path, heatmap):

    # Load original image
    original = cv2.imread(img_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (224, 224))

    # Resize Grad-CAM
    heatmap_resized = cv2.resize(
        heatmap,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    # Convert heatmap to 0–255
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply JET colormap
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # Convert BGR to RGB
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

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original Image", fontsize=14)
    plt.axis("off")

    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_resized, cmap="jet")
    plt.title("Grad-CAM Heatmap", fontsize=14)
    plt.axis("off")

    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Grad-CAM Overlay", fontsize=14)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------ batch driver (added)
# Usage: python src/explainability/gradcam_vgg16.py "path/to/slice1.jpg" "path/to/slice2.jpg" ...
import sys, glob
images = sys.argv[1:] or sorted(glob.glob(os.path.join(DATA_DIR, "*", "*.jpg")))[:6]
for img_path in images:
    img_array = preprocess_image(img_path)
    probability = float(model.predict(img_array, verbose=0)[0][0])
    print(os.path.basename(img_path), "P(Very mild Dementia) =", round(probability, 4))
    heatmap = gradcam(img_array)
    show_gradcam(img_path, heatmap)
