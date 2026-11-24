"""Grad-CAM for the VGG16-20 benchmark (block5_conv3, predicted-class sigmoid score). Edit img_path to explain another radiograph.

Extracted from the executed notebook `notebooks/01_benchmarks_resplit_VGG16_DenseNet121_MobileNetV2_CNN_InceptionV3_and_GradCAM.ipynb` (cell 6,7,8,9,10,11,12,13,14,15,16); the model definition and the
training/evaluation code are unchanged. Only hard-coded Windows paths were replaced by the environment
variables below, and figures are written to disk instead of being displayed.

    PN_RAW_DIR     original Kaggle/Mendeley release with train/, val/, test/ (default: data/raw/chest_xray)
    PN_DATA_DIR    re-split data with train/, val/, test/, each containing NORMAL/ and PNEUMONIA/
                   (default: data/chest_xray_new)
    PN_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/gradcam_vgg16)

Run from the `pneumonia/` folder:  python src/explainability/gradcam_vgg16.py
"""
import os
RAW_DIR = os.path.abspath(os.environ.get("PN_RAW_DIR", "data/raw/chest_xray"))
DATA_DIR = os.path.abspath(os.environ.get("PN_DATA_DIR", "data/chest_xray_new"))
OUT_DIR = os.path.abspath(os.environ.get("PN_OUTPUT_DIR", "outputs/gradcam_vgg16"))
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

print("TensorFlow version:", tf.__version__)
print("Libraries loaded successfully.")

model = tf.keras.models.load_model(
    os.environ.get("PN_MODEL", os.path.join(os.path.dirname(OUT_DIR), "vgg16_20", "pneumonia_vgg16 20Epochs_model.h5"))
)

print("Model loaded successfully.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

base_model = model.layers[0]

print("Base model:", base_model.name)
print("Base model output shape:", base_model.output_shape)

target_layer = base_model.get_layer("block5_conv3")

print("Grad-CAM layer:", target_layer.name)
print("Number of filters:", target_layer.filters)

def preprocess_image(path):
    img = tf.keras.preprocessing.image.load_img(
        path,
        target_size=(224, 224)
    )

    img = tf.keras.preprocessing.image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = img / 255.0

    return img

img_path = os.path.join(DATA_DIR, "test", "PNEUMONIA", "person15_virus_46.jpeg")

img_array = preprocess_image(img_path)

print("Image prepared successfully.")
print("Image shape:", img_array.shape)
print("Minimum pixel value:", img_array.min())
print("Maximum pixel value:", img_array.max())

prediction = model.predict(img_array, verbose=0)

probability = float(prediction[0][0])

print("Prediction probability:", probability)

if probability >= 0.5:
    predicted_class = "Pneumonia"
else:
    predicted_class = "Normal"

print("Predicted class:", predicted_class)

def gradcam(img_array):

    # Select the target convolutional layer
    target_layer = base_model.get_layer("block5_conv3")

    # Create a model that returns:
    # 1. target convolutional feature maps
    # 2. VGG16 feature output
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[target_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:

        # Forward pass through VGG16
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

    # Calculate gradients
    grads = tape.gradient(
        target,
        conv_outputs
    )

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # Weight feature maps by their gradients
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize for visualization
    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()

heatmap = gradcam(img_array)

print("Heatmap generated successfully.")
print("Heatmap shape:", heatmap.shape)
print("Heatmap minimum:", heatmap.min())
print("Heatmap maximum:", heatmap.max())

def show_gradcam(img_path, heatmap):

    # Load original X-ray
    original = cv2.imread(img_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (224, 224))

    # Resize Grad-CAM heatmap
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

    # 1. Original X-ray
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original X-ray", fontsize=14)
    plt.axis("off")

    # 2. Grad-CAM heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_resized, cmap="jet")
    plt.title("Grad-CAM Heatmap", fontsize=14)
    plt.axis("off")

    # 3. Grad-CAM overlay
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Grad-CAM Overlay", fontsize=14)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

show_gradcam(img_path, heatmap)
