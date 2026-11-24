"""Grad-CAM for the Protocol A VGG16-20 benchmark (target layer block5_conv3, gradient of the pre-sigmoid TB logit). Edit img_path to explain another radiograph.

Extracted from the executed notebook `notebooks/09_GradCAM_VGG16_explanations.ipynb` (cell 2,3,4,5,6,7,8,9,10,11,12,13,14,15); the model definition and the
training/evaluation code are unchanged. Only hard-coded paths were replaced by the environment variables
below, and figures are written to disk instead of being displayed.

    TB_DATA_DIR    folder containing Normal/ and Tuberculosis/ of the TB Chest Radiography Database
                   (default: data/TB_Chest_Radiography_Database)
    TB_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/gradcam)

Run from the `tuberculosis/` folder:  python src/explainability/gradcam_vgg16.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("TB_DATA_DIR", "data/TB_Chest_Radiography_Database"))
OUT_DIR = os.path.abspath(os.environ.get("TB_OUTPUT_DIR", "outputs/gradcam"))
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
    os.environ.get("TB_VGG16_MODEL", os.path.join(os.path.dirname(OUT_DIR), "vgg16_20", "tb_vgg16_20epochs.h5"))
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

img_path = os.path.join(DATA_DIR, "Tuberculosis", "TB.54.jpg")

img_array = preprocess_image(img_path)

print("Image prepared successfully.")
print("Image shape:", img_array.shape)
print("Minimum pixel value:", img_array.min())
print("Maximum pixel value:", img_array.max())

prediction = model.predict(img_array, verbose=0)

probability = float(prediction[0][0])

print("Prediction probability:", probability)

if probability >= 0.5:
    predicted_class = "Tuberculosis"
else:
    predicted_class = "Normal"

print("Predicted class:", predicted_class)

def gradcam(img_array):

    # Create a model that returns:
    # 1. VGG16 target convolutional feature maps
    # 2. VGG16 feature output
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            target_layer.output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:

        # Forward pass through VGG16
        conv_outputs, base_output = grad_model(
            img_array,
            training=False
        )

        # Pass VGG16 features through the classification head
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

grad_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=[
        target_layer.output,
        base_model.output
    ]
)

with tf.GradientTape() as tape:

    conv_outputs, base_output = grad_model(
        img_array,
        training=False
    )

    x = base_output

    for layer in model.layers[1:]:
        x = layer(x, training=False)

    predictions = x[:, 0]

    target = predictions

grads = tape.gradient(
    target,
    conv_outputs
)

print("Target layer output shape:", conv_outputs.shape)
print("Gradient shape:", grads.shape)

print("Prediction:", float(predictions[0]))

print("Gradient minimum:", float(tf.reduce_min(grads)))
print("Gradient maximum:", float(tf.reduce_max(grads)))

print("Gradient mean:", float(tf.reduce_mean(grads)))
print("Gradient absolute mean:", float(tf.reduce_mean(tf.abs(grads))))

grad_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=[
        target_layer.output,
        base_model.output
    ]
)

final_layer = model.layers[-1]

with tf.GradientTape() as tape:

    # Forward pass through VGG16
    conv_outputs, base_output = grad_model(
        img_array,
        training=False
    )

    # Classification head, excluding final sigmoid
    x = base_output

    for layer in model.layers[1:-1]:
        x = layer(x, training=False)

    # Calculate the pre-sigmoid logit directly
    weights, bias = final_layer.get_weights()

    logit = tf.matmul(
        x,
        tf.constant(weights, dtype=tf.float32)
    ) + tf.constant(bias, dtype=tf.float32)

    target = logit[:, 0]

# Calculate gradients of the logit
grads = tape.gradient(
    target,
    conv_outputs
)

print("Target layer output shape:", conv_outputs.shape)
print("Pre-sigmoid logit:", float(logit[0][0]))
print("Sigmoid probability:", float(tf.sigmoid(logit)[0][0]))

print("Gradient shape:", grads.shape)
print("Gradient minimum:", float(tf.reduce_min(grads)))
print("Gradient maximum:", float(tf.reduce_max(grads)))
print("Gradient mean:", float(tf.reduce_mean(grads)))
print("Gradient absolute mean:", float(tf.reduce_mean(tf.abs(grads))))

def gradcam_logit(img_array):

    # Create model for Grad-CAM
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            target_layer.output,
            base_model.output
        ]
    )

    final_layer = model.layers[-1]

    with tf.GradientTape() as tape:

        # Forward pass through VGG16
        conv_outputs, base_output = grad_model(
            img_array,
            training=False
        )

        # Forward pass through classifier
        x = base_output

        for layer in model.layers[1:-1]:
            x = layer(x, training=False)

        # Get final Dense layer weights
        weights, bias = final_layer.get_weights()

        # Calculate pre-sigmoid logit
        logit = tf.matmul(
            x,
            tf.constant(weights, dtype=tf.float32)
        ) + tf.constant(bias, dtype=tf.float32)

        target = logit[:, 0]

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

    # Weighted combination of feature maps
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # Keep only positive contributions
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize for visualization
    max_val = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_val > 0,
        heatmap / max_val,
        tf.zeros_like(heatmap)
    )

    return heatmap.numpy()

heatmap = gradcam_logit(img_array)

print("Grad-CAM generated successfully.")
print("Heatmap shape:", heatmap.shape)
print("Heatmap minimum:", heatmap.min())
print("Heatmap maximum:", heatmap.max())
print("Heatmap mean:", heatmap.mean())

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

    # Create three-panel figure
    plt.figure(figsize=(15, 5))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original X-ray", fontsize=14)
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


show_gradcam(img_path, heatmap)
