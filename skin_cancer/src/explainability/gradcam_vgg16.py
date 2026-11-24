"""Grad-CAM for the VGG16-Flatten benchmark (150x150, block5_conv3). Edit img_path to explain another lesion.

Extracted from the executed notebook `notebooks/03_GradCAM_VGG16.ipynb` (cell 0,1,2,3,4,5,6,7,8,9,10,11,12); the model definition and the
training/evaluation code are unchanged. Only hard-coded paths were replaced by the environment variables below
(and, in the hybrid script, class folders are matched case-insensitively), and figures are written to disk.

    SC_DATA_DIR    folder with train/, val/ and test/, each containing benign/ and malignant/
                   (default: data/dataset_skin)
    SC_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/gradcam_vgg16)

Run from the `skin_cancer/` folder:  python src/explainability/gradcam_vgg16.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("SC_DATA_DIR", "data/dataset_skin"))
OUT_DIR = os.path.abspath(os.environ.get("SC_OUTPUT_DIR", "outputs/gradcam_vgg16"))
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
    os.environ.get("SC_MODEL", os.path.join(os.path.dirname(OUT_DIR), "vgg16_flatten", "skin_disease_vgg16_model.h5"))
)

print("Model loaded successfully.")
print("Model input shape:", model.input_shape)

model.summary()

base_model = model.layers[0]

print("Base model:", base_model.name)
print("Base model input:", base_model.input_shape)
print("Base model output:", base_model.output_shape)

conv_layer = base_model.get_layer("block5_conv3")

print("Grad-CAM layer:", conv_layer.name)

rescale=1.0 / 255.0

IMG_SIZE = 150

def preprocess_image(img_path):

    img = tf.keras.utils.load_img(
        img_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    img_array = tf.keras.utils.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array

img_array = preprocess_image(img_path)

print("Image shape:", img_array.shape)
print("Minimum pixel value:", img_array.min())
print("Maximum pixel value:", img_array.max())

img_path = os.path.join(DATA_DIR, "test", "benign", "700.jpg")

print("Image path:")
print(img_path)

pred = model.predict(
    img_array,
    verbose=0
)[0][0]

print("Prediction probability:", float(pred))

import tensorflow as tf
import numpy as np

def gradcam(img_array):
    # Get the VGG16 base model
    base_model = model.layers[0]
    
    # Find the target convolutional layer
    target_layer = base_model.get_layer("block5_conv3")
    
    # Create a model that returns:
    # 1. the activation from block5_conv3
    # 2. the final prediction
    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[target_layer.output, base_model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, base_output = grad_model(img_array, training=False)
        
        # Pass VGG16 output through the classifier
        x = base_output
        
        for layer in model.layers[1:]:
            x = layer(x, training=False)
        
        predictions = x
        
        # Target the model's output
        loss = predictions[:, 0]
    
    # Calculate gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Remove batch dimension
    conv_outputs = conv_outputs[0]
    
    # Weight feature maps by their gradients
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )
    
    # ReLU
    heatmap = tf.maximum(heatmap, 0)
    
    # Normalize to 0–1
    max_value = tf.reduce_max(heatmap)
    
    if max_value > 0:
        heatmap /= max_value
    
    return heatmap.numpy()

heatmap = gradcam(img_array)

print("Heatmap shape:", heatmap.shape)
print("Heatmap minimum:", heatmap.min())
print("Heatmap maximum:", heatmap.max())

import matplotlib.pyplot as plt
import cv2

# Load the original image
original_img = cv2.imread(img_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# Resize heatmap to original image size
heatmap_resized = cv2.resize(
    heatmap,
    (original_img.shape[1], original_img.shape[0])
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
    original_img,
    0.6,
    heatmap_color,
    0.4,
    0
)

# Display
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap_resized, cmap="jet")
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Grad-CAM Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()
