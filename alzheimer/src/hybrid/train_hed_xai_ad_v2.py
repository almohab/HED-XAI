"""HED-XAI-AD-V2 (design iteration): VGG16 + DenseNet121, multi-scale fusion, CBAM attention, 20 epochs.

Extracted from the executed notebook `02_HED-XAI-AD_hybrid_models_training_and_evaluation.ipynb` (cell 0); the model definition and the
training/evaluation code are unchanged. Only hard-coded Windows paths were replaced by the
environment variables below, and figures are written to disk instead of being displayed.

    AD_DATA_DIR    folder containing "Non Demented/" and "Very mild Dementia/" (default: data/balanced_data)
    AD_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/HED-XAI-AD-V2)

Run from the `alzheimer/` folder:  python src/hybrid/train_hed_xai_ad_v2.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("AD_DATA_DIR", "data/balanced_data"))
OUT_DIR = os.path.abspath(os.environ.get("AD_OUTPUT_DIR", "outputs/HED-XAI-AD-V2"))
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

# ============================================================
# HED-XAI-ALZHEIMER-V2: Dual-Backbone Hybrid Deep Learning for
# Very mild Dementia Classification from Alzheimer's MRI Images
#
# Proposed architecture (V2):
#   VGG16 (primary backbone, ImageNet)  +  DenseNet121 (secondary backbone)
#   + multi-scale feature extraction from BOTH backbones
#   + deep spatial fusion (concatenation + learned 1x1 projection)
#   + CBAM attention (channel attention AND spatial attention)
#   + regularised dense binary classifier
#
# Training improvements over V1:
#   - Backbone-specific ImageNet preprocessing built INTO the model
#     (feed raw 0-255 RGB images; correct for both VGG16 and DenseNet121)
#   - Deeper fine-tuning (VGG16 block4+block5, DenseNet121 dense block 5)
#   - Label smoothing in binary cross-entropy
#   - Linear warm-up + cosine-decay learning rate, combined with
#     ReduceLROnPlateau-style reduction (single callback, no conflict)
#   - Label-preserving (intra-class) MixUp augmentation
#   - Stronger but controlled geometric/intensity augmentation
#   - Optional horizontal-flip test-time augmentation (reported separately)
#
# Grad-CAM compatible: all backbone layers live in the top-level graph,
# so e.g. model.get_layer("cbam_refined_features") or
# model.get_layer("block5_conv3") can be used directly for Grad-CAM.
#
# This is a research prototype, not a clinically validated tool.
# Fixed training duration: 20 epochs.
#
# Expected dataset structure:
# dataset/
#   Non Demented/
#       image1.png
#       ...
#   Very mild Dementia/
#       image1.png
#       ...
#
# Install packages in your TensorFlow environment:
# pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow scipy
# ============================================================

import os
import json
import math
import time
import random
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc,
)

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Conv2D,
    Activation,
    Add,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Concatenate,
    Multiply,
    Reshape,
    Permute,
    BatchNormalization,
    LayerNormalization,
    Normalization,
)
from tensorflow.keras.applications import VGG16, DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    CSVLogger,
    EarlyStopping,
    Callback,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model


# ============================================================
# 1. CONFIGURATION -- EDIT THESE TWO PATHS
# ============================================================

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16           # Use 8 if you get an out-of-memory (OOM) error.
EPOCHS = 20               # Fixed 20 epochs (early stopping is OFF by default).
LEARNING_RATE = 1e-4      # Peak learning rate after warm-up.
MIN_LEARNING_RATE = 1e-7
WARMUP_EPOCHS = 2         # Linear LR warm-up protects pretrained weights.
THRESHOLD = 0.5

# Fine-tuning: every non-BatchNorm layer from this layer onward is trainable.
VGG16_TRAINABLE_FROM = "block4_conv1"          # fine-tune VGG16 block4 + block5
DENSENET_TRAINABLE_FROM = "conv5_block1_0_bn"  # fine-tune DenseNet121 dense block 5

# Regularisation.
LABEL_SMOOTHING = 0.1
L2_WEIGHT = 1e-4
MIXUP_PROBABILITY = 0.5   # Fraction of training batches that receive MixUp.
MIXUP_ALPHA = 0.2         # Beta(alpha, alpha) mixing distribution.

# Plateau-based LR reduction applied on top of cosine decay.
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 2

# Optional early stopping (default False => full 20 epochs are trained).
USE_EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 6

# Checkpoint selection metric (same as V1).
CHECKPOINT_MONITOR = "val_loss"

# Horizontal-flip test-time augmentation (reported as extra metrics;
# the primary metrics remain single-pass for a fair baseline comparison).
USE_TTA = True

# Save the training/evaluation figures as PNG files (also shown on screen).
SAVE_PLOTS = True

# Grad-CAM sanity check on a few test images after evaluation.
RUN_GRADCAM_DEMO = True
GRADCAM_LAYER = "cbam_refined_features"   # alternatives: "block5_conv3", "relu"
GRADCAM_SAMPLES = 6

# IMPORTANT: Change BASE_PATH to the folder containing Non Demented and Very mild Dementia.
BASE_PATH = DATA_DIR

# Output directory for model, CSV files, and figures.
OUTPUT_DIR = OUT_DIR

MODEL_NAME = "HED-XAI-ALZHEIMER-V2"
FILE_PREFIX = "HED_XAI_ALZHEIMER_V2"
CLASS_NAMES = ["Non Demented", "Very mild Dementia"]
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_Best.h5")
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_Final.h5")
RESULTS_CSV = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_Final_Metrics.csv")
REPORT_CSV = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_Classification_Report.csv")
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_Test_Predictions.csv")
SPLIT_MANIFEST = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_Split_Manifest.csv")
TRAINING_LOG = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_Training_Log.csv")
CONFIG_FILE = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_Experiment_Config.json")
HISTORY_CSV = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_History.csv")
SUMMARY_TXT = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_Model_Summary.txt")

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 72)
print("HED-XAI-ALZHEIMER-V2: VGG16 + DenseNet121 HYBRID WITH CBAM ATTENTION")
print("=" * 72)
print("TensorFlow version:", tf.__version__)
print("GPUs available:", len(tf.config.list_physical_devices("GPU")))
print("Dataset:", BASE_PATH)
print("Output:", OUTPUT_DIR)
print("Epochs:", EPOCHS)


def save_current_figure(file_name):
    """Save the active matplotlib figure into OUTPUT_DIR when enabled."""
    if SAVE_PLOTS:
        plt.savefig(os.path.join(OUTPUT_DIR, file_name), dpi=300, bbox_inches="tight")


# ============================================================
# 2. VALIDATE DATASET AND LOAD IMAGE PATHS
# ============================================================

if not os.path.isdir(BASE_PATH):
    raise FileNotFoundError(
        f"\nDataset folder not found:\n{BASE_PATH}\n\n"
        "Edit BASE_PATH in Section 1 so it points to the folder "
        "that contains the Non Demented and Very mild Dementia folders."
    )

for class_name in CLASS_NAMES:
    class_folder = os.path.join(BASE_PATH, class_name)
    if not os.path.isdir(class_folder):
        raise FileNotFoundError(
            f"Expected class folder was not found: {class_folder}\n"
            f"Your dataset must contain folders named exactly: {CLASS_NAMES}"
        )

records = []
for label, class_name in enumerate(CLASS_NAMES):
    class_folder = os.path.join(BASE_PATH, class_name)
    for file_path in sorted(Path(class_folder).rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
            records.append({
                "image_path": str(file_path),
                "label": label,
                "class_name": class_name,
            })

df = pd.DataFrame(records)
if df.empty:
    raise ValueError("No images were found. Check BASE_PATH and file extensions.")

print("\nOriginal image count:", len(df))
print(df["class_name"].value_counts().to_string())

if df["label"].nunique() != 2:
    raise ValueError("Both Non Demented and Very mild Dementia classes must contain images.")


# ============================================================
# 3. REMOVE BYTE-IDENTICAL DUPLICATES BEFORE SPLITTING
# ============================================================
# SHA-256 removes exact file duplicates only. It does not detect all
# visually similar/recompressed images and does not ensure patient-level
# independence. If patient IDs are available, prefer patient-level splitting.

def sha256_file(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


print("\nCalculating SHA-256 hashes to identify exact duplicate files...")
df["sha256"] = df["image_path"].apply(sha256_file)

hash_label_counts = df.groupby("sha256")["label"].nunique()
conflicting_hashes = hash_label_counts[hash_label_counts > 1]
if len(conflicting_hashes) > 0:
    raise ValueError(
        "Identical files have conflicting class labels. Please review the labels."
    )

original_count = len(df)
df = df.drop_duplicates(subset="sha256", keep="first").reset_index(drop=True)
duplicates_removed = original_count - len(df)

print("Exact duplicate files removed:", duplicates_removed)
print("Unique images retained:", len(df))
print(df["class_name"].value_counts().to_string())


# ============================================================
# 4. STRATIFIED 70 / 15 / 15 SPLIT
# ============================================================
# If images from the same patient can be identified, use patient-level
# splitting instead. Image-level splitting may overestimate performance
# when the same patient contributes multiple images.

train_df, temporary_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["label"],
)

val_df, test_df = train_test_split(
    temporary_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temporary_df["label"],
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("\nDataset split:")
for split_name, split_df in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df),
]:
    print(f"\n{split_name}: {len(split_df)} images")
    print(split_df["class_name"].value_counts().to_string())

# Verify exact files do not overlap.
assert set(train_df["sha256"]).isdisjoint(set(val_df["sha256"]))
assert set(train_df["sha256"]).isdisjoint(set(test_df["sha256"]))
assert set(val_df["sha256"]).isdisjoint(set(test_df["sha256"]))

manifest = pd.concat(
    [
        train_df.assign(split="train"),
        val_df.assign(split="validation"),
        test_df.assign(split="test"),
    ],
    ignore_index=True,
)
manifest.to_csv(SPLIT_MANIFEST, index=False)
print("\nSplit manifest saved:", SPLIT_MANIFEST)


# ============================================================
# 5. IMAGE GENERATORS (+ LABEL-PRESERVING MIXUP)
# ============================================================
# The model contains its own backbone-specific ImageNet preprocessing,
# so the generators deliver RAW RGB pixel values in the range [0, 255].
# Augmentation is applied only to training images.

AUGMENTATION = {
    "rotation_range": 10,
    "width_shift_range": 0.06,
    "height_shift_range": 0.06,
    "zoom_range": 0.10,
    "horizontal_flip": True,
    "brightness_range": [0.90, 1.10],
    "fill_mode": "constant",
    "cval": 0.0,              # MRI background is black.
}

train_datagen = ImageDataGenerator(**AUGMENTATION)
evaluation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_generator = evaluation_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_generator = evaluation_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=False,
)


class LabelPreservingMixupSequence(tf.keras.utils.Sequence):
    """Wrap a Keras image iterator and apply intra-class MixUp.

    Images are mixed only with partners of the SAME class, so labels stay
    hard (0/1). This keeps MixUp's regularisation effect while avoiding
    artificial "half-demented" images and keeps training accuracy, AUC,
    precision and recall correctly computed.
    """

    def __init__(self, base_iterator, probability, alpha, seed):
        super().__init__()
        self.base_iterator = base_iterator
        self.probability = probability
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base_iterator)

    def __getitem__(self, index):
        images, labels = self.base_iterator[index]
        images = images.astype(np.float32)
        labels = np.asarray(labels, dtype=np.float32)

        if self.probability > 0 and self.rng.random() < self.probability:
            mixed = images.copy()
            for class_value in np.unique(labels):
                class_idx = np.where(labels == class_value)[0]
                if len(class_idx) < 2:
                    continue
                partners = self.rng.permutation(class_idx)
                lam = self.rng.beta(self.alpha, self.alpha, size=len(class_idx))
                lam = np.maximum(lam, 1.0 - lam).astype(np.float32)
                lam = lam[:, None, None, None]
                mixed[class_idx] = lam * images[class_idx] + (1.0 - lam) * images[partners]
            images = mixed

        return images, labels

    def on_epoch_end(self):
        self.base_iterator.on_epoch_end()


train_sequence = LabelPreservingMixupSequence(
    train_generator,
    probability=MIXUP_PROBABILITY,
    alpha=MIXUP_ALPHA,
    seed=SEED,
)


# ============================================================
# 6. BUILD THE PROPOSED DUAL-BACKBONE HYBRID MODEL
# ============================================================
# HED-XAI-ALZHEIMER-V2 prototype:
#   (a) VGG16 + DenseNet121 pretrained feature extractors (shared input)
#   (b) multi-scale global features from both backbones
#   (c) deep spatial fusion of the final 7x7 maps (concat + 1x1 projection)
#   (d) CBAM attention: channel attention followed by spatial attention
#   (e) regularised dense binary classification head
#
# Backbone layers are built directly on the shared input tensor (not as
# nested sub-models), so every convolutional layer is reachable with
# model.get_layer(name) -- this is what makes Grad-CAM straightforward.
#
# This architecture is a proposed research design. Its novelty and value
# must be established by literature review and controlled ablation studies.

print(f"\nBuilding {MODEL_NAME} model...")

IMAGENET_MEAN_RGB = [123.68, 116.779, 103.939]
IMAGENET_STD_RGB = [0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0]


def channel_attention(feature_map, reduction_ratio=16, name="cbam_channel"):
    """CBAM channel attention: shared MLP over average- and max-pooled features."""
    channels = int(feature_map.shape[-1])
    hidden_units = max(8, channels // reduction_ratio)

    shared_dense_1 = Dense(
        hidden_units,
        activation="relu",
        kernel_initializer="he_normal",
        name=f"{name}_mlp_hidden",
    )
    shared_dense_2 = Dense(
        channels,
        kernel_initializer="he_normal",
        name=f"{name}_mlp_output",
    )

    avg_pool = GlobalAveragePooling2D(name=f"{name}_avg_pool")(feature_map)
    max_pool = GlobalMaxPooling2D(name=f"{name}_max_pool")(feature_map)

    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))

    weights = Add(name=f"{name}_add")([avg_out, max_out])
    weights = Activation("sigmoid", name=f"{name}_sigmoid")(weights)
    weights = Reshape((1, 1, channels), name=f"{name}_reshape")(weights)
    return Multiply(name=f"{name}_refinement")([feature_map, weights])


def spatial_attention(feature_map, kernel_size=7, name="cbam_spatial"):
    """CBAM spatial attention using only built-in, serialisable Keras layers.

    Channel-wise mean and max are computed by reshaping (H, W, C) to
    (C, H*W) and applying 1-D global pooling over the channel axis.
    """
    height = int(feature_map.shape[1])
    width = int(feature_map.shape[2])
    channels = int(feature_map.shape[3])

    flat = Reshape((height * width, channels), name=f"{name}_flatten")(feature_map)
    flat = Permute((2, 1), name=f"{name}_permute")(flat)          # (C, H*W)

    avg_map = GlobalAveragePooling1D(name=f"{name}_channel_mean")(flat)
    max_map = GlobalMaxPooling1D(name=f"{name}_channel_max")(flat)
    avg_map = Reshape((height, width, 1), name=f"{name}_mean_map")(avg_map)
    max_map = Reshape((height, width, 1), name=f"{name}_max_map")(max_map)

    descriptor = Concatenate(axis=-1, name=f"{name}_descriptor")([avg_map, max_map])
    attention_map = Conv2D(
        1,
        kernel_size,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        name=f"{name}_attention_map",
    )(descriptor)
    return Multiply(name="cbam_refined_features")([feature_map, attention_map])


inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="alzheimer_image_input")

# --- Backbone-specific preprocessing (weight-free, stored in the model) ---
# VGG16: ImageNet mean subtraction. The original VGG16 expects BGR input,
# so the first convolution kernel is channel-flipped below to accept RGB.
vgg_input = Normalization(
    mean=IMAGENET_MEAN_RGB,
    variance=[1.0, 1.0, 1.0],
    name="vgg16_imagenet_preprocess",
)(inputs)

# DenseNet121: (x / 255 - mean) / std  ==  (x - 255*mean) / (255*std)
densenet_input = Normalization(
    mean=[0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0],
    variance=[s ** 2 for s in IMAGENET_STD_RGB],
    name="densenet121_imagenet_preprocess",
)(inputs)

# --- Primary backbone: VGG16 ---
vgg_backbone = VGG16(
    weights="imagenet",
    include_top=False,
    input_tensor=vgg_input,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
first_conv = vgg_backbone.get_layer("block1_conv1")
kernel, bias = first_conv.get_weights()
first_conv.set_weights([kernel[:, :, ::-1, :].copy(), bias])   # BGR -> RGB weights

# --- Secondary backbone: DenseNet121 ---
densenet_backbone = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_tensor=densenet_input,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)


def set_trainable_from(backbone, start_layer_name):
    """Freeze the backbone, then unfreeze non-BatchNorm layers from start_layer_name."""
    layer_names = [layer.name for layer in backbone.layers]
    if start_layer_name not in layer_names:
        raise ValueError(f"Layer '{start_layer_name}' not found in {backbone.name}.")
    start_index = layer_names.index(start_layer_name)

    backbone.trainable = True
    for index, layer in enumerate(backbone.layers):
        is_batch_norm = isinstance(layer, BatchNormalization)
        # BatchNorm stays frozen (inference mode) for stable transfer learning;
        # preprocessing Normalization layers are always frozen.
        layer.trainable = (
            index >= start_index
            and not is_batch_norm
            and not isinstance(layer, Normalization)
        )


set_trainable_from(vgg_backbone, VGG16_TRAINABLE_FROM)
set_trainable_from(densenet_backbone, DENSENET_TRAINABLE_FROM)

# --- Multi-scale feature maps ---
vgg_scale_layers = ["block3_conv3", "block4_conv3", "block5_conv3"]   # 56x56, 28x28, 14x14
densenet_scale_layers = ["pool3_relu", "pool4_relu"]                  # 28x28, 14x14

multi_scale_maps = (
    [(f"vgg_{name}", vgg_backbone.get_layer(name).output) for name in vgg_scale_layers]
    + [(f"densenet_{name}", densenet_backbone.get_layer(name).output)
       for name in densenet_scale_layers]
)

scale_embeddings = []
for scale_name, feature_map in multi_scale_maps:
    pooled = GlobalAveragePooling2D(name=f"gap_{scale_name}")(feature_map)
    projected = Dense(
        128,
        activation="relu",
        kernel_regularizer=l2(L2_WEIGHT),
        name=f"proj_{scale_name}",
    )(pooled)
    scale_embeddings.append(projected)

multi_scale_vector = Concatenate(name="multi_scale_feature_fusion")(scale_embeddings)

# --- Deep spatial fusion of the final 7x7 maps from both backbones ---
vgg_deep = vgg_backbone.get_layer("block5_pool").output        # 7x7x512
densenet_deep = densenet_backbone.get_layer("relu").output     # 7x7x1024

deep_fused = Concatenate(axis=-1, name="dual_backbone_spatial_fusion")(
    [vgg_deep, densenet_deep]
)
deep_fused = Conv2D(
    512,
    1,
    padding="same",
    use_bias=False,
    kernel_initializer="he_normal",
    kernel_regularizer=l2(L2_WEIGHT),
    name="fusion_projection_conv",
)(deep_fused)
# LayerNormalization (not BatchNorm) in the new layers: it has no moving
# statistics, so training and validation/inference behave identically even
# while the fine-tuned backbones change their feature scale.
deep_fused = LayerNormalization(name="fusion_projection_ln")(deep_fused)
deep_fused = Activation("relu", name="fusion_projection_relu")(deep_fused)

# --- CBAM attention on the fused features (Grad-CAM target layer) ---
refined = channel_attention(deep_fused, reduction_ratio=16, name="cbam_channel")
refined = spatial_attention(refined, kernel_size=7, name="cbam_spatial")

deep_vector = Concatenate(name="fused_global_descriptor")([
    GlobalAveragePooling2D(name="fused_gap")(refined),
    GlobalMaxPooling2D(name="fused_gmp")(refined),
])

# --- Classification head ---
x = Concatenate(name="hybrid_feature_vector")([deep_vector, multi_scale_vector])
x = LayerNormalization(name="head_layer_norm")(x)
x = Dropout(0.40, name="head_dropout_1")(x)
x = Dense(512, activation="relu", kernel_regularizer=l2(L2_WEIGHT), name="classifier_dense_1")(x)
x = Dropout(0.40, name="head_dropout_2")(x)
x = Dense(128, activation="relu", kernel_regularizer=l2(L2_WEIGHT), name="classifier_dense_2")(x)
x = Dropout(0.30, name="head_dropout_3")(x)
outputs = Dense(1, activation="sigmoid", name="very_mild_dementia_probability")(x)

model = Model(inputs=inputs, outputs=outputs, name="HED_XAI_ALZHEIMER_V2")

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

# The full summary (~500 layers) is written to a text file.
with open(SUMMARY_TXT, "w", encoding="utf-8") as summary_file:
    model.summary(print_fn=lambda line, *args, **kwargs: summary_file.write(line + "\n"))

total_params_built = model.count_params()
trainable_params_built = int(sum(np.prod(v.shape) for v in model.trainable_weights))
print(f"Total layers:          {len(model.layers)}")
print(f"Total parameters:      {total_params_built:,}")
print(f"Trainable parameters:  {trainable_params_built:,}")
print("Full model summary saved:", SUMMARY_TXT)


# ============================================================
# 7. SAVE CONFIGURATION AND CALLBACKS
# ============================================================

ARCHITECTURE_TEXT = (
    "VGG16 + DenseNet121 dual-backbone; multi-scale features "
    "(VGG16 block3/4/5, DenseNet121 pool3/pool4); deep 7x7 spatial fusion "
    "(concatenation + 1x1 learned projection); CBAM channel + spatial "
    "attention; regularised dense binary classifier"
)

config = {
    "model_name": MODEL_NAME,
    "architecture": ARCHITECTURE_TEXT,
    "input_size": [IMG_SIZE, IMG_SIZE, 3],
    "input_range": "raw RGB 0-255 (preprocessing is inside the model)",
    "classes": {"Non Demented": 0, "Very mild Dementia": 1},
    "split": "Stratified 70/15/15 after SHA-256 exact duplicate removal",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "optimizer": "Adam",
    "learning_rate_peak": LEARNING_RATE,
    "learning_rate_min": MIN_LEARNING_RATE,
    "lr_schedule": (
        f"{WARMUP_EPOCHS}-epoch linear warm-up + cosine decay, multiplied by "
        f"plateau factor {PLATEAU_FACTOR} after {PLATEAU_PATIENCE} epochs "
        "without val_loss improvement"
    ),
    "loss": f"Binary cross-entropy with label smoothing {LABEL_SMOOTHING}",
    "l2_weight": L2_WEIGHT,
    "vgg16_trainable_from": VGG16_TRAINABLE_FROM,
    "densenet121_trainable_from": DENSENET_TRAINABLE_FROM,
    "backbone_batch_norm_layers": "frozen (inference mode)",
    "new_layer_normalisation": "LayerNormalization (fusion projection and head)",
    "mixup": {
        "type": "label-preserving (intra-class) MixUp",
        "probability": MIXUP_PROBABILITY,
        "alpha": MIXUP_ALPHA,
    },
    "early_stopping": USE_EARLY_STOPPING,
    "checkpoint_monitor": CHECKPOINT_MONITOR,
    "threshold": THRESHOLD,
    "test_time_augmentation": "horizontal flip (extra metrics)" if USE_TTA else "off",
    "seed": SEED,
    "preprocessing": (
        "Inside model: VGG16 ImageNet mean subtraction (RGB-adapted first conv); "
        "DenseNet121 ImageNet mean/std normalisation"
    ),
    "augmentation": AUGMENTATION,
    "gradcam_target_layers": ["cbam_refined_features", "block5_conv3", "relu"],
    "tensorflow_version": tf.__version__,
}

with open(CONFIG_FILE, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=4)


class WarmupCosinePlateauScheduler(Callback):
    """Linear warm-up -> cosine decay, with an extra plateau-based reduction.

    ReduceLROnPlateau and a cosine schedule normally overwrite each other;
    this callback combines both: lr = cosine(epoch) * plateau_scale.
    """

    def __init__(self, base_lr, min_lr, total_epochs, warmup_epochs,
                 monitor="val_loss", factor=0.5, patience=2, verbose=1):
        super().__init__()
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.plateau_scale = 1.0
        self.best = np.inf
        self.wait = 0
        self.current_lr = base_lr

    def _cosine_lr(self, epoch):
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        decay_epochs = max(1, self.total_epochs - self.warmup_epochs)
        progress = min(1.0, (epoch - self.warmup_epochs) / decay_epochs)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def _set_lr(self, value):
        lr_attribute = self.model.optimizer.learning_rate
        if hasattr(lr_attribute, "assign"):
            lr_attribute.assign(value)
        else:
            self.model.optimizer.learning_rate = value

    def on_epoch_begin(self, epoch, logs=None):
        self.current_lr = max(self.min_lr, self._cosine_lr(epoch) * self.plateau_scale)
        self._set_lr(self.current_lr)
        if self.verbose:
            print(f"\nEpoch {epoch + 1}: learning rate = {self.current_lr:.2e}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        logs["learning_rate"] = float(self.current_lr)
        current = logs.get(self.monitor)
        if current is None:
            return
        if current < self.best:
            self.best = current
            self.wait = 0
        elif epoch >= self.warmup_epochs:
            self.wait += 1
            if self.wait >= self.patience:
                self.plateau_scale *= self.factor
                self.wait = 0
                if self.verbose:
                    print(f"Epoch {epoch + 1}: {self.monitor} plateau -> "
                          f"LR scale reduced to {self.plateau_scale:.3f}")


monitor_mode = "min" if "loss" in CHECKPOINT_MONITOR else "max"

checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    monitor=CHECKPOINT_MONITOR,
    save_best_only=True,
    mode=monitor_mode,
    verbose=1,
)

lr_scheduler = WarmupCosinePlateauScheduler(
    base_lr=LEARNING_RATE,
    min_lr=MIN_LEARNING_RATE,
    total_epochs=EPOCHS,
    warmup_epochs=WARMUP_EPOCHS,
    monitor="val_loss",
    factor=PLATEAU_FACTOR,
    patience=PLATEAU_PATIENCE,
)

csv_logger = CSVLogger(TRAINING_LOG, append=False)
callbacks = [lr_scheduler, checkpoint, csv_logger]

if USE_EARLY_STOPPING:
    callbacks.append(EarlyStopping(
        monitor=CHECKPOINT_MONITOR,
        mode=monitor_mode,
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ))


# ============================================================
# 8. TRAIN FOR 20 EPOCHS
# ============================================================
# EarlyStopping is disabled by default, so training runs for all 20
# epochs unless a runtime error or interruption occurs.

print("\nStarting training...")
training_start = time.perf_counter()

history = model.fit(
    train_sequence,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

training_seconds = time.perf_counter() - training_start
actual_epochs = len(history.history["loss"])
if monitor_mode == "min":
    best_epoch = int(np.argmin(history.history[CHECKPOINT_MONITOR])) + 1
else:
    best_epoch = int(np.argmax(history.history[CHECKPOINT_MONITOR])) + 1

print("\nTraining completed.")
print("Epochs completed:", actual_epochs)
print(f"Best {CHECKPOINT_MONITOR} epoch:", best_epoch)
print(f"Training time: {training_seconds:.2f} seconds")
print(f"Training time: {training_seconds / 60:.2f} minutes")

model.save(FINAL_MODEL_PATH)


# ============================================================
# 9. DISPLAY TRAINING HISTORY AND SAVE HISTORY CSV
# ============================================================

history_df = pd.DataFrame(history.history)
history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
history_df.to_csv(HISTORY_CSV, index=False)

epochs_axis = np.arange(1, actual_epochs + 1)
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

axes[0].plot(epochs_axis, history.history["accuracy"], label="Training accuracy")
axes[0].plot(epochs_axis, history.history["val_accuracy"], label="Validation accuracy")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs_axis, history.history["loss"], label="Training loss")
axes[1].plot(epochs_axis, history.history["val_loss"], label="Validation loss")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

if "auc" in history.history:
    axes[2].plot(epochs_axis, history.history["auc"], label="Training AUC")
    axes[2].plot(epochs_axis, history.history["val_auc"], label="Validation AUC")
axes[2].set_title("AUC")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("AUC")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.suptitle(f"{MODEL_NAME} Training Curves")
plt.tight_layout()
save_current_figure(f"{FILE_PREFIX}_Training_Curves.png")
plt.show()


# ============================================================
# 10. LOAD BEST VALIDATION CHECKPOINT
# ============================================================

print("\nLoading best checkpoint:", BEST_MODEL_PATH)
best_model = load_model(BEST_MODEL_PATH)


# ============================================================
# 11. TEST-SET EVALUATION AND INFERENCE TIMING
# ============================================================

print("\nEvaluating the held-out test set...")
test_generator.reset()
evaluation = best_model.evaluate(test_generator, verbose=1, return_dict=True)
test_loss = float(evaluation["loss"])

# Warm-up for a more stable inference timing estimate.
test_generator.reset()
warmup_images, _ = next(test_generator)
_ = best_model.predict(warmup_images[:min(4, len(warmup_images))], verbose=0)

test_generator.reset()
inference_start = time.perf_counter()
y_prob = best_model.predict(test_generator, verbose=1).ravel()
inference_seconds = time.perf_counter() - inference_start

y_true = test_df["label"].to_numpy(dtype=int)
if len(y_true) != len(y_prob):
    raise ValueError(
        f"Prediction count ({len(y_prob)}) does not match test labels ({len(y_true)})."
    )

y_pred = (y_prob >= THRESHOLD).astype(int)
test_count = len(y_true)
inference_ms_per_image = (inference_seconds / test_count) * 1000
throughput = test_count / inference_seconds if inference_seconds > 0 else 0.0


def predict_with_flip_tta(trained_model, generator):
    """Average predictions of original and horizontally flipped images."""
    generator.reset()
    probabilities = []
    for batch_index in range(len(generator)):
        images, _ = generator[batch_index]
        flipped = np.ascontiguousarray(images[:, :, ::-1, :])
        p_original = np.asarray(trained_model(images, training=False)).ravel()
        p_flipped = np.asarray(trained_model(flipped, training=False)).ravel()
        probabilities.append((p_original + p_flipped) / 2.0)
    return np.concatenate(probabilities)


# ============================================================
# 12. CLASSIFICATION METRICS
# ============================================================

def compute_binary_metrics(true_labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(true_labels, predictions, labels=[0, 1])
    tn_, fp_, fn_, tp_ = matrix.ravel()
    return {
        "accuracy": accuracy_score(true_labels, predictions),
        "precision": precision_score(true_labels, predictions, zero_division=0),
        "sensitivity": recall_score(true_labels, predictions, zero_division=0),
        "specificity": tn_ / (tn_ + fp_) if (tn_ + fp_) else 0.0,
        "npv": tn_ / (tn_ + fn_) if (tn_ + fn_) else 0.0,
        "f1": f1_score(true_labels, predictions, zero_division=0),
        "roc_auc": (roc_auc_score(true_labels, probabilities)
                    if len(np.unique(true_labels)) == 2 else float("nan")),
    }


cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
sensitivity = recall_score(y_true, y_pred, zero_division=0)
specificity = tn / (tn + fp) if (tn + fp) else 0.0
npv = tn / (tn + fn) if (tn + fn) else 0.0
f1 = f1_score(y_true, y_pred, zero_division=0)

if len(np.unique(y_true)) == 2:
    roc_auc = roc_auc_score(y_true, y_prob)
    average_precision = average_precision_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(pr_recall, pr_precision)
else:
    roc_auc = float("nan")
    average_precision = float("nan")
    fpr, tpr = np.array([]), np.array([])
    pr_precision, pr_recall = np.array([]), np.array([])
    pr_auc = float("nan")

print("\n" + "=" * 72)
print(f"FINAL TEST RESULTS -- {MODEL_NAME}")
print("=" * 72)
print(f"Test loss:            {test_loss:.4f}")
print(f"Accuracy:             {accuracy * 100:.2f}%")
print(f"Precision:            {precision * 100:.2f}%")
print(f"Sensitivity/Recall:   {sensitivity * 100:.2f}%")
print(f"Specificity:          {specificity * 100:.2f}%")
print(f"Negative predictive value: {npv * 100:.2f}%")
print(f"F1-score:             {f1:.4f}")
print(f"ROC-AUC:              {roc_auc:.4f}")
print(f"PR-AUC (trapezoidal): {pr_auc:.4f}")
print(f"Average precision:    {average_precision:.4f}")
print("Confusion matrix (rows=actual, columns=predicted):")
print(cm)
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
)

report_dict = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
    output_dict=True,
)
pd.DataFrame(report_dict).transpose().to_csv(REPORT_CSV)

# Optional test-time augmentation (reported separately, not the primary result).
tta_metrics = None
y_prob_tta = None
if USE_TTA:
    print("\nRunning horizontal-flip test-time augmentation...")
    y_prob_tta = predict_with_flip_tta(best_model, test_generator)
    tta_metrics = compute_binary_metrics(y_true, y_prob_tta, THRESHOLD)
    print(f"TTA accuracy:         {tta_metrics['accuracy'] * 100:.2f}%")
    print(f"TTA F1-score:         {tta_metrics['f1']:.4f}")
    print(f"TTA ROC-AUC:          {tta_metrics['roc_auc']:.4f}")


# ============================================================
# 13. SAVE TEST PREDICTIONS
# ============================================================

predictions_df = test_df.copy()
predictions_df["actual_class"] = predictions_df["label"].map({
    0: "Non Demented",
    1: "Very mild Dementia",
})
predictions_df["predicted_label"] = y_pred
predictions_df["predicted_class"] = [
    CLASS_NAMES[int(value)] for value in y_pred
]
predictions_df["very_mild_dementia_probability"] = y_prob
predictions_df["correct_prediction"] = (
    predictions_df["label"].to_numpy() == y_pred
)
if y_prob_tta is not None:
    predictions_df["tta_very_mild_dementia_probability"] = y_prob_tta
    predictions_df["tta_predicted_label"] = (y_prob_tta >= THRESHOLD).astype(int)
predictions_df.to_csv(PREDICTIONS_CSV, index=False)


# ============================================================
# 14. DISPLAY CONFUSION MATRIX, ROC AND PRECISION-RECALL CURVES
# ============================================================

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title(f"{MODEL_NAME} Confusion Matrix")
plt.tight_layout()
save_current_figure(f"{FILE_PREFIX}_Confusion_Matrix.png")
plt.show()

if len(fpr) > 0:
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{MODEL_NAME} ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_current_figure(f"{FILE_PREFIX}_ROC_Curve.png")
    plt.show()

    plt.figure(figsize=(7, 6))
    plt.plot(pr_recall, pr_precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{MODEL_NAME} Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_current_figure(f"{FILE_PREFIX}_PR_Curve.png")
    plt.show()


# ============================================================
# 15. MODEL SIZE, PARAMETERS AND FINAL METRICS TABLE
# ============================================================

total_params = best_model.count_params()
trainable_params = int(sum(
    np.prod(variable.shape) for variable in best_model.trainable_weights
))
non_trainable_params = int(sum(
    np.prod(variable.shape) for variable in best_model.non_trainable_weights
))
model_size_mb = os.path.getsize(BEST_MODEL_PATH) / (1024 ** 2)

results = [
    ("Model", MODEL_NAME),
    ("Architecture", "VGG16 + DenseNet121 dual-backbone multi-scale fusion + CBAM attention"),
    ("Positive class", "Very mild Dementia"),
    ("Negative class", "Non Demented"),
    ("Input size", f"{IMG_SIZE} x {IMG_SIZE} x 3"),
    ("Requested epochs", EPOCHS),
    ("Epochs completed", actual_epochs),
    (f"Best {CHECKPOINT_MONITOR} epoch", best_epoch),
    ("Batch size", BATCH_SIZE),
    ("Original image files", original_count),
    ("Exact duplicates removed", duplicates_removed),
    ("Unique images retained", len(df)),
    ("Training images", len(train_df)),
    ("Validation images", len(val_df)),
    ("Test images", len(test_df)),
    ("Test loss (label-smoothed BCE)", test_loss),
    ("Test accuracy (%)", accuracy * 100),
    ("Precision (%)", precision * 100),
    ("Sensitivity (%)", sensitivity * 100),
    ("Specificity (%)", specificity * 100),
    ("Negative predictive value (%)", npv * 100),
    ("F1-score", f1),
    ("ROC-AUC", roc_auc),
    ("PR-AUC", pr_auc),
    ("Average precision", average_precision),
    ("True negative", int(tn)),
    ("False positive", int(fp)),
    ("False negative", int(fn)),
    ("True positive", int(tp)),
]

if tta_metrics is not None:
    results += [
        ("TTA (h-flip) accuracy (%)", tta_metrics["accuracy"] * 100),
        ("TTA (h-flip) precision (%)", tta_metrics["precision"] * 100),
        ("TTA (h-flip) sensitivity (%)", tta_metrics["sensitivity"] * 100),
        ("TTA (h-flip) specificity (%)", tta_metrics["specificity"] * 100),
        ("TTA (h-flip) F1-score", tta_metrics["f1"]),
        ("TTA (h-flip) ROC-AUC", tta_metrics["roc_auc"]),
    ]

results += [
    ("Total parameters", int(total_params)),
    ("Trainable parameters", trainable_params),
    ("Non-trainable parameters", non_trainable_params),
    ("Best model size (MB)", model_size_mb),
    ("Training time (seconds)", training_seconds),
    ("Training time (minutes)", training_seconds / 60),
    ("Test inference time (seconds)", inference_seconds),
    ("Inference time per image (ms)", inference_ms_per_image),
    ("Inference time per image (seconds)", inference_seconds / test_count if test_count else 0.0),
    ("Inference throughput (images/second)", throughput),
    ("Decision threshold", THRESHOLD),
    ("Random seed", SEED),
]
results_df = pd.DataFrame(results, columns=["Metric", "Value"])
results_df.to_csv(RESULTS_CSV, index=False)

print("\nFinal metrics table:")
print(results_df.to_string(index=False))


# ============================================================
# 16. GRAD-CAM SANITY CHECK (EXPLAINABILITY)
# ============================================================
# Model input = raw RGB 0-255 image resized to IMG_SIZE (no external
# preprocessing). Any Grad-CAM tool can use the same convention.

def make_gradcam_heatmap(trained_model, image_batch, layer_name):
    """Return (heatmap in [0,1], probability) for the predicted class."""
    grad_model = Model(
        inputs=trained_model.inputs,
        outputs=[trained_model.get_layer(layer_name).output, trained_model.output],
    )
    image_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(image_tensor, training=False)
        probability = prediction[:, 0]
        # Explain the predicted class (positive if p >= threshold).
        score = tf.where(probability >= THRESHOLD, probability, 1.0 - probability)
    gradients = tape.gradient(score, conv_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_output[0] * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), float(probability.numpy()[0])


def load_raw_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    return np.asarray(image, dtype=np.float32)


if RUN_GRADCAM_DEMO and len(test_df) > 0:
    print(f"\nGenerating Grad-CAM examples from layer '{GRADCAM_LAYER}'...")
    sample_df = (
        test_df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max(1, GRADCAM_SAMPLES // 2)), random_state=SEED))
        .reset_index(drop=True)
    )
    fig, axes = plt.subplots(2, len(sample_df), figsize=(3 * len(sample_df), 6.5), squeeze=False)
    for column, row in sample_df.iterrows():
        raw_image = load_raw_image(row["image_path"])
        heatmap, probability = make_gradcam_heatmap(best_model, raw_image[None, ...], GRADCAM_LAYER)
        heatmap_resized = np.asarray(
            Image.fromarray(np.uint8(255 * heatmap)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        ) / 255.0
        predicted_name = CLASS_NAMES[int(probability >= THRESHOLD)]

        axes[0, column].imshow(raw_image.astype(np.uint8))
        axes[0, column].set_title(f"Actual: {row['class_name']}", fontsize=8)
        axes[0, column].axis("off")
        axes[1, column].imshow(raw_image.astype(np.uint8))
        axes[1, column].imshow(heatmap_resized, cmap="jet", alpha=0.40)
        axes[1, column].set_title(f"Pred: {predicted_name} (P={probability:.3f})", fontsize=7)
        axes[1, column].axis("off")
    plt.suptitle(f"{MODEL_NAME} Grad-CAM ({GRADCAM_LAYER})")
    plt.tight_layout()
    save_current_figure(f"{FILE_PREFIX}_GradCAM_Examples.png")
    plt.show()


# ============================================================
# 17. RESULTS SUMMARY
# ============================================================

print("\nSaved outputs:")
for path in [
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    RESULTS_CSV,
    REPORT_CSV,
    PREDICTIONS_CSV,
    SPLIT_MANIFEST,
    TRAINING_LOG,
    CONFIG_FILE,
    HISTORY_CSV,
    SUMMARY_TXT,
]:
    print(" -", path)

print("\n" + "=" * 72)
print(f"SUMMARY -- {MODEL_NAME}")
print("=" * 72)
print(f"Requested epochs: {EPOCHS}")
print(f"Completed epochs: {actual_epochs}")
print(f"Best checkpoint epoch ({CHECKPOINT_MONITOR}): {best_epoch}")
print(f"Test accuracy (single pass): {accuracy * 100:.2f}%")
if tta_metrics is not None:
    print(f"Test accuracy (h-flip TTA):  {tta_metrics['accuracy'] * 100:.2f}%")
print(f"Sensitivity: {sensitivity * 100:.2f}% | Specificity: {specificity * 100:.2f}% | "
      f"F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
print(f"Total training time: {training_seconds:.2f} seconds ({training_seconds / 60:.2f} minutes)")
print(f"Mean inference time per image: {inference_ms_per_image:.4f} ms")
print(f"Parameters: {int(total_params):,} total | {trainable_params:,} trainable")
print(f"Results directory: {OUTPUT_DIR}")
print("\nTraining and evaluation script completed.")
if SAVE_PLOTS:
    print("Evaluation plots were displayed and saved as PNG files in the results directory.")
else:
    print("Evaluation plots are displayed on screen only; no plot images are saved.")
print("Use the best checkpoint for Grad-CAM analysis (feed raw 0-255 RGB images).")
print("Research note: validate the proposed contribution with ablations and external data.")
