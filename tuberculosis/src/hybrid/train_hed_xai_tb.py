"""HED-XAI-TB (Protocol B): SHA-256 de-duplication, stratified 70/15/15 split (seed 42), MobileNetV2 multi-scale fusion + channel attention, 10 epochs (set EPOCHS = 20 for the confirmation run), full test evaluation.

Extracted from the executed notebook `notebooks/08_protocolB_HED-XAI-TB_hybrid_10_and_20_epochs.ipynb` (cell 0); the model definition and the
training/evaluation code are unchanged. Only hard-coded paths were replaced by the environment variables
below, and figures are written to disk instead of being displayed.

    TB_DATA_DIR    folder containing Normal/ and Tuberculosis/ of the TB Chest Radiography Database
                   (default: data/TB_Chest_Radiography_Database)
    TB_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/HED-XAI-TB)

Run from the `tuberculosis/` folder:  python src/hybrid/train_hed_xai_tb.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("TB_DATA_DIR", "data/TB_Chest_Radiography_Database"))
OUT_DIR = os.path.abspath(os.environ.get("TB_OUTPUT_DIR", "outputs/HED-XAI-TB"))
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
# HED-XAI-TB: Hybrid Efficient Deep Learning for Tuberculosis
# Classification from Chest X-ray Images
#
# Proposed architecture:
#   MobileNetV2 multi-scale features
#   + channel-attention feature refinement
#   + feature fusion
#   + binary classifier
#
# This is a research prototype, not a clinically validated tool.
# Fixed training duration: 10 epochs.
#
# Expected dataset structure:
# dataset/
#   Normal/
#       image1.png
#       ...
#   Tuberculosis/
#       image1.png
#       ...
#
# Install packages in your TensorFlow environment:
# pip install tensorflow numpy pandas matplotlib scikit-learn pillow
# ============================================================

import os
import json
import time
import random
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

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
    GlobalAveragePooling2D,
    Concatenate,
    Multiply,
    Reshape,
    BatchNormalization,
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


# ============================================================
# 1. CONFIGURATION -- EDIT THESE TWO PATHS
# ============================================================

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16           # Use 8 if RAM is limited; 32 if your computer supports it.
EPOCHS = 10               # Exactly 10 epochs; no early stopping.
LEARNING_RATE = 1e-4
FINE_TUNE_LAST_LAYERS = 30
THRESHOLD = 0.5

# IMPORTANT: Change BASE_PATH to the folder containing Normal and Tuberculosis.
BASE_PATH = DATA_DIR

# Output directory for model, CSV files, and figures.
OUTPUT_DIR = OUT_DIR

CLASS_NAMES = ["Normal", "Tuberculosis"]
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "HED_XAI_TB_Best.h5")
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "HED_XAI_TB_Final.h5")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "HED_XAI_TB_Final_Metrics.csv")
REPORT_CSV = os.path.join(OUTPUT_DIR, "HED_XAI_TB_Classification_Report.csv")
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "HED_XAI_TB_Test_Predictions.csv")
SPLIT_MANIFEST = os.path.join(OUTPUT_DIR, "HED_XAI_TB_Split_Manifest.csv")
TRAINING_LOG = os.path.join(OUTPUT_DIR, "HED_XAI_TB_Training_Log.csv")
CONFIG_FILE = os.path.join(OUTPUT_DIR, "HED_XAI_TB_Experiment_Config.json")

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 72)
print("HED-XAI-TB: HYBRID MULTI-SCALE ATTENTION MODEL")
print("=" * 72)
print("TensorFlow version:", tf.__version__)
print("Dataset:", BASE_PATH)
print("Output:", OUTPUT_DIR)
print("Epochs:", EPOCHS)


# ============================================================
# 2. VALIDATE DATASET AND LOAD IMAGE PATHS
# ============================================================

if not os.path.isdir(BASE_PATH):
    raise FileNotFoundError(
        f"\nDataset folder not found:\n{BASE_PATH}\n\n"
        "Edit BASE_PATH in Section 1 so it points to the folder "
        "that contains the Normal and Tuberculosis folders."
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
    raise ValueError("Both Normal and Tuberculosis classes must contain images.")


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
# 5. IMAGE GENERATORS
# ============================================================
# MobileNetV2 preprocess_input maps image values to its expected range.
# Augmentation is applied only to training images.

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=8,
    width_shift_range=0.04,
    height_shift_range=0.04,
    zoom_range=0.08,
    horizontal_flip=True,
    fill_mode="nearest",
)

evaluation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

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


# ============================================================
# 6. BUILD THE PROPOSED HYBRID MODEL
# ============================================================
# HED-XAI-TB prototype:
#   (a) MobileNetV2 pretrained feature extractor
#   (b) multi-scale global feature extraction from three stages
#   (c) concatenation-based feature fusion
#   (d) channel-attention gate over fused features
#   (e) dense binary classification head
#
# This architecture is a proposed research design. Its novelty and value
# must be established by literature review and controlled ablation studies.

print("\nBuilding HED-XAI-TB model...")

backbone = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

# Use outputs from early/middle/deep feature stages.
feature_layer_names = [
    "block_3_expand_relu",
    "block_6_expand_relu",
    "out_relu",
]
feature_outputs = [backbone.get_layer(name).output for name in feature_layer_names]
feature_extractor = Model(
    inputs=backbone.input,
    outputs=feature_outputs,
    name="MobileNetV2_MultiScale_Extractor",
)

# Freeze backbone, then fine-tune the final selected layers.
backbone.trainable = True
for layer in backbone.layers:
    layer.trainable = False

for layer in backbone.layers[-FINE_TUNE_LAST_LAYERS:]:
    # Keep BatchNorm layers frozen for more stable transfer learning.
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="chest_xray_input")

# training=False keeps pretrained BatchNorm statistics stable.
multi_scale_maps = feature_extractor(inputs, training=False)

pooled_features = [
    GlobalAveragePooling2D(name=f"global_pool_scale_{i+1}")(feature_map)
    for i, feature_map in enumerate(multi_scale_maps)
]

fused = Concatenate(name="multi_scale_feature_fusion")(pooled_features)
fused = BatchNormalization(name="fusion_batch_norm")(fused)
fused = Dense(256, activation="relu", name="fusion_projection")(fused)
fused = Dropout(0.35, name="fusion_dropout")(fused)

# Channel-attention gate: learn a feature-wise importance vector.
attention_hidden_units = max(16, 256 // 8)
attention = Dense(
    attention_hidden_units,
    activation="relu",
    name="channel_attention_hidden",
)(fused)
attention = Dense(
    256,
    activation="sigmoid",
    name="channel_attention_weights",
)(attention)
attended = Multiply(name="channel_attention_refinement")([fused, attention])

x = Dense(128, activation="relu", name="classifier_dense")(attended)
x = Dropout(0.25, name="classifier_dropout")(x)
outputs = Dense(1, activation="sigmoid", name="tuberculosis_probability")(x)

model = Model(inputs=inputs, outputs=outputs, name="HED_XAI_TB")

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

model.summary()


# ============================================================
# 7. SAVE CONFIGURATION AND CALLBACKS
# ============================================================

config = {
    "model_name": "HED-XAI-TB",
    "architecture": (
        "MobileNetV2 multi-scale feature extraction, concatenation-based "
        "feature fusion, channel-attention refinement, binary classifier"
    ),
    "input_size": [IMG_SIZE, IMG_SIZE, 3],
    "classes": {"Normal": 0, "Tuberculosis": 1},
    "split": "Stratified 70/15/15 after SHA-256 exact duplicate removal",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "optimizer": "Adam",
    "learning_rate": LEARNING_RATE,
    "fine_tune_last_layers": FINE_TUNE_LAST_LAYERS,
    "threshold": THRESHOLD,
    "seed": SEED,
    "preprocessing": "MobileNetV2 preprocess_input",
    "augmentation": {
        "rotation_range": 8,
        "width_shift_range": 0.04,
        "height_shift_range": 0.04,
        "zoom_range": 0.08,
        "horizontal_flip": True,
    },
}

with open(CONFIG_FILE, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=4)

checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    mode="min",
    verbose=1,
)

csv_logger = CSVLogger(TRAINING_LOG, append=False)
callbacks = [checkpoint, reduce_lr, csv_logger]


# ============================================================
# 8. TRAIN FOR EXACTLY 10 EPOCHS
# ============================================================
# No EarlyStopping callback is used, so training runs for all 10 epochs
# unless a runtime error or interruption occurs.

print("\nStarting training...")
training_start = time.perf_counter()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

training_seconds = time.perf_counter() - training_start
actual_epochs = len(history.history["loss"])
best_epoch = int(np.argmin(history.history["val_loss"])) + 1

print("\nTraining completed.")
print("Epochs completed:", actual_epochs)
print("Best validation-loss epoch:", best_epoch)
print(f"Training time: {training_seconds:.2f} seconds")
print(f"Training time: {training_seconds / 60:.2f} minutes")

model.save(FINAL_MODEL_PATH)


# ============================================================
# 9. SAVE TRAINING HISTORY FIGURE AND CSV
# ============================================================

history_df = pd.DataFrame(history.history)
history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
history_df.to_csv(
    os.path.join(OUTPUT_DIR, "HED_XAI_TB_History.csv"),
    index=False,
)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

axes[0].plot(history.history["accuracy"], label="Training accuracy")
axes[0].plot(history.history["val_accuracy"], label="Validation accuracy")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history["loss"], label="Training loss")
axes[1].plot(history.history["val_loss"], label="Validation loss")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

if "auc" in history.history:
    axes[2].plot(history.history["auc"], label="Training AUC")
    axes[2].plot(history.history["val_auc"], label="Validation AUC")
axes[2].set_title("AUC")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("AUC")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "HED_XAI_TB_Training_History.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()


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
test_loss, test_accuracy, test_auc_keras, test_precision_keras, test_recall_keras = (
    best_model.evaluate(test_generator, verbose=1)
)

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


# ============================================================
# 12. CLASSIFICATION METRICS
# ============================================================

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
print("FINAL TEST RESULTS -- HED-XAI-TB")
print("=" * 72)
print(f"Test loss:            {test_loss:.4f}")
print(f"Accuracy:             {accuracy * 100:.2f}%")
print(f"Precision:            {precision * 100:.2f}%")
print(f"Sensitivity/Recall:   {sensitivity * 100:.2f}%")
print(f"Specificity:           {specificity * 100:.2f}%")
print(f"Negative predictive value: {npv * 100:.2f}%")
print(f"F1-score:              {f1:.4f}")
print(f"ROC-AUC:               {roc_auc:.4f}")
print(f"PR-AUC (trapezoidal):  {pr_auc:.4f}")
print(f"Average precision:     {average_precision:.4f}")
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


# ============================================================
# 13. SAVE TEST PREDICTIONS
# ============================================================

predictions_df = test_df.copy()
predictions_df["actual_class"] = predictions_df["label"].map({
    0: "Normal",
    1: "Tuberculosis",
})
predictions_df["predicted_label"] = y_pred
predictions_df["predicted_class"] = [
    CLASS_NAMES[int(value)] for value in y_pred
]
predictions_df["tuberculosis_probability"] = y_prob
predictions_df["correct_prediction"] = (
    predictions_df["label"].to_numpy() == y_pred
)
predictions_df.to_csv(PREDICTIONS_CSV, index=False)


# ============================================================
# 14. SAVE CONFUSION MATRIX, ROC AND PRECISION-RECALL CURVES
# ============================================================

plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("HED-XAI-TB Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(CLASS_NAMES))
plt.xticks(tick_marks, CLASS_NAMES, rotation=20)
plt.yticks(tick_marks, CLASS_NAMES)
threshold_value = cm.max() / 2.0 if cm.size else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, format(cm[i, j], "d"),
            ha="center", va="center",
            color="white" if cm[i, j] > threshold_value else "black",
        )
plt.ylabel("Actual class")
plt.xlabel("Predicted class")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "HED_XAI_TB_Confusion_Matrix.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

if len(fpr) > 0:
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("HED-XAI-TB ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "HED_XAI_TB_ROC_Curve.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.plot(pr_recall, pr_precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("HED-XAI-TB Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "HED_XAI_TB_PR_Curve.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


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
    ("Model", "HED-XAI-TB"),
    ("Architecture", "MobileNetV2 multi-scale feature fusion + channel attention"),
    ("Positive class", "Tuberculosis"),
    ("Negative class", "Normal"),
    ("Input size", f"{IMG_SIZE} x {IMG_SIZE} x 3"),
    ("Requested epochs", EPOCHS),
    ("Epochs completed", actual_epochs),
    ("Best validation-loss epoch", best_epoch),
    ("Batch size", BATCH_SIZE),
    ("Original image files", original_count),
    ("Exact duplicates removed", duplicates_removed),
    ("Unique images retained", len(df)),
    ("Training images", len(train_df)),
    ("Validation images", len(val_df)),
    ("Test images", len(test_df)),
    ("Test loss", float(test_loss)),
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
    ("Total parameters", int(total_params)),
    ("Trainable parameters", trainable_params),
    ("Non-trainable parameters", non_trainable_params),
    ("Best model size (MB)", model_size_mb),
    ("Training time (seconds)", training_seconds),
    ("Training time (minutes)", training_seconds / 60),
    ("Test inference time (seconds)", inference_seconds),
    ("Inference time per image (ms)", inference_ms_per_image),
    ("Inference throughput (images/second)", throughput),
    ("Decision threshold", THRESHOLD),
    ("Random seed", SEED),
]
results_df = pd.DataFrame(results, columns=["Metric", "Value"])
results_df.to_csv(RESULTS_CSV, index=False)

print("\nFinal metrics table:")
print(results_df.to_string(index=False))

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
    os.path.join(OUTPUT_DIR, "HED_XAI_TB_History.csv"),
    os.path.join(OUTPUT_DIR, "HED_XAI_TB_Training_History.png"),
    os.path.join(OUTPUT_DIR, "HED_XAI_TB_Confusion_Matrix.png"),
    os.path.join(OUTPUT_DIR, "HED_XAI_TB_ROC_Curve.png"),
    os.path.join(OUTPUT_DIR, "HED_XAI_TB_PR_Curve.png"),
]:
    print(" -", path)

print("\nTraining and evaluation script completed.")
print("Use the best checkpoint for Grad-CAM analysis.")
print("Research note: validate the proposed contribution with ablations and external data.")
