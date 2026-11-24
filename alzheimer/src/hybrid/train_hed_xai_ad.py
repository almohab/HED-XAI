"""HED-XAI-AD (final): VGG16 + InceptionV3 dual-backbone hybrid with channel attention, two-stage training (10 + 40 epochs).

Extracted from the executed notebook `02_HED-XAI-AD_hybrid_models_training_and_evaluation.ipynb` (cell 1); the model definition and the
training/evaluation code are unchanged. Only hard-coded Windows paths were replaced by the
environment variables below, and figures are written to disk instead of being displayed.

    AD_DATA_DIR    folder containing "Non Demented/" and "Very mild Dementia/" (default: data/balanced_data)
    AD_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/HED-XAI-AD)

Run from the `alzheimer/` folder:  python src/hybrid/train_hed_xai_ad.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("AD_DATA_DIR", "data/balanced_data"))
OUT_DIR = os.path.abspath(os.environ.get("AD_OUTPUT_DIR", "outputs/HED-XAI-AD"))
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


# ================================================================
# VGG16 + INCEPTIONV3 HYBRID ATTENTION MODEL
# ALZHEIMER'S MRI BINARY CLASSIFICATION
#
# Class 0: Non Demented
# Class 1: Very mild Dementia
#
# Architecture:
# VGG16 + InceptionV3 + Feature Fusion
# + Channel Attention + Binary Classification
#
# Dataset split:
# 70% Training / 15% Validation / 15% Testing
#
# Stage 1: 10 epochs - Frozen backbones
# Stage 2: 40 epochs - Fine-tuning
#
# Evaluation:
# Accuracy, Precision, Sensitivity, Specificity,
# F1-score, ROC-AUC, PR-AUC, Confusion Matrix,
# Inference Time, Model Size and Training Time.
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import os
import time
import random
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    auc
)

from tensorflow.keras import Model

from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Concatenate,
    BatchNormalization,
    Multiply,
    Resizing
)

from tensorflow.keras.applications import (
    VGG16,
    InceptionV3
)

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
    CSVLogger
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import load_model


# ================================================================
# 2. CONFIGURATION
# ================================================================

SEED = 42

IMG_SIZE = 224
BATCH_SIZE = 8

EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 40

LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2 = 1e-5

THRESHOLD = 0.5

# Dataset path
BASE_PATH = DATA_DIR

# Output directory
OUTPUT_DIR = OUT_DIR

# Class names
CLASS_NAMES = [
    "Non Demented",
    "Very mild Dementia"
]

# Supported image extensions
VALID_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model paths
BEST_STAGE1_PATH = os.path.join(
    OUTPUT_DIR,
    "Best_Stage1.keras"
)

BEST_STAGE2_PATH = os.path.join(
    OUTPUT_DIR,
    "Best_Stage2.keras"
)

BEST_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "VGG16_InceptionV3_Hybrid_Best.keras"
)

FINAL_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "VGG16_InceptionV3_Hybrid_Final.keras"
)

# Result paths
RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "Final_Metrics.csv"
)

REPORT_CSV = os.path.join(
    OUTPUT_DIR,
    "Classification_Report.csv"
)

PREDICTIONS_CSV = os.path.join(
    OUTPUT_DIR,
    "Test_Predictions.csv"
)

HISTORY_CSV = os.path.join(
    OUTPUT_DIR,
    "Training_History.csv"
)

SPLIT_CSV = os.path.join(
    OUTPUT_DIR,
    "Split_Manifest.csv"
)


# ================================================================
# 3. REPRODUCIBILITY AND DEVICE CONFIGURATION
# ================================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Enable GPU memory growth if available

for gpu in tf.config.list_physical_devices("GPU"):

    try:
        tf.config.experimental.set_memory_growth(
            gpu, True
        )

    except RuntimeError:
        pass


print("=" * 70)
print("VGG16 + INCEPTIONV3 HYBRID ATTENTION MODEL")
print("=" * 70)

print("TensorFlow version:", tf.__version__)

print(
    "Available GPUs:",
    tf.config.list_physical_devices("GPU")
)

print("Dataset:", BASE_PATH)
print("Output:", OUTPUT_DIR)


# ================================================================
# 4. CUSTOM SERIALIZABLE PREPROCESSING LAYERS
# ================================================================

@tf.keras.utils.register_keras_serializable(
    package="AlzheimerHybrid"
)
class VGG16Preprocessing(tf.keras.layers.Layer):

    def call(self, inputs):

        return tf.keras.applications.vgg16.preprocess_input(
            inputs
        )


@tf.keras.utils.register_keras_serializable(
    package="AlzheimerHybrid"
)
class InceptionV3Preprocessing(tf.keras.layers.Layer):

    def call(self, inputs):

        return tf.keras.applications.inception_v3.preprocess_input(
            inputs
        )


# ================================================================
# 5. CHECK DATASET AND LOAD IMAGE PATHS
# ================================================================

if not os.path.isdir(BASE_PATH):

    raise FileNotFoundError(
        f"Dataset directory not found: {BASE_PATH}"
    )

records = []

for label, class_name in enumerate(CLASS_NAMES):

    class_folder = os.path.join(
        BASE_PATH,
        class_name
    )

    if not os.path.isdir(class_folder):

        raise FileNotFoundError(
            f"Missing class folder: {class_folder}"
        )

    for filename in sorted(os.listdir(class_folder)):

        if filename.lower().endswith(VALID_EXTENSIONS):

            image_path = os.path.join(
                class_folder,
                filename
            )

            records.append({
                "image_path": image_path,
                "label": label,
                "class_name": class_name
            })


df = pd.DataFrame(records)

if len(df) == 0:

    raise ValueError(
        "No valid images found in the dataset."
    )

if df["label"].nunique() != 2:

    raise ValueError(
        "Both classes must contain images."
    )


print("\nTotal images:", len(df))

print("\nClass distribution:")

print(df["class_name"].value_counts())


# ================================================================
# 6. REMOVE EXACT DUPLICATE FILES
# ================================================================
# Only byte-identical files are removed.
# Modified or augmented versions are not detected.
# ================================================================

def calculate_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):

            sha256.update(chunk)

    return sha256.hexdigest()


print("\nChecking exact duplicate files...")

df["image_hash"] = df["image_path"].apply(
    calculate_hash
)

# Check if identical files have different labels

label_conflicts = (
    df.groupby("image_hash")["label"].nunique()
)

if (label_conflicts > 1).any():

    raise ValueError(
        "Identical files have conflicting class labels. "
        "Please inspect the dataset."
    )

original_count = len(df)

df = df.drop_duplicates(
    subset="image_hash",
    keep="first"
).reset_index(drop=True)

print(
    "Exact duplicate files removed:",
    original_count - len(df)
)

print("Remaining images:", len(df))


# ================================================================
# 7. STRATIFIED DATASET SPLITTING
# ================================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=SEED
)

train_df = train_df.reset_index(drop=True)

val_df = val_df.reset_index(drop=True)

test_df = test_df.reset_index(drop=True)


print("\nDATASET SPLITTING")
print("-" * 50)

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))

print("\nTraining distribution:")
print(train_df["class_name"].value_counts())

print("\nValidation distribution:")
print(val_df["class_name"].value_counts())

print("\nTesting distribution:")
print(test_df["class_name"].value_counts())


# Save dataset split manifest

manifest = pd.concat(
    [
        train_df.assign(split="train"),
        val_df.assign(split="validation"),
        test_df.assign(split="test")
    ],
    ignore_index=True
)

manifest.to_csv(
    SPLIT_CSV,
    index=False
)


# ================================================================
# 8. IMAGE DATA GENERATORS
# ================================================================

# Images remain in the 0-255 pixel range.
# Each backbone uses its own preprocessing layer.

train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.04,
    height_shift_range=0.04,
    zoom_range=0.08,
    fill_mode="nearest"
)

evaluation_datagen = ImageDataGenerator()


train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col=["label"],
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


val_gen = evaluation_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="image_path",
    y_col=["label"],
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=False
)


test_gen = evaluation_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col=["label"],
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Verify generator output

sample_images, sample_labels = next(train_gen)

print("\nImage batch shape:", sample_images.shape)
print("Label batch shape:", sample_labels.shape)

if sample_labels.ndim != 2 or sample_labels.shape[1] != 1:

    raise ValueError(
        "Expected labels with shape (batch_size, 1)."
    )

train_gen.reset()


# ================================================================
# 9. BUILD HYBRID MODEL
# ================================================================

print("\nBuilding hybrid model...")

inputs = Input(
    shape=(IMG_SIZE, IMG_SIZE, 3),
    name="MRI_Input"
)


# ================================================================
# 10. VGG16 BRANCH
# ================================================================

print("Loading VGG16...")

vgg_base = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

vgg_base.trainable = False

vgg_input = VGG16Preprocessing(
    name="VGG16_Preprocessing"
)(inputs)

vgg_features = vgg_base(
    vgg_input,
    training=False
)

vgg_features = GlobalAveragePooling2D(
    name="VGG16_Global_Average_Pooling"
)(vgg_features)


# ================================================================
# 11. INCEPTIONV3 BRANCH
# ================================================================

print("Loading InceptionV3...")

inception_base = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_shape=(299, 299, 3)
)

inception_base.trainable = False

inception_input = Resizing(
    height=299,
    width=299,
    name="Inception_Resize"
)(inputs)

inception_input = InceptionV3Preprocessing(
    name="InceptionV3_Preprocessing"
)(inception_input)

inception_features = inception_base(
    inception_input,
    training=False
)

inception_features = GlobalAveragePooling2D(
    name="Inception_Global_Average_Pooling"
)(inception_features)


# ================================================================
# 12. FEATURE FUSION
# ================================================================

print("Combining VGG16 and InceptionV3 features...")

fused_features = Concatenate(
    name="Hybrid_Feature_Fusion"
)(
    [
        vgg_features,
        inception_features
    ]
)

fused_features = BatchNormalization(
    name="Fusion_Batch_Normalization"
)(fused_features)

fused_features = Dense(
    512,
    activation="relu",
    name="Fusion_Dense"
)(fused_features)

fused_features = Dropout(
    0.40,
    name="Fusion_Dropout"
)(fused_features)


# ================================================================
# 13. CHANNEL ATTENTION
# ================================================================

attention = Dense(
    64,
    activation="relu",
    name="Attention_Hidden"
)(fused_features)

attention = Dense(
    512,
    activation="sigmoid",
    name="Attention_Weights"
)(attention)

attended_features = Multiply(
    name="Channel_Attention"
)(
    [
        fused_features,
        attention
    ]
)


# ================================================================
# 14. CLASSIFICATION HEAD
# ================================================================

x = Dense(
    256,
    activation="relu",
    name="Classifier_Dense_1"
)(attended_features)

x = BatchNormalization(
    name="Classifier_Batch_Normalization"
)(x)

x = Dropout(
    0.35,
    name="Classifier_Dropout_1"
)(x)

x = Dense(
    128,
    activation="relu",
    name="Classifier_Dense_2"
)(x)

x = Dropout(
    0.25,
    name="Classifier_Dropout_2"
)(x)

outputs = Dense(
    1,
    activation="sigmoid",
    name="Binary_Output"
)(x)


model = Model(
    inputs=inputs,
    outputs=outputs,
    name="VGG16_InceptionV3_Hybrid_Attention"
)

model.summary()


# ================================================================
# 15. STAGE 1 CALLBACKS
# ================================================================

stage1_callbacks = [

    ModelCheckpoint(
        filepath=BEST_STAGE1_PATH,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),

    CSVLogger(
        os.path.join(
            OUTPUT_DIR,
            "Stage1_Training_Log.csv"
        )
    )
]


# ================================================================
# 16. STAGE 1: TRAIN WITH FROZEN BACKBONES
# ================================================================

print("\n" + "=" * 70)
print("STAGE 1: FROZEN BACKBONE TRAINING")
print("=" * 70)

model.compile(
    optimizer=Adam(
        learning_rate=LEARNING_RATE_STAGE1
    ),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        tf.keras.metrics.AUC(
            name="auc",
            curve="ROC"
        )
    ]
)

training_start = time.perf_counter()

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE1,
    callbacks=stage1_callbacks,
    verbose=1
)


# ================================================================
# 17. STAGE 2: FINE-TUNING
# ================================================================

print("\n" + "=" * 70)
print("STAGE 2: FINE-TUNING")
print("=" * 70)


# ------------------------------------------------
# Unfreeze final VGG16 convolutional layers
# ------------------------------------------------

vgg_base.trainable = True

for layer in vgg_base.layers:

    layer.trainable = False

for layer in vgg_base.layers:

    if layer.name in [
        "block5_conv1",
        "block5_conv2",
        "block5_conv3"
    ]:

        layer.trainable = True


# ------------------------------------------------
# Unfreeze selected InceptionV3 layers
# ------------------------------------------------

inception_base.trainable = True

for layer in inception_base.layers:

    layer.trainable = False

for layer in inception_base.layers[-30:]:

    if not isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = True


print(
    "Trainable VGG16 layers:",
    sum(
        layer.trainable
        for layer in vgg_base.layers
    )
)

print(
    "Trainable InceptionV3 layers:",
    sum(
        layer.trainable
        for layer in inception_base.layers
    )
)


# Recompile after changing trainable layers

model.compile(
    optimizer=Adam(
        learning_rate=LEARNING_RATE_STAGE2
    ),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        tf.keras.metrics.AUC(
            name="auc",
            curve="ROC"
        )
    ]
)


# Stage 2 callbacks

stage2_callbacks = [

    ModelCheckpoint(
        filepath=BEST_STAGE2_PATH,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),

    CSVLogger(
        os.path.join(
            OUTPUT_DIR,
            "Stage2_Training_Log.csv"
        )
    )
]


history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE2,
    callbacks=stage2_callbacks,
    verbose=1
)


training_time = (
    time.perf_counter() - training_start
)

print(
    "\nTotal training time:",
    round(training_time / 60, 2),
    "minutes"
)


# ================================================================
# 18. SELECT BEST CHECKPOINT USING VALIDATION LOSS
# ================================================================
# The independent test set is not used for model selection.
# ================================================================

print("\nComparing Stage 1 and Stage 2 checkpoints...")

stage1_model = load_model(
    BEST_STAGE1_PATH,
    compile=False
)

stage1_model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy"
)

val_gen.reset()

stage1_val_loss = stage1_model.evaluate(
    val_gen,
    verbose=0
)


stage2_model = load_model(
    BEST_STAGE2_PATH,
    compile=False
)

stage2_model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy"
)

val_gen.reset()

stage2_val_loss = stage2_model.evaluate(
    val_gen,
    verbose=0
)


print(
    "Stage 1 validation loss:",
    round(float(stage1_val_loss), 5)
)

print(
    "Stage 2 validation loss:",
    round(float(stage2_val_loss), 5)
)


if stage1_val_loss <= stage2_val_loss:

    print("\nSelected checkpoint: Stage 1")

    best_model = stage1_model

else:

    print("\nSelected checkpoint: Stage 2")

    best_model = stage2_model


# Save selected best model

best_model.save(BEST_MODEL_PATH)

best_model.save(FINAL_MODEL_PATH)

print("\nBest model saved successfully.")


# ================================================================
# 19. COMBINE TRAINING HISTORY
# ================================================================

combined_history = {}

history_keys = set(
    history1.history.keys()
).intersection(
    set(history2.history.keys())
)

for key in history_keys:

    combined_history[key] = (
        history1.history[key] +
        history2.history[key]
    )


history_df = pd.DataFrame(
    combined_history
)

history_df.insert(
    0,
    "Epoch",
    np.arange(1, len(history_df) + 1)
)

history_df.to_csv(
    HISTORY_CSV,
    index=False
)


# ================================================================
# 20. TRAINING AND VALIDATION CURVES
# ================================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)


# Accuracy

axes[0].plot(
    history_df["accuracy"],
    label="Training Accuracy"
)

axes[0].plot(
    history_df["val_accuracy"],
    label="Validation Accuracy"
)

axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)


# Loss

axes[1].plot(
    history_df["loss"],
    label="Training Loss"
)

axes[1].plot(
    history_df["val_loss"],
    label="Validation Loss"
)

axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)


# ROC-AUC

axes[2].plot(
    history_df["auc"],
    label="Training AUC"
)

axes[2].plot(
    history_df["val_auc"],
    label="Validation AUC"
)

axes[2].set_title("ROC-AUC")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("AUC")
axes[2].legend()
axes[2].grid(alpha=0.3)


plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Training_Curves.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ================================================================
# 21. TEST SET EVALUATION
# ================================================================

print("\n" + "=" * 70)
print("INDEPENDENT TEST SET EVALUATION")
print("=" * 70)

test_gen.reset()

test_evaluation = best_model.evaluate(
    test_gen,
    verbose=1
)

# Fixed after the original run: evaluate() returns a float when only the loss is compiled
test_loss = float(test_evaluation[0]) if isinstance(test_evaluation, (list, tuple)) else float(test_evaluation)


# ================================================================
# 22. PREDICTIONS AND END-TO-END INFERENCE TIME
# ================================================================

# Warm-up prediction

test_gen.reset()

warmup_images, _ = next(test_gen)

_ = best_model(
    warmup_images[:1],
    training=False
).numpy()

# Measure full generator-to-prediction time

test_gen.reset()

inference_start = time.perf_counter()

y_prob = best_model.predict(
    test_gen,
    verbose=1
).ravel()

inference_time = (
    time.perf_counter() - inference_start
)

y_true = test_df["label"].to_numpy(
    dtype=int
)

if len(y_true) != len(y_prob):

    raise ValueError(
        "Prediction count does not match test labels."
    )

y_pred = (
    y_prob >= THRESHOLD
).astype(int)


# ================================================================
# 23. MODEL-ONLY INFERENCE TIME
# ================================================================
# This measures forward-pass time on an already-loaded batch.
# It excludes image loading and generator preprocessing.
# ================================================================

test_gen.reset()

inference_images, _ = next(test_gen)

# Warm-up

_ = best_model(
    inference_images,
    training=False
).numpy()

REPEAT_INFERENCE = 30

model_inference_start = time.perf_counter()

for _ in range(REPEAT_INFERENCE):

    batch_predictions = best_model(
        inference_images,
        training=False
    )

    # Synchronize execution before measuring elapsed time
    _ = batch_predictions.numpy()

model_inference_total = (
    time.perf_counter() - model_inference_start
)

model_only_ms_per_image = (
    model_inference_total /
    (
        REPEAT_INFERENCE *
        len(inference_images)
    )
) * 1000


# End-to-end time per image

number_of_images = len(test_df)

end_to_end_ms_per_image = (
    inference_time / number_of_images
) * 1000

throughput = (
    number_of_images / inference_time
    if inference_time > 0
    else 0
)


# ================================================================
# 24. CONFUSION MATRIX AND CLASSIFICATION METRICS
# ================================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# Accuracy

accuracy = accuracy_score(
    y_true,
    y_pred
)


# Precision for positive class

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)


# Sensitivity = Recall of Very mild Dementia class

sensitivity = recall_score(
    y_true,
    y_pred,
    zero_division=0
)


# Specificity = Recall of Non Demented class

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)


# F1-score

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# ROC-AUC

roc_auc = roc_auc_score(
    y_true,
    y_prob
)


# Average precision

average_precision = average_precision_score(
    y_true,
    y_prob
)


# ROC curve

fpr, tpr, roc_thresholds = roc_curve(
    y_true,
    y_prob
)


# Precision-recall curve

pr_precision, pr_recall, pr_thresholds = (
    precision_recall_curve(
        y_true,
        y_prob
    )
)

pr_auc = auc(
    pr_recall,
    pr_precision
)


# ================================================================
# 25. DISPLAY FINAL METRICS
# ================================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"Test Loss:                  {test_loss:.4f}")

print(f"Accuracy:                   {accuracy * 100:.2f}%")

print(f"Precision:                  {precision * 100:.2f}%")

print(f"Sensitivity:                {sensitivity * 100:.2f}%")

print(f"Specificity:                {specificity * 100:.2f}%")

print(f"F1-score:                   {f1:.4f}")

print(f"ROC-AUC:                    {roc_auc:.4f}")

print(f"PR-AUC:                     {pr_auc:.4f}")

print(f"Average Precision:          {average_precision:.4f}")


print("\nCONFUSION MATRIX")

print(cm)

print("\nTrue Negative:", tn)

print("False Positive:", fp)

print("False Negative:", fn)

print("True Positive:", tp)


# ================================================================
# 26. CLASSIFICATION REPORT
# ================================================================

print("\nCLASSIFICATION REPORT")

report_text = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print(report_text)

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(
    report_dict
).transpose()

report_df.to_csv(
    REPORT_CSV
)


# ================================================================
# 27. CONFUSION MATRIX FIGURE
# ================================================================

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    annot_kws={"size": 14}
)

plt.title(
    "VGG16-InceptionV3 Hybrid Attention Model"
)

plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Confusion_Matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ================================================================
# 28. ROC CURVE
# ================================================================

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}",
    linewidth=2
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("Receiver Operating Characteristic Curve")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ROC_Curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ================================================================
# 29. PRECISION-RECALL CURVE
# ================================================================

plt.figure(figsize=(7, 6))

plt.plot(
    pr_recall,
    pr_precision,
    label=f"PR-AUC = {pr_auc:.4f}",
    linewidth=2
)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Precision_Recall_Curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ================================================================
# 30. MODEL SIZE AND PARAMETERS
# ================================================================

model_size_mb = (
    os.path.getsize(BEST_MODEL_PATH)
    / (1024 ** 2)
)

total_params = best_model.count_params()

trainable_params = sum(
    int(np.prod(variable.shape))
    for variable in best_model.trainable_weights
)

non_trainable_params = sum(
    int(np.prod(variable.shape))
    for variable in best_model.non_trainable_weights
)


print("\nMODEL INFORMATION")

print(
    "Model size:",
    round(model_size_mb, 2),
    "MB"
)

print("Total parameters:", total_params)

print("Trainable parameters:", trainable_params)

print("Non-trainable parameters:", non_trainable_params)


# ================================================================
# 31. INFERENCE PERFORMANCE
# ================================================================

print("\nINFERENCE PERFORMANCE")

print(
    "End-to-end prediction time:",
    round(inference_time, 4),
    "seconds"
)

print(
    "End-to-end time per image:",
    round(end_to_end_ms_per_image, 4),
    "ms"
)

print(
    "Model-only inference time per image:",
    round(model_only_ms_per_image, 4),
    "ms"
)

print(
    "End-to-end throughput:",
    round(throughput, 2),
    "images/second"
)


# ================================================================
# 32. SAVE TEST PREDICTIONS
# ================================================================

predictions_df = test_df.copy()

predictions_df["Actual_Class"] = [
    CLASS_NAMES[int(value)]
    for value in y_true
]

predictions_df["Predicted_Class"] = [
    CLASS_NAMES[int(value)]
    for value in y_pred
]

predictions_df["Prediction_Probability"] = y_prob

predictions_df["Correct"] = (
    y_true == y_pred
)

predictions_df.to_csv(
    PREDICTIONS_CSV,
    index=False
)


# ================================================================
# 33. SAVE FINAL METRICS
# ================================================================

results = [

    ("Model", "VGG16-InceptionV3 Hybrid Attention"),

    ("Input Size", "224 x 224 x 3"),

    ("Training Images", len(train_df)),

    ("Validation Images", len(val_df)),

    ("Testing Images", len(test_df)),

    ("Maximum Epochs", EPOCHS_STAGE1 + EPOCHS_STAGE2),

    ("Test Loss", test_loss),

    ("Accuracy (%)", accuracy * 100),

    ("Precision (%)", precision * 100),

    ("Sensitivity (%)", sensitivity * 100),

    ("Specificity (%)", specificity * 100),

    ("F1-score", f1),

    ("ROC-AUC", roc_auc),

    ("PR-AUC", pr_auc),

    ("Average Precision", average_precision),

    ("Training Time (seconds)", training_time),

    ("End-to-End Inference Time (ms/image)",
     end_to_end_ms_per_image),

    ("Model-Only Inference Time (ms/image)",
     model_only_ms_per_image),

    ("End-to-End Throughput (images/second)",
     throughput),

    ("Model Size (MB)", model_size_mb),

    ("Total Parameters", total_params),

    ("Trainable Parameters", trainable_params),

    ("Non-trainable Parameters", non_trainable_params),

    ("True Negative", tn),

    ("False Positive", fp),

    ("False Negative", fn),

    ("True Positive", tp),

    ("Classification Threshold", THRESHOLD),

    ("Stage 1 Validation Loss", float(stage1_val_loss)),

    ("Stage 2 Validation Loss", float(stage2_val_loss))
]


results_df = pd.DataFrame(
    results,
    columns=["Metric", "Value"]
)

print("\n" + "=" * 70)
print("FINAL METRICS TABLE")
print("=" * 70)

print(
    results_df.to_string(index=False)
)

results_df.to_csv(
    RESULTS_CSV,
    index=False
)


# ================================================================
# 34. EXPERIMENT COMPLETED
# ================================================================

print("\n" + "=" * 70)

print("EXPERIMENT COMPLETED SUCCESSFULLY")

print("=" * 70)

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print("\nSAVED MODELS:")
print(BEST_MODEL_PATH)
print(FINAL_MODEL_PATH)

print("\nSAVED CSV FILES:")
print(RESULTS_CSV)
print(REPORT_CSV)
print(PREDICTIONS_CSV)
print(HISTORY_CSV)
print(SPLIT_CSV)

print("\nSAVED FIGURES:")
print("1. Training_Curves.png")
print("2. Confusion_Matrix.png")
print("3. ROC_Curve.png")
print("4. Precision_Recall_Curve.png")

print("\nAll experiment outputs have been saved.")

print("=" * 70)
