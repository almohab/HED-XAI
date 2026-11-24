"""VGG16 (frozen), 20 epochs, re-run with ModelCheckpoint and full evaluation (reported model).

Extracted from the executed notebook `notebooks/05_VGG16_20_epochs_with_checkpoint_rerun.ipynb` (cell 6); the model definition and the
training/evaluation code are unchanged. Only hard-coded Windows paths were replaced by the
environment variables below, and figures are written to disk instead of being displayed.

    BT_DATA_DIR    folder with train/, val/ and test/ sub-folders, each containing glioma/,
                   meningioma/ and pituitary/ (default: data/converted_images_split)
    BT_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/vgg16_20)

Run from the `brain_tumor/` folder:  python src/baselines/vgg16_20_epochs.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("BT_DATA_DIR", "data/converted_images_split"))
OUT_DIR = os.path.abspath(os.environ.get("BT_OUTPUT_DIR", "outputs/vgg16_20"))
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
# VGG16 TRANSFER LEARNING
# 3-CLASS BRAIN TUMOR CLASSIFICATION
#
# Classes:
# 0 = glioma
# 1 = meningioma
# 2 = pituitary
#
# Includes:
# - Training
# - Validation
# - Best model checkpoint
# - H5 model saving
# - Accuracy
# - Loss
# - Confusion Matrix
# - Classification Report
# - Sensitivity
# - Specificity
# - Precision-Recall AUC
# - ROC AUC
# - Inference Time
# - Model Size
# - Parameter Count
# - Training Curves
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report
)


# ============================================================
# 2. TENSORFLOW INFORMATION
# ============================================================

print("TensorFlow version:", tf.__version__)


# ============================================================
# 3. CHECK GPU
# ============================================================

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("\nGPU detected:")
    for gpu in gpus:
        print(gpu)
else:
    print("\nNo GPU detected. Training will use CPU.")


# ============================================================
# 4. BASE DATA DIRECTORY
# ============================================================

base_dir = DATA_DIR


# ============================================================
# 5. DATASET DIRECTORIES
# ============================================================

train_dir = os.path.join(
    base_dir,
    'train'
)

val_dir = os.path.join(
    base_dir,
    'val'
)

test_dir = os.path.join(
    base_dir,
    'test'
)


# ============================================================
# 6. CHECK DIRECTORIES
# ============================================================

print("\n========================================")
print("CHECKING DATASET")
print("========================================")

if not os.path.exists(train_dir):
    raise FileNotFoundError(
        f"Training directory not found:\n{train_dir}"
    )

if not os.path.exists(val_dir):
    raise FileNotFoundError(
        f"Validation directory not found:\n{val_dir}"
    )

if not os.path.exists(test_dir):
    raise FileNotFoundError(
        f"Test directory not found:\n{test_dir}"
    )

print("Training directory: OK")
print("Validation directory: OK")
print("Test directory: OK")


# ============================================================
# 7. IMAGE PARAMETERS
# ============================================================

img_height = 224
img_width = 224
batch_size = 32
epochs = 20


# ============================================================
# 8. IMAGE DATA GENERATOR
# ============================================================

datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


# ============================================================
# 9. LOAD TRAINING DATA
# ============================================================

print("\nLoading training data...")

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(
        img_height,
        img_width
    ),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)


# ============================================================
# 10. LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation data...")

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(
        img_height,
        img_width
    ),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)


# ============================================================
# 11. LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(
        img_height,
        img_width
    ),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)


# ============================================================
# 12. CLASS INFORMATION
# ============================================================

num_classes = len(
    train_data.class_indices
)

class_indices = train_data.class_indices

class_names = list(
    class_indices.keys()
)


print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print(
    "Training images:",
    train_data.samples
)

print(
    "Validation images:",
    val_data.samples
)

print(
    "Test images:",
    test_data.samples
)

print(
    "Number of classes:",
    num_classes
)

print("\nClass mapping:")

for class_name, class_index in class_indices.items():

    print(
        f"{class_index}: {class_name}"
    )


# ============================================================
# 13. VERIFY CLASS MAPPING
# ============================================================

if train_data.class_indices != val_data.class_indices:

    raise ValueError(
        "Training and validation class mappings are different."
    )


if train_data.class_indices != test_data.class_indices:

    raise ValueError(
        "Training and test class mappings are different."
    )


# ============================================================
# 14. LOAD VGG16
# ============================================================

print("\n========================================")
print("LOADING VGG16")
print("========================================")

base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(
        img_height,
        img_width,
        3
    )
)


# ============================================================
# 15. FREEZE VGG16
# ============================================================

base_model.trainable = False


# ============================================================
# 16. BUILD MODEL
# ============================================================

x = base_model.output

x = Flatten()(x)

x = Dense(
    512,
    activation='relu'
)(x)

x = Dropout(
    0.5
)(x)

output = Dense(
    num_classes,
    activation='softmax'
)(x)


# ============================================================
# 17. CREATE MODEL
# ============================================================

model = Model(
    inputs=base_model.input,
    outputs=output
)


# ============================================================
# 18. MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 19. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=Adam(
        learning_rate=0.0001
    ),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# ============================================================
# 20. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# 21. CHECKPOINT PATH
#
# IMPORTANT:
# Keras 3 requires .keras for ModelCheckpoint
# ============================================================

checkpoint_path = os.path.join(OUT_DIR, 'VGG16_TB_best.keras')


# ============================================================
# 22. FINAL H5 MODEL PATH
# ============================================================

h5_model_path = os.path.join(OUT_DIR, 'VGG16_TB_Classification.h5')


# ============================================================
# 23. MODEL CHECKPOINT
# ============================================================

checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)


# ============================================================
# 24. START TRAINING
# ============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================")

training_start_time = time.perf_counter()


history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data,
    callbacks=[
        early_stopping,
        checkpoint
    ],
    verbose=2
)


training_end_time = time.perf_counter()


# ============================================================
# 25. TRAINING TIME
# ============================================================

total_training_time = (
    training_end_time -
    training_start_time
)


print("\n========================================")
print("TRAINING COMPLETED")
print("========================================")

print(
    f"Total training time: "
    f"{total_training_time:.2f} seconds"
)

print(
    f"Total training time: "
    f"{total_training_time / 60:.2f} minutes"
)


# ============================================================
# 26. LOAD BEST MODEL
# ============================================================

print("\n========================================")
print("LOADING BEST MODEL")
print("========================================")

best_model = tf.keras.models.load_model(
    checkpoint_path
)

print(
    "Best model loaded successfully."
)


# ============================================================
# 27. SAVE BEST MODEL AS H5
# ============================================================

print("\n========================================")
print("SAVING H5 MODEL")
print("========================================")

best_model.save(
    h5_model_path
)

print(
    "H5 model saved successfully:"
)

print(
    h5_model_path
)


# ============================================================
# 28. SAVE CLASS MAPPING
# ============================================================

class_names_path = os.path.join(
    base_dir,
    'class_names.json'
)


with open(
    class_names_path,
    'w'
) as f:

    json.dump(
        class_indices,
        f,
        indent=4
    )


print(
    "\nClass mapping saved:"
)

print(
    class_names_path
)


# ============================================================
# 29. MODEL SIZE
# ============================================================

model_size_bytes = os.path.getsize(
    h5_model_path
)

model_size_mb = (
    model_size_bytes /
    (1024 ** 2)
)

model_size_gb = (
    model_size_bytes /
    (1024 ** 3)
)


print("\n========================================")
print("MODEL SIZE")
print("========================================")

print(
    f"Model size: "
    f"{model_size_mb:.2f} MB"
)

print(
    f"Model size: "
    f"{model_size_gb:.4f} GB"
)

print(
    f"Total parameters: "
    f"{best_model.count_params():,}"
)


# ============================================================
# 30. EVALUATE TEST SET
# ============================================================

print("\n========================================")
print("TEST EVALUATION")
print("========================================")

test_data.reset()


test_loss, test_accuracy = (
    best_model.evaluate(
        test_data,
        verbose=1
    )
)


print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)


# ============================================================
# 31. GENERATE TEST PREDICTIONS
# ============================================================

print("\n========================================")
print("GENERATING PREDICTIONS")
print("========================================")


test_data.reset()


inference_start_time = time.perf_counter()


test_preds = best_model.predict(
    test_data,
    verbose=1
)


inference_end_time = time.perf_counter()


# True labels
test_labels = test_data.classes


# Predicted labels
test_preds_labels = np.argmax(
    test_preds,
    axis=1
)


print(
    "\nPrediction shape:",
    test_preds.shape
)

print(
    "True labels shape:",
    test_labels.shape
)


# ============================================================
# 32. INFERENCE TIME
# ============================================================

total_inference_time = (
    inference_end_time -
    inference_start_time
)


num_test_images = test_data.samples


average_inference_time = (
    total_inference_time /
    num_test_images
)


images_per_second = (
    num_test_images /
    total_inference_time
)


print("\n========================================")
print("INFERENCE TIME")
print("========================================")

print(
    f"Test images: "
    f"{num_test_images}"
)

print(
    f"Total inference time: "
    f"{total_inference_time:.4f} seconds"
)

print(
    f"Average inference time: "
    f"{average_inference_time * 1000:.2f} ms/image"
)

print(
    f"Inference speed: "
    f"{images_per_second:.2f} images/second"
)


# ============================================================
# 33. CONFUSION MATRIX
# ============================================================

conf_matrix = confusion_matrix(
    test_labels,
    test_preds_labels
)


print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(
    conf_matrix
)


# ============================================================
# 34. CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")


print(
    classification_report(
        test_labels,
        test_preds_labels,
        target_names=class_names,
        digits=4
    )
)


# ============================================================
# 35. CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(7, 6)
)


sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)


plt.title(
    'Confusion Matrix'
)

plt.xlabel(
    'Predicted Label'
)

plt.ylabel(
    'True Label'
)

plt.tight_layout()

plt.show()


# ============================================================
# 36. SENSITIVITY AND SPECIFICITY
# ============================================================

print("\n========================================")
print("SENSITIVITY AND SPECIFICITY")
print("========================================")


sensitivity_list = []

specificity_list = []


for class_index in range(
    num_classes
):

    # ----------------------------------------
    # TRUE BINARY LABELS
    # ----------------------------------------

    y_true_binary = (
        test_labels == class_index
    ).astype(int)


    # ----------------------------------------
    # PREDICTED BINARY LABELS
    # ----------------------------------------

    y_pred_binary = (
        test_preds_labels == class_index
    ).astype(int)


    # ----------------------------------------
    # ONE-VS-REST CONFUSION MATRIX
    # ----------------------------------------

    cm = confusion_matrix(
        y_true_binary,
        y_pred_binary,
        labels=[0, 1]
    )


    tn, fp, fn, tp = cm.ravel()


    # ----------------------------------------
    # SENSITIVITY
    # ----------------------------------------

    if (
        tp + fn
    ) > 0:

        sensitivity = (
            tp /
            (tp + fn)
        )

    else:

        sensitivity = 0.0


    # ----------------------------------------
    # SPECIFICITY
    # ----------------------------------------

    if (
        tn + fp
    ) > 0:

        specificity = (
            tn /
            (tn + fp)
        )

    else:

        specificity = 0.0


    sensitivity_list.append(
        sensitivity
    )

    specificity_list.append(
        specificity
    )


    print(
        f"\nClass: "
        f"{class_names[class_index]}"
    )

    print(
        f"TP: {tp}"
    )

    print(
        f"TN: {tn}"
    )

    print(
        f"FP: {fp}"
    )

    print(
        f"FN: {fn}"
    )

    print(
        f"Sensitivity: "
        f"{sensitivity * 100:.2f}%"
    )

    print(
        f"Specificity: "
        f"{specificity * 100:.2f}%"
    )


# ============================================================
# 37. MACRO AVERAGE
# ============================================================

mean_sensitivity = np.mean(
    sensitivity_list
)

mean_specificity = np.mean(
    specificity_list
)


print("\n========================================")
print("MACRO-AVERAGE")
print("========================================")

print(
    f"Mean Sensitivity: "
    f"{mean_sensitivity * 100:.2f}%"
)

print(
    f"Mean Specificity: "
    f"{mean_specificity * 100:.2f}%"
)


# ============================================================
# 38. PRECISION-RECALL CURVES
# ============================================================

print("\n========================================")
print("PRECISION-RECALL AUC")
print("========================================")


pr_auc_list = []


plt.figure(
    figsize=(8, 6)
)


for class_index in range(
    num_classes
):

    # Binary true labels
    y_true_binary = (
        test_labels == class_index
    ).astype(int)


    # IMPORTANT:
    # Use probability of ONLY this class
    y_score = test_preds[
        :,
        class_index
    ]


    precision, recall, _ = (
        precision_recall_curve(
            y_true_binary,
            y_score
        )
    )


    pr_auc = auc(
        recall,
        precision
    )


    pr_auc_list.append(
        pr_auc
    )


    print(
        f"{class_names[class_index]} "
        f"PR AUC: "
        f"{pr_auc:.4f}"
    )


    plt.plot(
        recall,
        precision,
        label=(
            f"{class_names[class_index]} "
            f"(AUC = {pr_auc:.4f})"
        )
    )


plt.xlabel(
    'Recall'
)

plt.ylabel(
    'Precision'
)

plt.title(
    'Precision-Recall Curves'
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.show()


# ============================================================
# 39. ROC CURVES
# ============================================================

print("\n========================================")
print("ROC AUC")
print("========================================")


roc_auc_list = []


plt.figure(
    figsize=(8, 6)
)


for class_index in range(
    num_classes
):

    # Binary true labels
    y_true_binary = (
        test_labels == class_index
    ).astype(int)


    # Probability for current class
    y_score = test_preds[
        :,
        class_index
    ]


    fpr, tpr, _ = roc_curve(
        y_true_binary,
        y_score
    )


    roc_auc = auc(
        fpr,
        tpr
    )


    roc_auc_list.append(
        roc_auc
    )


    print(
        f"{class_names[class_index]} "
        f"ROC AUC: "
        f"{roc_auc:.4f}"
    )


    plt.plot(
        fpr,
        tpr,
        label=(
            f"{class_names[class_index]} "
            f"(AUC = {roc_auc:.4f})"
        )
    )


# Random classifier line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--'
)


plt.xlabel(
    'False Positive Rate'
)

plt.ylabel(
    'True Positive Rate'
)

plt.title(
    'ROC Curves'
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.show()


# ============================================================
# 40. TRAINING ACCURACY CURVE
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)


plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)


plt.title(
    'Training and Validation Accuracy'
)

plt.xlabel(
    'Epoch'
)

plt.ylabel(
    'Accuracy'
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.show()


# ============================================================
# 41. TRAINING LOSS CURVE
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.plot(
    history.history['loss'],
    label='Training Loss'
)


plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)


plt.title(
    'Training and Validation Loss'
)

plt.xlabel(
    'Epoch'
)

plt.ylabel(
    'Loss'
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.show()


# ============================================================
# 42. MEAN AUC VALUES
# ============================================================

mean_pr_auc = np.mean(
    pr_auc_list
)

mean_roc_auc = np.mean(
    roc_auc_list
)


# ============================================================
# 43. FINAL RESULTS
# ============================================================

print("\n")
print("============================================================")
print("                  FINAL MODEL RESULTS")
print("============================================================")


print(
    f"\nTest Accuracy:"
    f" {test_accuracy * 100:.2f}%"
)


print(
    f"Test Loss:"
    f" {test_loss:.4f}"
)


print(
    f"Mean Sensitivity:"
    f" {mean_sensitivity * 100:.2f}%"
)


print(
    f"Mean Specificity:"
    f" {mean_specificity * 100:.2f}%"
)


print(
    f"Mean PR AUC:"
    f" {mean_pr_auc:.4f}"
)


print(
    f"Mean ROC AUC:"
    f" {mean_roc_auc:.4f}"
)


print(
    f"Average Inference Time:"
    f" {average_inference_time * 1000:.2f} ms/image"
)


print(
    f"Inference Speed:"
    f" {images_per_second:.2f} images/second"
)


print(
    f"Model Size:"
    f" {model_size_mb:.2f} MB"
)


print(
    f"Total Parameters:"
    f" {best_model.count_params():,}"
)


print(
    "\nBest H5 Model:"
)

print(
    h5_model_path
)


print(
    "\nClass Mapping:"
)

print(
    class_indices
)


print(
    "\n============================================================"
)

print(
    "TRAINING AND EVALUATION COMPLETED SUCCESSFULLY"
)

print(
    "============================================================"
)
