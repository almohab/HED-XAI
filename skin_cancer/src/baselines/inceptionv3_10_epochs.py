"""InceptionV3 (last 30 layers fine-tuned), 299x299, 10 epochs.

Extracted from the executed notebook `notebooks/01_benchmarks_CNN_VGG16_DenseNet121_MobileNetV2_Xception_InceptionV3.ipynb` (cell 16); the model definition and the
training/evaluation code are unchanged. Only hard-coded paths were replaced by the environment variables below
(and, in the hybrid script, class folders are matched case-insensitively), and figures are written to disk.

    SC_DATA_DIR    folder with train/, val/ and test/, each containing benign/ and malignant/
                   (default: data/dataset_skin)
    SC_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/inceptionv3_10)

Run from the `skin_cancer/` folder:  python src/baselines/inceptionv3_10_epochs.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("SC_DATA_DIR", "data/dataset_skin"))
OUT_DIR = os.path.abspath(os.environ.get("SC_OUTPUT_DIR", "outputs/inceptionv3_10"))
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
# INCEPTIONV3 FOR SKIN CANCER CLASSIFICATION
# 10 EPOCHS - H5 MODEL
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# Suppress non-critical warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("INCEPTIONV3 SKIN CANCER CLASSIFICATION")
print("=" * 70)

print("\nTensorFlow version:", tf.__version__)


# ============================================================
# 2. DATASET PATH
# ============================================================

base_path = DATA_DIR

train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "val")
test_path = os.path.join(base_path, "test")

categories = ['BENIGN', 'MALIGNANT']

IMG_SIZE = (299, 299)
BATCH_SIZE = 32


# ============================================================
# 3. CHECK DATASET STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("CHECKING DATASET")
print("=" * 70)

print("\nBase path:")
print(os.path.abspath(base_path))

print("\nTrain path:")
print(os.path.abspath(train_path))

print("\nValidation path:")
print(os.path.abspath(val_path))

print("\nTest path:")
print(os.path.abspath(test_path))


if not os.path.exists(base_path):
    raise FileNotFoundError(
        f"\nDataset folder not found:\n{os.path.abspath(base_path)}"
        "\n\nPlease make sure the DATA_DIR folder is in the "
        "same directory as your Jupyter Notebook."
    )

if not os.path.exists(train_path):
    raise FileNotFoundError(
        f"\nTraining folder not found:\n{os.path.abspath(train_path)}"
    )

if not os.path.exists(val_path):
    raise FileNotFoundError(
        f"\nValidation folder not found:\n{os.path.abspath(val_path)}"
    )

if not os.path.exists(test_path):
    raise FileNotFoundError(
        f"\nTesting folder not found:\n{os.path.abspath(test_path)}"
    )


for category in categories:

    train_folder = os.path.join(train_path, category)
    val_folder = os.path.join(val_path, category)
    test_folder = os.path.join(test_path, category)

    if not os.path.exists(train_folder):
        raise FileNotFoundError(
            f"Training class folder not found:\n{train_folder}"
        )

    if not os.path.exists(val_folder):
        raise FileNotFoundError(
            f"Validation class folder not found:\n{val_folder}"
        )

    if not os.path.exists(test_folder):
        raise FileNotFoundError(
            f"Testing class folder not found:\n{test_folder}"
        )

print("\n✓ Dataset structure verified successfully!")


# ============================================================
# 4. COUNT IMAGES
# ============================================================

def count_images(folder):

    extensions = (
        '.jpg',
        '.jpeg',
        '.png',
        '.bmp',
        '.tif',
        '.tiff'
    )

    count = 0

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(extensions):
                count += 1

    return count


print("\n" + "=" * 70)
print("DATASET STATISTICS")
print("=" * 70)

dataset_counts = {}

for split_name, split_path in [
    ("Training", train_path),
    ("Validation", val_path),
    ("Testing", test_path)
]:

    dataset_counts[split_name] = {}

    print(f"\n{split_name}:")

    total = 0

    for category in categories:

        folder = os.path.join(split_path, category)

        count = count_images(folder)

        dataset_counts[split_name][category] = count

        print(f"   {category}: {count}")

        total += count

    dataset_counts[split_name]["Total"] = total

    print(f"   Total: {total}")


# ============================================================
# 5. DATA AUGMENTATION
# ============================================================

print("\n" + "=" * 70)
print("CREATING IMAGE GENERATORS")
print("=" * 70)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


# ============================================================
# 6. TRAINING GENERATOR
# ============================================================

train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=categories,
    shuffle=True
)


# ============================================================
# 7. VALIDATION GENERATOR
# ============================================================

val_gen = val_datagen.flow_from_directory(
    val_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=categories,
    shuffle=False
)


# ============================================================
# 8. TEST GENERATOR
# ============================================================

test_gen = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=categories,
    shuffle=False
)


print("\nClass indices:")
print(train_gen.class_indices)

print("\nTraining images:", train_gen.samples)
print("Validation images:", val_gen.samples)
print("Testing images:", test_gen.samples)


# ============================================================
# 9. BUILD INCEPTIONV3 BASE MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING INCEPTIONV3 MODEL")
print("=" * 70)

base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(299, 299, 3)
)


# Freeze all layers
for layer in base_model.layers:
    layer.trainable = False


# Unfreeze last 30 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True


trainable_layers = sum(
    1 for layer in base_model.layers
    if layer.trainable
)

total_layers = len(base_model.layers)

print("\nTotal base model layers:", total_layers)
print("Trainable base model layers:", trainable_layers)
print("Frozen base model layers:", total_layers - trainable_layers)


# ============================================================
# 10. BUILD CLASSIFICATION MODEL
# ============================================================

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(
        256,
        activation='relu'
    ),

    Dropout(0.5),

    Dense(
        128,
        activation='relu'
    ),

    Dropout(0.3),

    Dense(
        1,
        activation='sigmoid'
    )
])


# ============================================================
# 11. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=Adam(
        learning_rate=1e-4
    ),

    loss='binary_crossentropy',

    metrics=[
        'accuracy'
    ]
)


print("\nModel compiled successfully!")


# ============================================================
# 12. MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 13. TRAIN MODEL - 10 EPOCHS
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

start_training_time = time.time()

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    verbose=1
)

end_training_time = time.time()

training_time = end_training_time - start_training_time

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"\nTotal training time: "
    f"{training_time / 60:.2f} minutes"
)


# ============================================================
# 14. SAVE MODEL AS H5
# ============================================================

model_filename = "inceptionv3_skin_cancer_model.h5"

model.save(model_filename)

print("\nModel saved successfully:")
print(os.path.abspath(model_filename))


# ============================================================
# 15. TRAINING CURVES
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CURVES")
print("=" * 70)

plt.figure(figsize=(10, 6))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title(
    'InceptionV3 Skin Cancer Classification - Accuracy'
)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()
plt.grid(True)

plt.show()


plt.figure(figsize=(10, 6))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title(
    'InceptionV3 Skin Cancer Classification - Loss'
)

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 16. FINAL TRAINING RESULTS
# ============================================================

final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]

final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print("\nFinal Training Accuracy:",
      f"{final_train_accuracy * 100:.2f}%")

print("Final Validation Accuracy:",
      f"{final_val_accuracy * 100:.2f}%")

print("Final Training Loss:",
      f"{final_train_loss:.4f}")

print("Final Validation Loss:",
      f"{final_val_loss:.4f}")


# ============================================================
# 17. TEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("TESTING MODEL")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    test_gen,
    verbose=1
)

print("\nTest Loss:",
      f"{test_loss:.4f}")

print(
    "Test Accuracy:",
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 18. GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

# Make sure generator starts from the beginning
test_gen.reset()

prediction_start_time = time.time()

y_probs = model.predict(
    test_gen,
    verbose=1
)

prediction_end_time = time.time()

y_probs = y_probs.ravel()

# Binary predictions
y_pred = (
    y_probs >= 0.5
).astype(int)

# True labels
y_true = test_gen.classes


# ============================================================
# 19. INFERENCE TIME
# ============================================================

total_inference_time = (
    prediction_end_time -
    prediction_start_time
)

num_test_images = len(y_true)

average_inference_time = (
    total_inference_time /
    num_test_images
)

print(
    "\nTotal inference time:",
    f"{total_inference_time:.4f} seconds"
)

print(
    "Average inference time per image:",
    f"{average_inference_time:.6f} seconds"
)

print(
    "Images processed:",
    num_test_images
)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=categories,
    yticklabels=categories
)

plt.title(
    'InceptionV3 - Confusion Matrix'
)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.show()


# ============================================================
# 21. EXTRACT TP, TN, FP, FN
# ============================================================

if cm.shape == (2, 2):

    tn, fp, fn, tp = cm.ravel()

else:

    tn = fp = fn = tp = 0


print("\nTrue Negative (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
print("True Positive (TP):", tp)


# ============================================================
# 22. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_true,
    y_pred,
    target_names=categories,
    digits=4
)

print(report)


# ============================================================
# 23. ROC CURVE AND ROC-AUC
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_probs
)

roc_auc = auc(
    fpr,
    tpr
)


print("\n" + "=" * 70)
print("ROC-AUC")
print("=" * 70)

print(
    "ROC-AUC:",
    f"{roc_auc:.4f}"
)

print(
    "ROC-AUC (%):",
    f"{roc_auc * 100:.2f}%"
)


plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f'InceptionV3 (AUC = {roc_auc:.4f})'
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--'
)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title(
    'InceptionV3 - ROC Curve'
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 24. PRECISION-RECALL CURVE
# ============================================================

precision, recall, pr_thresholds = precision_recall_curve(
    y_true,
    y_probs
)

average_precision = average_precision_score(
    y_true,
    y_probs
)


print("\n" + "=" * 70)
print("PRECISION-RECALL ANALYSIS")
print("=" * 70)

print(
    "Average Precision (AP):",
    f"{average_precision:.4f}"
)

print(
    "Average Precision (%):",
    f"{average_precision * 100:.2f}%"
)


plt.figure(figsize=(8, 6))

plt.plot(
    recall,
    precision,
    label=f'InceptionV3 (AP = {average_precision:.4f})'
)

plt.xlabel('Recall')
plt.ylabel('Precision')

plt.title(
    'InceptionV3 - Precision-Recall Curve'
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 25. SENSITIVITY AND SPECIFICITY
# ============================================================

if (tp + fn) > 0:

    sensitivity = tp / (tp + fn)

else:

    sensitivity = 0


if (tn + fp) > 0:

    specificity = tn / (tn + fp)

else:

    specificity = 0


print("\n" + "=" * 70)
print("SENSITIVITY AND SPECIFICITY")
print("=" * 70)

print(
    "Sensitivity:",
    f"{sensitivity:.4f}"
)

print(
    "Sensitivity (%):",
    f"{sensitivity * 100:.2f}%"
)

print(
    "Specificity:",
    f"{specificity:.4f}"
)

print(
    "Specificity (%):",
    f"{specificity * 100:.2f}%"
)


# ============================================================
# 26. MODEL SIZE
# ============================================================

model_size_bytes = os.path.getsize(
    model_filename
)

model_size_mb = (
    model_size_bytes /
    (1024 ** 2)
)


print("\n" + "=" * 70)
print("MODEL SIZE")
print("=" * 70)

print(
    "Model size:",
    f"{model_size_mb:.2f} MB"
)


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL INCEPTIONV3 RESULTS")
print("=" * 70)

print(
    f"Training Accuracy       : "
    f"{final_train_accuracy * 100:.2f}%"
)

print(
    f"Validation Accuracy     : "
    f"{final_val_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy           : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"ROC-AUC                 : "
    f"{roc_auc * 100:.2f}%"
)

print(
    f"Average Precision       : "
    f"{average_precision * 100:.2f}%"
)

print(
    f"Sensitivity             : "
    f"{sensitivity * 100:.2f}%"
)

print(
    f"Specificity             : "
    f"{specificity * 100:.2f}%"
)

print(
    f"Training Time           : "
    f"{training_time / 60:.2f} minutes"
)

print(
    f"Avg. Inference/Image    : "
    f"{average_inference_time:.6f} seconds"
)

print(
    f"Model Size              : "
    f"{model_size_mb:.2f} MB"
)

print(
    f"Model File              : "
    f"{model_filename}"
)

print("=" * 70)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 70)
