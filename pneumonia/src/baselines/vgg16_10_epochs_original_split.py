"""VGG16 (frozen), Flatten-Dense head, 10 epochs, on the ORIGINAL Kermany partition (5,216 / 16 / 624).

Extracted from the executed notebook `notebooks/01_benchmarks_resplit_VGG16_DenseNet121_MobileNetV2_CNN_InceptionV3_and_GradCAM.ipynb` (cell 0); the model definition and the
training/evaluation code are unchanged. Only hard-coded Windows paths were replaced by the environment
variables below, and figures are written to disk instead of being displayed.

    PN_RAW_DIR     original Kaggle/Mendeley release with train/, val/, test/ (default: data/raw/chest_xray)
    PN_DATA_DIR    re-split data with train/, val/, test/, each containing NORMAL/ and PNEUMONIA/
                   (default: data/chest_xray_new)
    PN_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/vgg16_10_original_split)

Run from the `pneumonia/` folder:  python src/baselines/vgg16_10_epochs_original_split.py
"""
import os
RAW_DIR = os.path.abspath(os.environ.get("PN_RAW_DIR", "data/raw/chest_xray"))
DATA_DIR = os.path.abspath(os.environ.get("PN_DATA_DIR", "data/chest_xray_new"))
OUT_DIR = os.path.abspath(os.environ.get("PN_OUTPUT_DIR", "outputs/vgg16_10_original_split"))
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
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

# =========================
# Dataset Path
# =========================

train_path = os.path.join(RAW_DIR, "train")
val_path   = os.path.join(RAW_DIR, "val")
test_path  = os.path.join(RAW_DIR, "test")

# =========================
# Image Parameters
# =========================

IMG_SIZE = 224
BATCH_SIZE = 32

# =========================
# Data Augmentation
# =========================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# =========================
# Load Dataset
# =========================

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = test_datagen.flow_from_directory(
    val_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# =========================
# Load VGG16 Model
# =========================

base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
for layer in base_model.layers:
    layer.trainable = False

# =========================
# Build Model
# =========================

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# =========================
# Compile Model
# =========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# Train Model
# =========================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# =========================
# Evaluate Model
# =========================

test_loss, test_accuracy = model.evaluate(test_data)

print("\nTest Accuracy:", test_accuracy)

# =========================
# Predictions
# =========================

y_prob = model.predict(test_data)
y_pred = (y_prob > 0.5).astype(int)

y_true = test_data.classes

# =========================
# Classification Report
# =========================

print("\nClassification Report:\n")

target_names = ['NORMAL', 'PNEUMONIA']

print(classification_report(
    y_true,
    y_pred,
    target_names=target_names
))

# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()

classes = ['NORMAL', 'PNEUMONIA']
tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, cm[i, j],
                 ha='center',
                 va='center',
                 color='red')

plt.show()
# 16. SENSITIVITY & SPECIFICITY
# =========================
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("Sensitivity:", round(sensitivity, 4))
print("Specificity:", round(specificity, 4))

# =========================
# 17. INFERENCE TIME
# =========================
start = time.time()
_ = model.predict(test_gen)
end = time.time()

total_time = end - start
avg_time = total_time / len(test_df)

print("Total Inference Time:", round(total_time, 4), "sec")
print("Time per Image:", round(avg_time, 6), "sec")


# =========================
# ROC Curve
# =========================

fpr, tpr, thresholds = roc_curve(y_true, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label='AUC = %0.4f' % roc_auc)
plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

plt.show()

print("\nROC-AUC Score:", roc_auc)

# =========================
# Precision-Recall Curve
# =========================

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})', color='darkorange', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (Class {target_class})')
plt.legend(loc='lower left')
plt.grid(True)
plt.tight_layout()
plt.show()
# Step 11: Training and Validation Accuracy/Loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
