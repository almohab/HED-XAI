"""Custom CNN-A from scratch, 20 epochs (the supplied notebook copy of this run was interrupted at epoch 15).

Extracted from the executed notebook `notebooks/01_benchmarks_resplit_VGG16_DenseNet121_MobileNetV2_CNN_InceptionV3_and_GradCAM.ipynb` (cell 27); the model definition and the
training/evaluation code are unchanged. Only hard-coded Windows paths were replaced by the environment
variables below, and figures are written to disk instead of being displayed.

    PN_RAW_DIR     original Kaggle/Mendeley release with train/, val/, test/ (default: data/raw/chest_xray)
    PN_DATA_DIR    re-split data with train/, val/, test/, each containing NORMAL/ and PNEUMONIA/
                   (default: data/chest_xray_new)
    PN_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/cnn_a_20)

Run from the `pneumonia/` folder:  python src/baselines/cnn_a_20_epochs.py
"""
import os
RAW_DIR = os.path.abspath(os.environ.get("PN_RAW_DIR", "data/raw/chest_xray"))
DATA_DIR = os.path.abspath(os.environ.get("PN_DATA_DIR", "data/chest_xray_new"))
OUT_DIR = os.path.abspath(os.environ.get("PN_OUTPUT_DIR", "outputs/cnn_a_20"))
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
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

from sklearn.utils.class_weight import compute_class_weight

# =========================================================
# 1. DATASET PATHS
# =========================================================

train_path = os.path.join(DATA_DIR, "train")
val_path   = os.path.join(DATA_DIR, "val")
test_path  = os.path.join(DATA_DIR, "test")

# =========================================================
# 2. IMAGE PARAMETERS
# =========================================================

IMG_SIZE = 224
BATCH_SIZE = 32

# =========================================================
# 3. DATA AUGMENTATION
# =========================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# =========================================================
# 4. LOAD DATASET
# =========================================================

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

# =========================================================
# 5. COMPUTE CLASS WEIGHTS
# =========================================================

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))

print("\nClass Weights:")
print(class_weights)

# =========================================================
# 6. BUILD CNN MODEL
# =========================================================

model = Sequential([

    # First Convolution Block
    Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same',
        input_shape=(224,224,3)
    ),

    BatchNormalization(),

    MaxPooling2D((2,2)),

    # Second Convolution Block
    Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    ),

    BatchNormalization(),

    MaxPooling2D((2,2)),

    # Third Convolution Block
    Conv2D(
        128,
        (3,3),
        activation='relu',
        padding='same'
    ),

    BatchNormalization(),

    MaxPooling2D((2,2)),

    # Fourth Convolution Block
    Conv2D(
        256,
        (3,3),
        activation='relu',
        padding='same'
    ),

    BatchNormalization(),

    MaxPooling2D((2,2)),

    # Flatten Layer
    Flatten(),

    # Fully Connected Layer
    Dense(256, activation='relu'),

    Dropout(0.5),

    # Output Layer
    Dense(1, activation='sigmoid')
])

# =========================================================
# 7. COMPILE MODEL
# =========================================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================================================
# 8. MODEL SUMMARY
# =========================================================

model.summary()

# =========================================================
# 9. TRAIN MODEL
# =========================================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    class_weight=class_weights
)

# =========================================================
# 10. EVALUATE MODEL
# =========================================================

test_loss, test_accuracy = model.evaluate(test_data)

print("\nTest Accuracy:", round(test_accuracy, 4))

# =========================================================
# 11. PREDICTIONS
# =========================================================

y_prob = model.predict(test_data)

y_pred = (y_prob > 0.5).astype(int)

y_true = test_data.classes

# =========================================================
# 12. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:\n")

target_names = ['NORMAL', 'PNEUMONIA']

print(classification_report(
    y_true,
    y_pred,
    target_names=target_names
))

# =========================================================
# 13. CONFUSION MATRIX
# =========================================================

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
        plt.text(
            j,
            i,
            cm[i, j],
            ha='center',
            va='center',
            color='red'
        )

plt.show()

# =========================================================
# 14. SENSITIVITY & SPECIFICITY
# =========================================================

tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)

specificity = tn / (tn + fp)

print("Sensitivity:", round(sensitivity, 4))

print("Specificity:", round(specificity, 4))

# =========================================================
# 15. ROC CURVE
# =========================================================

fpr, tpr, thresholds = roc_curve(y_true, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,6))

plt.plot(fpr, tpr, label='AUC = %0.4f' % roc_auc)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc='lower right')

plt.grid(True)

plt.show()

print("\nROC-AUC Score:", round(roc_auc, 4))

# =========================================================
# 16. PRECISION-RECALL CURVE
# =========================================================

precision, recall, thresholds = precision_recall_curve(
    y_true,
    y_prob
)

pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))

plt.plot(
    recall,
    precision,
    label=f'PR AUC = {pr_auc:.4f}'
)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Precision-Recall Curve')

plt.legend(loc='lower left')

plt.grid(True)

plt.show()

print("PR-AUC Score:", round(pr_auc, 4))

# =========================================================
# 17. INFERENCE TIME
# =========================================================

start = time.time()

_ = model.predict(test_data)

end = time.time()

total_time = end - start

avg_time = total_time / test_data.samples

print("\nTotal Inference Time:", round(total_time, 4), "seconds")

print("Time per Image:", round(avg_time, 6), "seconds")

# =========================================================
# 18. TRAINING & VALIDATION CURVES
# =========================================================

plt.figure(figsize=(12, 4))

# Accuracy Plot

plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'],
         label='Training Accuracy')

plt.plot(history.history['val_accuracy'],
         label='Validation Accuracy')

plt.title('Accuracy over Epochs')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

# Loss Plot

plt.subplot(1, 2, 2)

plt.plot(history.history['loss'],
         label='Training Loss')

plt.plot(history.history['val_loss'],
         label='Validation Loss')

plt.title('Loss over Epochs')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.tight_layout()

plt.show()

# =========================================================
# 19. SAVE MODEL
# =========================================================

model.save("pneumonia_cnn_20Epochs_model.h5")

print("\nModel Saved Successfully!")
