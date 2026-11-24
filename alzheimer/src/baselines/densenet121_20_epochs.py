"""DenseNet121 benchmark, 20 epochs (run on a separate 8,000-image configuration).

Extracted from the executed notebook `01_baselines_CNN_VGG16_MobileNetV2_InceptionV3_DenseNet121.ipynb` (cell 28); the model definition and the
training/evaluation code are unchanged. Only hard-coded Windows paths were replaced by the
environment variables below, and figures are written to disk instead of being displayed.

    AD_DATA_DIR    folder containing "Non Demented/" and "Very mild Dementia/" (default: data/balanced_data)
    AD_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/densenet121_20)

Run from the `alzheimer/` folder:  python src/baselines/densenet121_20_epochs.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("AD_DATA_DIR", "data/balanced_data"))
OUT_DIR = os.path.abspath(os.environ.get("AD_OUTPUT_DIR", "outputs/densenet121_20"))
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

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# =========================
# ============================================================
# DATASET PATH
# ============================================================

base_path = DATA_DIR

categories = [
    "Non Demented",
    "Very mild Dementia"
]

label_map = {
    "Non Demented": 0,
    "Very mild Dementia": 1
}
# =========================
# 2. LOAD DATA
# =========================
for category in categories:
    folder = os.path.join(base_path, category)

    for file in os.listdir(folder):
        image_paths.append(os.path.join(folder, file))
        labels.append(category)

df = pd.DataFrame({
    'image_path': image_paths,
    'label': labels
})

df['label'] = df['label'].map({
    'Non Demented': 0,
    'Very mild Dementia': 1
})

# =========================
# 3. SPLIT
# =========================
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df['label'],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['label'],
    random_state=42
)

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))

# =========================
# 4. AUGMENTATION
# =========================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# =========================
# 5. GENERATORS
# =========================
train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

test_gen = val_datagen.flow_from_dataframe(
    test_df,
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',
    shuffle=False
)

# =========================
# 6. DENSENET121 MODEL
# =========================
base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze most layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Fine-tune last 4 layers
for layer in base_model.layers[-4:]:
    layer.trainable = True

model = Sequential([
    base_model,

    GlobalAveragePooling2D(),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

# =========================
# 7. COMPILE
# =========================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# =========================
# 8. TRAIN (50 EPOCHS)
# =========================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20
)

# =========================
# 9. SAVE MODEL (.h5)
# =========================
model.save("densenet121_model Alzheimer.h5")

print("Model saved as: densenet121_model Alzheimer.h5")

# =========================
# 10. TRAINING PLOTS
# =========================
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)

plt.plot(
    history.history['accuracy'],
    label='Train'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation'
)

plt.legend()
plt.title("DenseNet121 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# Loss
plt.subplot(1, 2, 2)

plt.plot(
    history.history['loss'],
    label='Train'
)

plt.plot(
    history.history['val_loss'],
    label='Validation'
)

plt.legend()
plt.title("DenseNet121 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()

# =========================
# 11. TEST
# =========================
test_loss, test_acc = model.evaluate(test_gen)

print("\nTest Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_acc, 4))

# =========================
# 12. PREDICTIONS
# =========================
y_true = test_df['label'].values

y_probs = model.predict(test_gen)

# Convert probabilities to 1D array
y_probs = y_probs.ravel()

# Classification threshold
y_pred = (y_probs > 0.5).astype(int)

# =========================
# 13. CONFUSION MATRIX
# =========================
cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Non Demented', 'Very mild Dementia'],
    yticklabels=['Non Demented', 'Very mild Dementia']
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("DenseNet121 Confusion Matrix")

plt.tight_layout()
plt.show()

# =========================
# CLASSIFICATION REPORT
# =========================
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            'Non Demented',
            'Very mild Dementia'
        ],
        digits=4
    )
)

# =========================
# 14. ROC CURVE
# =========================
fpr, tpr, thresholds = roc_curve(
    y_true,
    y_probs
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"DenseNet121 AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    '--'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("DenseNet121 ROC Curve")
plt.legend()

plt.tight_layout()
plt.show()

print("ROC-AUC:", round(roc_auc, 4))

# =========================
# 15. PRECISION-RECALL CURVE
# =========================
precision, recall, thresholds_pr = precision_recall_curve(
    y_true,
    y_probs
)

ap_score = average_precision_score(
    y_true,
    y_probs
)

plt.figure(figsize=(7, 6))

plt.plot(
    recall,
    precision,
    label=f"DenseNet121 AP = {ap_score:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("DenseNet121 Precision-Recall Curve")
plt.legend()

plt.tight_layout()
plt.show()

print("Average Precision:", round(ap_score, 4))

# =========================
# 16. SENSITIVITY & SPECIFICITY
# =========================
tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred
).ravel()

sensitivity = tp / (tp + fn)

specificity = tn / (tn + fp)

print("\nPerformance Metrics:")
print("True Negative (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
print("True Positive (TP):", tp)

print("Sensitivity:", round(sensitivity, 4))
print("Specificity:", round(specificity, 4))

# =========================
# 17. INFERENCE TIME
# =========================

# Reset test generator
test_gen.reset()

start = time.time()

_ = model.predict(
    test_gen,
    verbose=1
)

end = time.time()

total_time = end - start

avg_time = total_time / len(test_df)

print("\nInference Performance:")
print(
    "Total Inference Time:",
    round(total_time, 4),
    "sec"
)

print(
    "Time per Image:",
    round(avg_time, 6),
    "sec"
)

# =========================
# 18. MODEL SIZE
# =========================
model_file = "densenet121_model Alzheimer.h5"

size_mb = os.path.getsize(
    model_file
) / (1024 * 1024)

print("\nModel Information:")
print(
    "Model Size:",
    round(size_mb, 2),
    "MB"
)

# =========================
# 19. TRAINABLE PARAMETERS
# =========================
trainable_params = np.sum([
    np.prod(v.shape)
    for v in model.trainable_weights
])

non_trainable_params = np.sum([
    np.prod(v.shape)
    for v in model.non_trainable_weights
])

print(
    "Trainable Parameters:",
    trainable_params
)

print(
    "Non-trainable Parameters:",
    non_trainable_params
)

print(
    "Total Parameters:",
    trainable_params + non_trainable_params
)
