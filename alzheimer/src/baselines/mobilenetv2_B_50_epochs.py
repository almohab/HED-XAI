"""MobileNetV2 (B): last 30 layers trainable, 50 epochs, EarlyStopping.

Extracted from the executed notebook `01_baselines_CNN_VGG16_MobileNetV2_InceptionV3_DenseNet121.ipynb` (cell 7); the model definition and the
training/evaluation code are unchanged. Only hard-coded Windows paths were replaced by the
environment variables below, and figures are written to disk instead of being displayed.

    AD_DATA_DIR    folder containing "Non Demented/" and "Very mild Dementia/" (default: data/balanced_data)
    AD_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/mobilenetv2_B)

Run from the `alzheimer/` folder:  python src/baselines/mobilenetv2_B_50_epochs.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("AD_DATA_DIR", "data/balanced_data"))
OUT_DIR = os.path.abspath(os.environ.get("AD_OUTPUT_DIR", "outputs/mobilenetv2_B"))
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
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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
# 1. PATH
# =========================
base_path = DATA_DIR

categories = ['Non Demented', 'Very mild Dementia']

image_paths = []
labels = []

# =========================
# 2. LOAD DATA
# =========================
for category in categories:
    folder = os.path.join(base_path, category)
    for file in os.listdir(folder):
        image_paths.append(os.path.join(folder, file))
        labels.append(category)

df = pd.DataFrame({'image_path': image_paths, 'label': labels})

df['label'] = df['label'].map({
    'Non Demented': 0,
    'Very mild Dementia': 1
})

# =========================
# 3. SPLIT
# =========================
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df['label'], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)

print(len(train_df), len(val_df), len(test_df))

# =========================
# 4. AUGMENTATION
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# =========================
# 5. GENERATORS
# =========================
train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col='label',
    target_size=(224,224),
    batch_size=32,
    class_mode='raw'
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col='image_path',
    y_col='label',
    target_size=(224,224),
    batch_size=32,
    class_mode='raw'
)

test_gen = val_datagen.flow_from_dataframe(
    test_df,
    x_col='image_path',
    y_col='label',
    target_size=(224,224),
    batch_size=32,
    class_mode='raw',
    shuffle=False
)

# =========================
# 6. MOBILENETV2 MODEL (ONE TRAINING)
# =========================
base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

# Freeze most layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Unfreeze last layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Custom head
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)

output = Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

# =========================
# 7. COMPILE
# =========================
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# 8. TRAIN (50 EPOCHS ONLY)
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[early_stop]
)

# =========================
# 9. SAVE MODEL
# =========================
model.save("mobilenetv2_model.h5")

# =========================
# 10. PLOTS
# =========================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend()
plt.title("Loss")

plt.show()

# =========================
# 11. TEST
# =========================
test_loss, test_acc = model.evaluate(test_gen)
print("Test Accuracy:", test_acc)

# =========================
# 12. PREDICTIONS
# =========================
y_true = test_df['label'].values
y_probs = model.predict(test_gen)
y_pred = (y_probs > 0.5).astype(int)

# =========================
# 13. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

print(classification_report(
    y_true, y_pred,
    target_names=['Non Demented', 'Very mild Dementia']
))

# =========================
# 14. ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.title("ROC Curve")
plt.show()

# =========================
# 15. PR CURVE
# =========================
precision, recall, _ = precision_recall_curve(y_true, y_probs)
ap_score = average_precision_score(y_true, y_probs)

plt.plot(recall, precision, label=f"AP={ap_score:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title("PR Curve")
plt.show()

# =========================
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
# 18. MODEL SIZE
# =========================
size_mb = os.path.getsize("mobilenetv2_model.h5") / (1024 * 1024)
print("Model Size:", round(size_mb, 2), "MB")
