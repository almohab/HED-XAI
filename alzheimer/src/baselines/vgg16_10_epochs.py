"""VGG16 benchmark, 10 epochs (Flatten head, block5 fine-tuned).

Extracted from the executed notebook `01_baselines_CNN_VGG16_MobileNetV2_InceptionV3_DenseNet121.ipynb` (cell 1); the model definition and the
training/evaluation code are unchanged. Only hard-coded Windows paths were replaced by the
environment variables below, and figures are written to disk instead of being displayed.

    AD_DATA_DIR    folder containing "Non Demented/" and "Very mild Dementia/" (default: data/balanced_data)
    AD_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/vgg16_10)

Run from the `alzheimer/` folder:  python src/baselines/vgg16_10_epochs.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("AD_DATA_DIR", "data/balanced_data"))
OUT_DIR = os.path.abspath(os.environ.get("AD_OUTPUT_DIR", "outputs/vgg16_10"))
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

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# =========================
# 1. PATH (YOUR DATA)
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

# Convert to binary
df['label'] = df['label'].map({
    'Non Demented': 0,
    'Very mild Dementia': 1
})

# =========================
# 3. SPLIT
# =========================
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

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
# 6. MODEL (VGG16)
# =========================
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze most layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

model = Sequential([
    base_model,
    Flatten(),
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

# =========================
# 8. TRAIN (ONLY ONCE)
# =========================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# =========================
# 9. PLOT
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
# 10. TEST
# =========================
test_loss, test_acc = model.evaluate(test_gen)
print("Test Accuracy:", test_acc)

# =========================
# 11. RESULTS
# =========================
y_true = test_df['label'].values
y_probs = model.predict(test_gen)
y_pred = (y_probs > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

print(classification_report(y_true, y_pred))

# =========================
# 12. ROC
# =========================
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title("ROC Curve")
plt.show()
