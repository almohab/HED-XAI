"""VGG16 (last 4 layers trainable), GAP head, class weights, 20 epochs with EarlyStopping (stopped at 12).

Extracted from the executed notebook `notebooks/01_benchmarks_CNN_VGG16_DenseNet121_MobileNetV2_Xception_InceptionV3.ipynb` (cell 7); the model definition and the
training/evaluation code are unchanged. Only hard-coded paths were replaced by the environment variables below
(and, in the hybrid script, class folders are matched case-insensitively), and figures are written to disk.

    SC_DATA_DIR    folder with train/, val/ and test/, each containing benign/ and malignant/
                   (default: data/dataset_skin)
    SC_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/vgg16_gap_20)

Run from the `skin_cancer/` folder:  python src/baselines/vgg16_gap_20_epochs.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("SC_DATA_DIR", "data/dataset_skin"))
OUT_DIR = os.path.abspath(os.environ.get("SC_OUTPUT_DIR", "outputs/vgg16_gap_20"))
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
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    roc_auc_score
)

# =========================
# CONFIG
# =========================
img_size = 224
batch_size = 16
epochs = 20

train_dir = os.path.join(DATA_DIR, "train")
val_dir   = os.path.join(DATA_DIR, "val")
test_dir  = os.path.join(DATA_DIR, "test")

# =========================
# DATA GENERATORS
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

test_gen = val_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# =========================
# CLASS WEIGHTS
# =========================
classes = train_gen.classes
weights = compute_class_weight('balanced', classes=np.unique(classes), y=classes)
class_weights = dict(enumerate(weights))

# =========================
# MODEL (VGG16)
# =========================
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(img_size, img_size, 3)
)

# Freeze most layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# CALLBACKS
# =========================
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('skin_cancer_vgg16.keras', save_best_only=True)
]

# =========================
# TRAIN
# =========================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=callbacks
)

# =========================
# EVALUATION
# =========================
loss, acc = model.evaluate(test_gen)
print(f"\nTest Accuracy: {acc*100:.2f}%")

y_probs = model.predict(test_gen).ravel()
y_pred = (y_probs > 0.5).astype(int)
y_true = test_gen.classes

# =========================
# CLASSIFICATION REPORT
# =========================
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.colorbar()
plt.show()

# =========================
# ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# =========================
# PRECISION-RECALL CURVE
# =========================
precision, recall, _ = precision_recall_curve(y_true, y_probs)
f1 = f1_score(y_true, y_pred)

plt.plot(recall, precision, label=f'F1 = {f1:.3f}')
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()

print(f"ROC-AUC Score: {roc_auc_score(y_true, y_probs):.3f}")

# =========================
# TRAINING CURVES
# =========================
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss")
plt.legend()

plt.show()

# =========================
# SAVE MODEL + TIME
# =========================
model.save("skin_cancer_model.keras")

size_mb = os.path.getsize("skin_cancer_model.keras") / (1024*1024)
print("Model Size:", round(size_mb, 2), "MB")

start = time.time()
_ = model.predict(test_gen)
end = time.time()

print("Inference Time:", round(end-start, 3), "sec")
