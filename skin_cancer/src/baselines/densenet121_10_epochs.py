"""DenseNet121 (last 30 layers trainable), GAP head, class weights, 10 epochs.

Extracted from the executed notebook `notebooks/01_benchmarks_CNN_VGG16_DenseNet121_MobileNetV2_Xception_InceptionV3.ipynb` (cell 5,6); the model definition and the
training/evaluation code are unchanged. Only hard-coded paths were replaced by the environment variables below
(and, in the hybrid script, class folders are matched case-insensitively), and figures are written to disk.

    SC_DATA_DIR    folder with train/, val/ and test/, each containing benign/ and malignant/
                   (default: data/dataset_skin)
    SC_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/densenet121_10)

Run from the `skin_cancer/` folder:  python src/baselines/densenet121_10_epochs.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("SC_DATA_DIR", "data/dataset_skin"))
OUT_DIR = os.path.abspath(os.environ.get("SC_OUTPUT_DIR", "outputs/densenet121_10"))
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
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# =========================
# CONFIG
# =========================
img_size = 224
batch_size = 16
epochs = 10

train_dir = os.path.join(DATA_DIR, "train")
val_dir   = os.path.join(DATA_DIR, "val")
test_dir  = os.path.join(DATA_DIR, "test")

# =========================
# DATA
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True
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
# MODEL (DenseNet121)
# =========================
base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(img_size, img_size, 3)
)

# Freeze most layers
for layer in base_model.layers[:-30]:
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
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ModelCheckpoint('densenet_model.keras', save_best_only=True)
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

y_pred = model.predict(test_gen).ravel()
y_pred_classes = (y_pred > 0.5).astype(int)
y_true = test_gen.classes

# =========================
# REPORT
# =========================
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_classes))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:\n", cm)

plt.imshow(cm, cmap='Blues')
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
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# =========================
# PR CURVE
# =========================
precision, recall, _ = precision_recall_curve(y_true, y_pred)
f1 = f1_score(y_true, y_pred_classes)

plt.plot(recall, precision, label=f'F1 = {f1:.3f}')
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()

print(f"ROC-AUC Score: {roc_auc_score(y_true, y_pred):.3f}")

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
# SAVE + METRICS FOR PAPER & GRAD-CAM
# =========================
import os
import time
import json
import pickle

# -------- SAVE MODEL --------
model.save("final_densenet_model.keras")
print("✅ Model saved")

# -------- SAVE CLASS LABELS --------
with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

# -------- SAVE TRAINING HISTORY --------
with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# -------- MODEL SIZE --------
model_path = "final_densenet_model.keras"
size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"Model Size: {size_mb:.2f} MB")

# -------- TRAINING TIME --------
# (Only works if you defined start_time before training)
try:
    training_time = end_time - start_time
    print(f"Training Time: {training_time/60:.2f} minutes")
except:
    print("⚠️ Training time not recorded (wrap model.fit with time)")

# -------- INFERENCE TIME --------
# Take one batch
sample_batch = next(test_gen)[0]

# Warm-up
_ = model.predict(sample_batch, verbose=0)

# Measure
start_inf = time.time()
_ = model.predict(sample_batch, verbose=0)
end_inf = time.time()

time_per_image = ((end_inf - start_inf) / len(sample_batch)) * 1000
print(f"Inference Time: {time_per_image:.2f} ms/image")

# -------- OPTIONAL: AVERAGE INFERENCE --------
total_time = 0
total_images = 0

for i in range(len(test_gen)):
    images, _ = test_gen[i]

    start = time.time()
    _ = model.predict(images, verbose=0)
    end = time.time()

    total_time += (end - start)
    total_images += len(images)

avg_time = (total_time / total_images) * 1000
print(f"Average Inference Time: {avg_time:.2f} ms/image")
