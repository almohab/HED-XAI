"""Custom CNN from scratch, 150x150, 20 epochs.

Extracted from the executed notebook `notebooks/01_benchmarks_CNN_VGG16_DenseNet121_MobileNetV2_Xception_InceptionV3.ipynb` (cell 1); the model definition and the
training/evaluation code are unchanged. Only hard-coded paths were replaced by the environment variables below
(and, in the hybrid script, class folders are matched case-insensitively), and figures are written to disk.

    SC_DATA_DIR    folder with train/, val/ and test/, each containing benign/ and malignant/
                   (default: data/dataset_skin)
    SC_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/cnn_20)

Run from the `skin_cancer/` folder:  python src/baselines/cnn_20_epochs.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("SC_DATA_DIR", "data/dataset_skin"))
OUT_DIR = os.path.abspath(os.environ.get("SC_OUTPUT_DIR", "outputs/cnn_20"))
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

import zipfile
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Step 1: Define the path for your extracted dataset
extract_path = DATA_DIR  # Path to the extracted folder
train_dir = os.path.join(extract_path, 'train')
val_dir = os.path.join(extract_path, 'val')
test_dir = os.path.join(extract_path, 'test')

# Step 2: Load and preprocess the data
img_size = 150  # Resize images to 150x150

def load_images(data_dir, img_size):
    data = []
    labels = []
    categories = ['BENIGN', 'MALIGNANT']  # Adjust according to your dataset labels
    
    for category in categories:
        path = os.path.join(data_dir, category)
        class_label = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                resized_img = cv2.resize(img_array, (img_size, img_size))
                data.append(resized_img)
                labels.append(class_label)
            except Exception as e:
                print(f"Error loading image: {e}")
    
    return np.array(data), np.array(labels)

train_data, train_labels = load_images(train_dir, img_size)
val_data, val_labels = load_images(val_dir, img_size)
test_data, test_labels = load_images(test_dir, img_size)

# Step 3: Normalize the data
X_train = train_data / 255.0
X_val = val_data / 255.0
X_test = test_data / 255.0

# Reshape data to fit the model input requirements (150, 150, 3)
X_train = X_train.reshape(-1, img_size, img_size, 3)
X_val = X_val.reshape(-1, img_size, img_size, 3)
X_test = X_test.reshape(-1, img_size, img_size, 3)

# Convert labels to NumPy arrays
y_train = np.array(train_labels)
y_val = np.array(val_labels)
y_test = np.array(test_labels)

# Step 4: Data Augmentation
data_generator = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

data_generator.fit(X_train)

# Step 5: Build the CNN model
model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPool2D(pool_size=(2, 2)),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    
    Flatten(),
    
    Dense(128, activation='relu'),
    Dropout(0.5),
    
    Dense(1, activation='sigmoid')  # Binary classification (Normal vs Malignant)
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Step 6: Train the model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=0.00001)

history = model.fit(
    data_generator.flow(X_train, y_train, batch_size=32),
    epochs=20,  # Increased epochs for better training
    validation_data=(X_val, y_val),
    callbacks=[reduce_lr]
)

# Step 7: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 8: Confusion Matrix and Classification Report
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

print("Classification Report:\n", classification_report(y_test, y_pred_classes))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_classes))

# Step 9: Visualize training results
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
