"""Pool the original Kermany train/val/test folders and re-split each class 70/15/15 into data/chest_xray_new.

NOTE: the original cell shuffled without a seed, so the dissertation split cannot be regenerated exactly; a fixed
seed (PN_SPLIT_SEED, default 42) was added here so that future splits are reproducible.

Extracted from the executed notebook `notebooks/01_benchmarks_resplit_VGG16_DenseNet121_MobileNetV2_CNN_InceptionV3_and_GradCAM.ipynb` (cell 1); the model definition and the
training/evaluation code are unchanged. Only hard-coded Windows paths were replaced by the environment
variables below, and figures are written to disk instead of being displayed.

    PN_RAW_DIR     original Kaggle/Mendeley release with train/, val/, test/ (default: data/raw/chest_xray)
    PN_DATA_DIR    re-split data with train/, val/, test/, each containing NORMAL/ and PNEUMONIA/
                   (default: data/chest_xray_new)
    PN_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/dataset)

Run from the `pneumonia/` folder:  python src/prepare_dataset.py
"""
import os
RAW_DIR = os.path.abspath(os.environ.get("PN_RAW_DIR", "data/raw/chest_xray"))
DATA_DIR = os.path.abspath(os.environ.get("PN_DATA_DIR", "data/chest_xray_new"))
OUT_DIR = os.path.abspath(os.environ.get("PN_OUTPUT_DIR", "outputs/dataset"))
os.makedirs(OUT_DIR, exist_ok=True)
import random
random.seed(int(os.environ.get("PN_SPLIT_SEED", 42)))  # added for reproducibility (not in the original run)
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
import shutil
import random

# ==========================================
# Original Dataset Path
# ==========================================

base_path = RAW_DIR

train_path = os.path.join(base_path, "train")
val_path   = os.path.join(base_path, "val")
test_path  = os.path.join(base_path, "test")

# ==========================================
# New Dataset Path
# ==========================================

new_base_path = DATA_DIR

new_train = os.path.join(new_base_path, "train")
new_val   = os.path.join(new_base_path, "val")
new_test  = os.path.join(new_base_path, "test")

classes = ["NORMAL", "PNEUMONIA"]

# ==========================================
# Create New Folders
# ==========================================

for dataset in [new_train, new_val, new_test]:
    for cls in classes:
        os.makedirs(os.path.join(dataset, cls), exist_ok=True)

# ==========================================
# Collect All Images
# ==========================================

all_images = {}

for cls in classes:

    images = []

    # Read train images
    train_cls = os.path.join(train_path, cls)
    images += [os.path.join(train_cls, img) for img in os.listdir(train_cls)]

    # Read val images
    val_cls = os.path.join(val_path, cls)
    images += [os.path.join(val_cls, img) for img in os.listdir(val_cls)]

    # Read test images
    test_cls = os.path.join(test_path, cls)
    images += [os.path.join(test_cls, img) for img in os.listdir(test_cls)]

    random.shuffle(images)

    all_images[cls] = images

# ==========================================
# Split Dataset
# ==========================================

for cls in classes:

    images = all_images[cls]

    total = len(images)

    train_split = int(total * 0.70)
    val_split   = int(total * 0.15)

    train_imgs = images[:train_split]
    val_imgs   = images[train_split:train_split + val_split]
    test_imgs  = images[train_split + val_split:]

    # Copy Train Images
    for img in train_imgs:
        shutil.copy(img, os.path.join(new_train, cls))

    # Copy Validation Images
    for img in val_imgs:
        shutil.copy(img, os.path.join(new_val, cls))

    # Copy Test Images
    for img in test_imgs:
        shutil.copy(img, os.path.join(new_test, cls))

# ==========================================
# Done
# ==========================================

print("Dataset Successfully Reorganized!")

# ==========================================
# Show Dataset Statistics
# ==========================================

for dataset_name, dataset_path in zip(
    ["TRAIN", "VALIDATION", "TEST"],
    [new_train, new_val, new_test]
):

    print(f"\n{dataset_name}")

    total_images = 0

    for cls in classes:

        cls_path = os.path.join(dataset_path, cls)

        num_images = len(os.listdir(cls_path))

        total_images += num_images

        print(f"{cls}: {num_images}")

    print(f"Total Images: {total_images}")
