# Data

The images are **not** included in this repository. Download the Kaggle dataset *Skin Cancer: Malignant vs. Benign*
(processed images from the ISIC Archive; 2,637 training and 660 test images) and arrange it as

```
data/dataset_skin/
├── train/{benign,malignant}/   # 2,110 images (1,152 / 958) after separating the validation folder
├── val/{benign,malignant}/     #   527 images (288 / 239), separated from the distributed training set
└── test/{benign,malignant}/    #   660 images (360 / 300), the distributed test set
```

or point `SC_DATA_DIR` to that folder. The validation folder used in the dissertation was created once and kept fixed;
its file list is written by the hybrid script to `HED_XAI_SC_Split_Manifest.csv` (with the SHA-256 of every file), which
should be used to reproduce the partition exactly. The images carry no patient, lesion or skin-phototype metadata.
