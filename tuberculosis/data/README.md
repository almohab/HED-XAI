# Data

The images are **not** included in this repository. Download the *Tuberculosis (TB) Chest X-ray Database*
(Rahman et al., "Reliable tuberculosis detection using chest X-ray with deep learning, segmentation and
visualization", IEEE Access 8, 191586–191601, 2020; distributed on Kaggle) and place it as

```
data/TB_Chest_Radiography_Database/
├── Normal/          # 3,500 images
└── Tuberculosis/    # 3,500 images
```

or point `TB_DATA_DIR` to that folder.

- **Protocol B (used for all comparisons):** `src/hybrid/train_hed_xai_tb.py` hashes every file with SHA-256,
  removes the 330 byte-identical duplicates (all in the Tuberculosis class → 6,670 unique images) and makes a
  stratified 70/15/15 split with seed 42 (4,669 / 1,000 / 1,001; test 525 Normal / 476 TB). The split is written to
  `HED_XAI_TB_Split_Manifest.csv` together with the hash of every file.
- **Protocol A (exploratory):** the baseline scripts split the original 7,000 images (duplicates included)
  70/15/15 with `train_test_split(..., random_state=42, stratify=labels)` → 4,900 / 1,050 / 1,050.
- No patient identifiers are distributed with the database, so a patient-level split is not possible.
