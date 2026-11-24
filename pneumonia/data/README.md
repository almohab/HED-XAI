# Data

The images are **not** included in this repository. Download the *Chest X-Ray Images (Pneumonia)* dataset of
Kermany et al. (Cell 172(5):1122–1131, 2018; Mendeley Data / Kaggle release with 5,216 / 16 / 624 images) and place it as

```
data/raw/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/
```

- `python src/prepare_dataset.py` pools all images and copies a 70/15/15 split per class to `data/chest_xray_new/`.
  The dissertation split was made **without a seed** (4,099 / 877 / 880); the script now uses `PN_SPLIT_SEED` (default 42),
  so a new run gives a different — but reproducible — partition of the same sizes.
- Patient identifiers are encoded in the file names (`personN_…` for pneumonia, `IM-NNNN-…` / `NORMAL2-IM-NNNN-…` for
  normal images). `python src/patient_level_tools.py split` creates a patient-level 70/15/15 split in
  `data/chest_xray_patient/`; `… overlap data/chest_xray_new` reports how many test patients also occur in train/val.
