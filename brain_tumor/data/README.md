# Data

The images are **not** included in this repository. Download the Figshare brain tumour dataset
(J. Cheng, dataset #1512427: 3,064 contrast-enhanced T1-weighted slices from 233 patients, stored as MATLAB v7.3
`.mat` files) and place the `.mat` files under `data/raw/BrainTumorDataset/`.

1. `python src/prepare_dataset.py` converts every slice to an 8-bit JPEG (min–max scaling, transposed matrix) and
   writes it to `data/converted_images/<label>/`, where the Figshare labels are **1 = meningioma, 2 = glioma,
   3 = pituitary**. Rename the three folders to `meningioma`, `glioma` and `pituitary`.
2. The second part of the script shuffles each class with `random.seed(42)` and copies 70/15/15 of it to
   `data/converted_images_split/{train,val,test}/<class>/` → 2,144 / 458 / 462 images.
3. Every `.mat` file also stores the patient ID (`cjdata.PID`). It was not used in the dissertation; use it with
   `GroupShuffleSplit` / `StratifiedGroupKFold` for a patient-level split.
