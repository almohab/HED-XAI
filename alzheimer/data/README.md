# Data

The MRI images are **not** included in this repository. They come from the OASIS-1 cross-sectional study
(Marcus et al., *J. Cogn. Neurosci.* 2007) and are subject to the OASIS data-use terms.

1. Download the Kaggle distribution of OASIS-1 MRI slices (folders `Non Demented`, `Very mild Dementia`,
   `Mild Dementia`, `Moderate Dementia`; file names such as `OAS1_0140_MR1_mpr-1_142.jpg`) and place its
   `Data/` folder at `data/raw/Data/`.
2. Run `python src/prepare_balanced_dataset.py` to create `data/balanced_data/` with 2,000 randomly sampled
   slices of **Non Demented** and **Very mild Dementia** each. Delete the two other class folders it creates
   if they are present — only these two classes are used.
3. The training scripts compute SHA-256 hashes (0 duplicates were found), split the data 70/15/15 with
   `random_state=42`, and save the split manifest CSV in the output folder.

The OASIS subject ID is part of every file name (`OAS1_XXXX`), so a participant-level split can be built with
`sklearn.model_selection.StratifiedGroupKFold` / `GroupShuffleSplit`.
