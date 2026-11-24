"""Evaluation-only script for HED-XAI-AD (loads the best checkpoint and the saved split; no training).

Extracted from the executed notebook `02_HED-XAI-AD_hybrid_models_training_and_evaluation.ipynb` (cell 2); the model definition and the
training/evaluation code are unchanged. Only hard-coded Windows paths were replaced by the
environment variables below, and figures are written to disk instead of being displayed.

    AD_DATA_DIR    folder containing "Non Demented/" and "Very mild Dementia/" (default: data/balanced_data)
    AD_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/HED-XAI-AD)

Run from the `alzheimer/` folder:  python src/hybrid/evaluate_hed_xai_ad.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("AD_DATA_DIR", "data/balanced_data"))
OUT_DIR = os.path.abspath(os.environ.get("AD_OUTPUT_DIR", "outputs/HED-XAI-AD"))
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


# ================================================================
# EVALUATION ONLY: VGG16 + INCEPTIONV3 HYBRID ATTENTION
# NO MODEL TRAINING
#
# Loads the previously trained Best model.
# Uses the original saved train/validation/test split.
#
# Calculates:
# Accuracy, Precision, Sensitivity, Specificity, F1-score,
# ROC-AUC, PR-AUC, Confusion Matrix, Inference Time,
# Model Size and Classification Report.
# ================================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    auc,
    log_loss
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


# ================================================================
# 1. PATHS AND CONFIGURATION
# ================================================================

BASE_PATH = DATA_DIR

OUTPUT_DIR = OUT_DIR

BEST_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "VGG16_InceptionV3_Hybrid_Best.keras"
)

SPLIT_CSV = os.path.join(
    OUTPUT_DIR,
    "Split_Manifest.csv"
)

IMG_SIZE = 224
BATCH_SIZE = 8
THRESHOLD = 0.5

CLASS_NAMES = [
    "Non Demented",
    "Very mild Dementia"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# 2. CHECK SAVED FILES
# ================================================================

if not os.path.isfile(BEST_MODEL_PATH):
    raise FileNotFoundError(
        f"Saved model not found: {BEST_MODEL_PATH}"
    )

if not os.path.isfile(SPLIT_CSV):
    raise FileNotFoundError(
        f"Original split manifest not found: {SPLIT_CSV}"
    )

print("Saved model found.")
print("Original dataset split found.")


# ================================================================
# 3. CUSTOM PREPROCESSING LAYERS
# ================================================================

@tf.keras.utils.register_keras_serializable(
    package="AlzheimerHybrid"
)
class VGG16Preprocessing(tf.keras.layers.Layer):

    def call(self, inputs):
        return tf.keras.applications.vgg16.preprocess_input(
            inputs
        )


@tf.keras.utils.register_keras_serializable(
    package="AlzheimerHybrid"
)
class InceptionV3Preprocessing(tf.keras.layers.Layer):

    def call(self, inputs):
        return tf.keras.applications.inception_v3.preprocess_input(
            inputs
        )


# ================================================================
# 4. LOAD PREVIOUSLY TRAINED MODEL
# ================================================================

print("\nLoading previously trained model...")

best_model = load_model(
    BEST_MODEL_PATH,
    custom_objects={
        "VGG16Preprocessing": VGG16Preprocessing,
        "InceptionV3Preprocessing": InceptionV3Preprocessing
    },
    compile=False
)

print("Model loaded successfully.")

print("\nModel summary:")

best_model.summary()


# ================================================================
# 5. LOAD ORIGINAL DATASET SPLIT
# ================================================================
# This uses the exact split manifest saved before training.
# Therefore, the test images are not randomly reassigned.
# ================================================================

manifest = pd.read_csv(SPLIT_CSV)

test_df = manifest[
    manifest["split"] == "test"
].copy().reset_index(drop=True)

if len(test_df) == 0:
    raise ValueError("No test images found in split manifest.")

if not test_df["image_path"].apply(
    os.path.isfile
).all():
    raise FileNotFoundError(
        "One or more test image paths no longer exist."
    )

y_true = test_df["label"].to_numpy(dtype=int)

print("\nTest images:", len(test_df))

print("\nTest class distribution:")
print(test_df["class_name"].value_counts())


# ================================================================
# 6. CREATE TEST GENERATOR
# ================================================================

test_datagen = ImageDataGenerator()

test_gen = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col=["label"],
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nTest generator created.")


# ================================================================
# 7. WARM-UP AND GENERATE TEST PREDICTIONS
# ================================================================

print("\nGenerating predictions on the independent test set...")

test_gen.reset()

warmup_images, _ = next(test_gen)

_ = best_model(
    warmup_images[:1],
    training=False
).numpy()

test_gen.reset()

inference_start = time.perf_counter()

y_prob = best_model.predict(
    test_gen,
    verbose=1
).ravel()

inference_time = (
    time.perf_counter() - inference_start
)

if len(y_true) != len(y_prob):
    raise ValueError(
        "Number of predictions does not match test labels."
    )

y_pred = (
    y_prob >= THRESHOLD
).astype(int)

print("\nPredictions completed.")


# ================================================================
# 8. CALCULATE CLASSIFICATION METRICS
# ================================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

# Sensitivity = TP / (TP + FN)
sensitivity = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)

# Specificity = TN / (TN + FP)
specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_prob
)

average_precision = average_precision_score(
    y_true,
    y_prob
)

# Binary cross-entropy test loss
test_loss = log_loss(
    y_true,
    y_prob,
    labels=[0, 1]
)

# ROC curve
fpr, tpr, roc_thresholds = roc_curve(
    y_true,
    y_prob
)

# Precision-recall curve
pr_precision, pr_recall, pr_thresholds = (
    precision_recall_curve(
        y_true,
        y_prob
    )
)

pr_auc = auc(
    pr_recall,
    pr_precision
)


# ================================================================
# 9. MODEL-ONLY INFERENCE TIME
# ================================================================
# Measures the model forward pass on already-loaded images.
# Does not include image loading or generator preprocessing.
# ================================================================

test_gen.reset()

sample_images, _ = next(test_gen)

# Warm-up
_ = best_model(
    sample_images,
    training=False
).numpy()

REPEAT_INFERENCE = 30

model_start = time.perf_counter()

for _ in range(REPEAT_INFERENCE):

    batch_output = best_model(
        sample_images,
        training=False
    )

    # Synchronize computation
    _ = batch_output.numpy()

model_total_time = (
    time.perf_counter() - model_start
)

model_only_ms_per_image = (
    model_total_time /
    (REPEAT_INFERENCE * len(sample_images))
) * 1000

# End-to-end prediction time
number_of_images = len(test_df)

end_to_end_ms_per_image = (
    inference_time / number_of_images
) * 1000

throughput = (
    number_of_images / inference_time
    if inference_time > 0
    else 0
)


# ================================================================
# 10. MODEL SIZE AND PARAMETERS
# ================================================================

model_size_mb = (
    os.path.getsize(BEST_MODEL_PATH)
    / (1024 ** 2)
)

total_params = best_model.count_params()


# ================================================================
# 11. PRINT FINAL RESULTS
# ================================================================

print("\n" + "=" * 70)
print("FINAL INDEPENDENT TEST RESULTS")
print("=" * 70)

print(f"Test Loss:                  {test_loss:.4f}")
print(f"Accuracy:                   {accuracy * 100:.2f}%")
print(f"Precision:                  {precision * 100:.2f}%")
print(f"Sensitivity:                {sensitivity * 100:.2f}%")
print(f"Specificity:                {specificity * 100:.2f}%")
print(f"F1-score:                   {f1:.4f}")
print(f"ROC-AUC:                    {roc_auc:.4f}")
print(f"PR-AUC:                     {pr_auc:.4f}")
print(f"Average Precision:          {average_precision:.4f}")

print("\nCONFUSION MATRIX")
print(cm)

print("\nTrue Negative:", tn)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Positive:", tp)

print("\nINFERENCE PERFORMANCE")

print(
    "Model-only inference:",
    round(model_only_ms_per_image, 4),
    "ms/image"
)

print(
    "End-to-end inference:",
    round(end_to_end_ms_per_image, 4),
    "ms/image"
)

print(
    "Total prediction time:",
    round(inference_time, 4),
    "seconds"
)

print(
    "Throughput:",
    round(throughput, 2),
    "images/second"
)

print("\nMODEL INFORMATION")

print("Model size:", round(model_size_mb, 2), "MB")

print("Total parameters:", total_params)


# ================================================================
# 12. CLASSIFICATION REPORT
# ================================================================

print("\nCLASSIFICATION REPORT")

report_text = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print(report_text)

report_df = pd.DataFrame(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True
    )
).transpose()

report_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Classification_Report.csv"
    )
)


# ================================================================
# 13. CONFUSION MATRIX FIGURE
# ================================================================

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    annot_kws={"size": 14}
)

plt.title("VGG16-InceptionV3 Hybrid Attention Model")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Confusion_Matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ================================================================
# 14. ROC CURVE
# ================================================================

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}",
    linewidth=2
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic Curve")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ROC_Curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ================================================================
# 15. PRECISION-RECALL CURVE
# ================================================================

plt.figure(figsize=(7, 6))

plt.plot(
    pr_recall,
    pr_precision,
    label=f"PR-AUC = {pr_auc:.4f}",
    linewidth=2
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Precision_Recall_Curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ================================================================
# 16. SAVE TEST PREDICTIONS
# ================================================================

predictions_df = test_df.copy()

predictions_df["Actual_Class"] = [
    CLASS_NAMES[int(value)]
    for value in y_true
]

predictions_df["Predicted_Class"] = [
    CLASS_NAMES[int(value)]
    for value in y_pred
]

predictions_df["Prediction_Probability"] = y_prob

predictions_df["Correct"] = (
    y_true == y_pred
)

predictions_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Test_Predictions.csv"
    ),
    index=False
)


# ================================================================
# 17. SAVE FINAL METRICS
# ================================================================

results = [

    ("Model", "VGG16-InceptionV3 Hybrid Attention"),

    ("Test Images", number_of_images),

    ("Test Loss", test_loss),

    ("Accuracy (%)", accuracy * 100),

    ("Precision (%)", precision * 100),

    ("Sensitivity (%)", sensitivity * 100),

    ("Specificity (%)", specificity * 100),

    ("F1-score", f1),

    ("ROC-AUC", roc_auc),

    ("PR-AUC", pr_auc),

    ("Average Precision", average_precision),

    ("Model-Only Inference (ms/image)",
     model_only_ms_per_image),

    ("End-to-End Inference (ms/image)",
     end_to_end_ms_per_image),

    ("Total Prediction Time (seconds)",
     inference_time),

    ("Throughput (images/second)",
     throughput),

    ("Model Size (MB)", model_size_mb),

    ("Total Parameters", total_params),

    ("True Negative", tn),

    ("False Positive", fp),

    ("False Negative", fn),

    ("True Positive", tp),

    ("Classification Threshold", THRESHOLD)
]

results_df = pd.DataFrame(
    results,
    columns=["Metric", "Value"]
)

print("\nFINAL METRICS TABLE")

print(results_df.to_string(index=False))

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Final_Metrics.csv"
    ),
    index=False
)


# ================================================================
# 18. COMPLETION
# ================================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETED - NO TRAINING WAS PERFORMED")
print("=" * 70)

print("Results saved in:")
print(OUTPUT_DIR)

print("\nGenerated files:")

print("1. Final_Metrics.csv")
print("2. Classification_Report.csv")
print("3. Test_Predictions.csv")
print("4. Confusion_Matrix.png")
print("5. ROC_Curve.png")
print("6. Precision_Recall_Curve.png")

print("\nYour previously trained model remains unchanged.")
print("=" * 70)
