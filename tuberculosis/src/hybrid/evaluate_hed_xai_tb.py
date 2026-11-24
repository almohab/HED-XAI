"""HED-XAI-TB: re-evaluate the saved best checkpoint from the stored predictions/split manifest and draw the evaluation figures.

Extracted from the executed notebook `notebooks/08_protocolB_HED-XAI-TB_hybrid_10_and_20_epochs.ipynb` (cell 1); the model definition and the
training/evaluation code are unchanged. Only hard-coded paths were replaced by the environment variables
below, and figures are written to disk instead of being displayed.

    TB_DATA_DIR    folder containing Normal/ and Tuberculosis/ of the TB Chest Radiography Database
                   (default: data/TB_Chest_Radiography_Database)
    TB_OUTPUT_DIR  where models, figures and CSV files are written (default: outputs/HED-XAI-TB)

Run from the `tuberculosis/` folder:  python src/hybrid/evaluate_hed_xai_tb.py
"""
import os
DATA_DIR = os.path.abspath(os.environ.get("TB_DATA_DIR", "data/TB_Chest_Radiography_Database"))
OUT_DIR = os.path.abspath(os.environ.get("TB_OUTPUT_DIR", "outputs/HED-XAI-TB"))
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

"""
HED-XAI-TB: Model Evaluation and Figure Generation
-------------------------------------------------
Evaluates the saved best model and produces:
- Confusion matrix
- ROC curve and ROC-AUC
- Precision–Recall curve and PR-AUC
- Training/validation accuracy and loss
- Classification report and metrics CSV

Update OUTPUT_DIR if your results are stored elsewhere.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# -------------------- CONFIGURATION --------------------
OUTPUT_DIR = Path(OUT_DIR)
MODEL_PATH = OUTPUT_DIR / "HED_XAI_TB_Best.h5"
HISTORY_PATH = OUTPUT_DIR / "HED_XAI_TB_History.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "HED_XAI_TB_Test_Predictions.csv"
MANIFEST_PATH = OUTPUT_DIR / "HED_XAI_TB_Split_Manifest.csv"

CLASS_NAMES = ["Normal", "Tuberculosis"]
POSITIVE_CLASS_NAME = "Tuberculosis"
THRESHOLD = 0.5

# If prediction CSV cannot be used, set this to True to run prediction
# directly from the test split manifest and original dataset images.
PREDICT_FROM_MODEL_IF_NEEDED = True
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# -------------------- LOAD HISTORY --------------------
def find_column(df, candidates):
    """Find a column using case-insensitive candidate names."""
    mapping = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower().strip() in mapping:
            return mapping[candidate.lower().strip()]
    return None


def plot_training_history():
    if not HISTORY_PATH.exists():
        print(f"Training history not found: {HISTORY_PATH}")
        return

    hist = pd.read_csv(HISTORY_PATH)
    epoch_col = find_column(hist, ["epoch"])
    epochs = hist[epoch_col].to_numpy() if epoch_col else np.arange(1, len(hist) + 1)

    train_acc = find_column(hist, ["accuracy", "binary_accuracy"])
    val_acc = find_column(hist, ["val_accuracy", "val_binary_accuracy"])
    train_loss = find_column(hist, ["loss"])
    val_loss = find_column(hist, ["val_loss"])

    if train_acc and val_acc:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, hist[train_acc], marker="o", label="Training accuracy")
        plt.plot(epochs, hist[val_acc], marker="s", label="Validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("HED-XAI-TB: Training and Validation Accuracy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "Evaluation_Accuracy.png", dpi=300)
        plt.show()
    else:
        print("Accuracy columns were not found in the history CSV.")

    if train_loss and val_loss:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, hist[train_loss], marker="o", label="Training loss")
        plt.plot(epochs, hist[val_loss], marker="s", label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("HED-XAI-TB: Training and Validation Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "Evaluation_Loss.png", dpi=300)
        plt.show()
    else:
        print("Loss columns were not found in the history CSV.")


# -------------------- GET TEST LABELS / PROBABILITIES --------------------
def load_predictions_csv():
    """
    Reads test predictions saved by the training script.
    The CSV should contain true labels and predicted probabilities/scores.
    Common column names are handled below.
    """
    if not PREDICTIONS_PATH.exists():
        return None, None

    df = pd.read_csv(PREDICTIONS_PATH)
    print("Prediction CSV columns:", list(df.columns))

    y_col = find_column(
        df,
        ["true_label", "y_true", "actual_label", "actual", "label", "true", "y"]
    )
    p_col = find_column(
        df,
        [
            "tuberculosis_probability", "tb_probability", "probability",
            "predicted_probability", "y_prob", "prob_tb", "score",
            "prediction_probability", "predicted_prob"
        ]
    )
    pred_col = find_column(
        df, ["predicted_label", "y_pred", "predicted", "prediction", "pred_label"]
    )

    if y_col is None:
        print("Could not identify true-label column in prediction CSV.")
        return None, None

    raw_y = df[y_col]
    if raw_y.dtype == object:
        y_map = {
            "normal": 0, "0": 0,
            "tuberculosis": 1, "tb": 1, "1": 1
        }
        y_text = raw_y.astype(str).str.strip().str.lower()
        y_true = y_text.map(y_map)
        if y_true.isna().any():
            print("Unknown true-label values:", raw_y[y_true.isna()].unique())
            return None, None
        y_true = y_true.astype(int).to_numpy()
    else:
        y_true = raw_y.astype(int).to_numpy()

    if p_col is not None:
        y_prob = pd.to_numeric(df[p_col], errors="coerce").to_numpy()
        if np.isnan(y_prob).any():
            print("Probability column contains missing/non-numeric values.")
            return None, None
        # Handle two-column probabilities stored as stringified arrays separately if needed.
        if y_prob.min() < 0 or y_prob.max() > 1:
            print("Probability values are outside [0, 1]. Check the selected column.")
            return None, None
        return y_true, y_prob

    # If only predicted labels exist, metrics/confusion matrix can still be produced,
    # but genuine ROC and PR curves cannot be calculated from hard labels.
    if pred_col is not None:
        print(
            "Only hard predicted labels were found. Confusion matrix and classification "
            "metrics can be computed, but ROC/PR curves require probability scores."
        )
        return y_true, None

    print("No probability or predicted-label column was identified.")
    return None, None


def predict_from_test_manifest():
    """
    Optional fallback: load the saved model and predict images in the test split.
    Requires the split manifest to contain a filename/path and class label.
    """
    if not MODEL_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Prediction CSV, model, or split manifest is missing. "
            "Check OUTPUT_DIR and the filenames."
        )

    manifest = pd.read_csv(MANIFEST_PATH)
    print("Split manifest columns:", list(manifest.columns))

    split_col = find_column(manifest, ["split", "subset", "set", "partition"])
    path_col = find_column(
        manifest, ["filepath", "file_path", "image_path", "path", "filename", "file_name"]
    )
    label_col = find_column(
        manifest, ["class_name", "class", "label", "true_label", "target"]
    )

    if path_col is None or label_col is None:
        raise ValueError(
            "Cannot identify image path and label columns in split manifest. "
            "Inspect the printed column names and update path_col/label_col."
        )

    test_df = manifest.copy()
    if split_col:
        split_text = test_df[split_col].astype(str).str.lower()
        test_df = test_df[split_text.isin(["test", "testing"])]

    if test_df.empty:
        raise ValueError(
            "No test rows found in split manifest. Check the split column values."
        )

    y_text = test_df[label_col].astype(str).str.strip().str.lower()
    y_map = {"normal": 0, "tuberculosis": 1, "tb": 1, "0": 0, "1": 1}
    y_true = y_text.map(y_map)
    if y_true.isna().any():
        raise ValueError(f"Unrecognized class labels: {y_text[y_true.isna()].unique()}")

    # Use paths as stored. If only filenames are stored, prepend dataset root.
    dataset_root = Path(
        DATA_DIR
    )
    paths = []
    for p in test_df[path_col].astype(str):
        pth = Path(p)
        if not pth.is_absolute():
            pth = dataset_root / pth
        paths.append(pth)

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Some test image paths do not exist. Example missing path: " + missing[0]
        )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    def load_image(path):
        img = tf.keras.utils.load_img(path, target_size=IMAGE_SIZE)
        arr = tf.keras.utils.img_to_array(img)
        # HED-XAI-TB training script should use the same preprocessing here.
        # This default assumes the training pipeline rescaled images to [0, 1].
        arr = arr / 255.0
        return arr

    x_test = np.stack([load_image(p) for p in paths]).astype("float32")
    y_prob = model.predict(x_test, batch_size=BATCH_SIZE, verbose=1).reshape(-1)
    return y_true.astype(int).to_numpy(), y_prob


# -------------------- EVALUATION --------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_training_history()

    y_true, y_prob = load_predictions_csv()

    if y_true is None and PREDICT_FROM_MODEL_IF_NEEDED:
        print("Using saved model + test split manifest to create predictions.")
        y_true, y_prob = predict_from_test_manifest()

    if y_true is None:
        raise RuntimeError(
            "Could not load test labels. Verify HED_XAI_TB_Test_Predictions.csv "
            "or the split manifest."
        )

    # If only labels exist, do not fabricate probability curves.
    if y_prob is None:
        print("No probabilities available; ROC and PR curves will be skipped.")
        raise RuntimeError(
            "Please ensure the prediction CSV includes tuberculosis probabilities, "
            "or use the model + test-manifest fallback."
        )

    y_pred = (y_prob >= THRESHOLD).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(values_format="d")
    plt.title("HED-XAI-TB: Test-Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Evaluation_Confusion_Matrix.png", dpi=300)
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("HED-XAI-TB: ROC Curve")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Evaluation_ROC_Curve.png", dpi=300)
    plt.show()

    # Precision–Recall curve
    pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(pr_recall, pr_precision)
    avg_precision = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(
        pr_recall, pr_precision,
        label=f"PR-AUC = {pr_auc:.4f}; Average precision = {avg_precision:.4f}"
    )
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title("HED-XAI-TB: Precision–Recall Curve")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Evaluation_PR_Curve.png", dpi=300)
    plt.show()

    # Metrics from test labels
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    metrics = {
        "model": "HED-XAI-TB",
        "threshold": THRESHOLD,
        "test_samples": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity_recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc_trapezoidal": pr_auc,
        "average_precision": avg_precision,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    print("\nEVALUATION METRICS")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.6f}")
        else:
            print(f"{key:25s}: {value}")

    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "HED_XAI_TB_Evaluation_Metrics.csv", index=False)
    with open(OUTPUT_DIR / "HED_XAI_TB_Evaluation_Metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nEvaluation complete. Figures and metrics saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
