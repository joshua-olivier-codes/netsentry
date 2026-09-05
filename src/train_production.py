from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# =========================================================
# NETSENTRY - PRODUCTION MODEL TRAINING
#
# This is NOT the strict unseen-day experiment.
# train.py / evaluate.py / threshold_analysis.py / train_unseen_day.py
# are left untouched and remain the benchmark.
#
# This script trains production models intended for
# deployment:
#
#   - Trains on Monday THROUGH Friday (all CICIDS2017 days),
#     so DDoS / PortScan / Bot are represented in training.
#   - Uses a genuine random stratified TRAIN / VALIDATION / TEST
#     split (60% / 20% / 20%) across the pooled data, NOT a
#     split by day. Once Friday is part of training, it can no
#     longer be called an "unseen day" test.
#   - The binary decision threshold is selected using the
#     VALIDATION split only. The TEST split is touched exactly
#     once, after the threshold is fixed, to report final
#     numbers.
#   - A copy of the untouched TEST split is written to
#     data/processed/production_holdout_test.csv so
#     validate_production.py can run genuine end-to-end API
#     checks on rows the models never saw during training or
#     threshold selection.
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "network_traffic.csv"

MODEL_DIR = BASE_DIR / "models"

BINARY_MODEL_PATH = MODEL_DIR / "intrusion_detector_production.joblib"
MULTICLASS_MODEL_PATH = MODEL_DIR / "intrusion_detector_multiclass_production.joblib"
FEATURE_PATH = MODEL_DIR / "feature_names_production.joblib"
CONFIG_PATH = MODEL_DIR / "production_config.json"

HOLDOUT_PATH = PROCESSED_DIR / "production_holdout_test.csv"

RANDOM_STATE = 42

# Fraction of the pooled data held out for validation and test.
# 60% train / 20% validation / 20% test.
VAL_FRACTION = 0.20
TEST_FRACTION = 0.20

# Thresholds scanned when selecting the binary operating point.
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)

# Multiclass class balancing cap (mirrors train.py's convention).
# Applied to the TRAIN split only, never to validation/test, so
# reported metrics reflect the true, unbalanced class mix.
MAX_PER_CLASS = 20000


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =========================================================
# LOAD PROCESSED DATA
# =========================================================


def load_processed_data():

    print_header("1. LOADING PROCESSED DATASET")

    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:\n{PROCESSED_FILE}\n\n"
            "Run preprocess.py first:\n"
            "    python src/preprocess.py"
        )

    df = pd.read_csv(PROCESSED_FILE, low_memory=False)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    if "target" not in df.columns or "attack_type" not in df.columns:
        raise ValueError(
            "Expected 'target' and 'attack_type' columns from preprocess.py."
        )

    return df


# =========================================================
# NORMALIZE ATTACK LABELS
# =========================================================


def normalize_attack_label(label):

    label = str(label).strip()

    if label.upper() == "BENIGN":
        return "BENIGN"

    # Normalize malformed encoding from CICIDS (same fix as train.py).
    replacements = {
        "Web Attack \ufffd Brute Force": "Web Attack - Brute Force",
        "Web Attack \ufffd XSS": "Web Attack - XSS",
        "Web Attack \ufffd Sql Injection": "Web Attack - Sql Injection",
    }

    return replacements.get(label, label)


# =========================================================
# BUILD SHARED FEATURE MATRIX
# =========================================================
#
# Binary and multiclass production models share ONE feature
# set. This guarantees the "multiclass features are a subset
# of binary features" invariant that the API relies on, and
# removes the need for a separate "binary-only feature" concept.
# =========================================================


def build_feature_matrix(df):

    print_header("2. BUILDING FEATURE MATRIX")

    df = df.copy()

    df["attack_type"] = df["attack_type"].apply(normalize_attack_label)

    drop_columns = [
        column
        for column in df.columns
        if column.lower() in ("label", "target", "attack_type")
    ]

    X = df.drop(columns=drop_columns, errors="ignore")

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = X.fillna(0)

    constant_columns = [column for column in X.columns if X[column].nunique() <= 1]

    if constant_columns:
        print(f"Removing {len(constant_columns)} constant columns.")
        X = X.drop(columns=constant_columns)

    feature_names = list(X.columns)

    print(f"Feature count: {len(feature_names)}")

    y_target = df["target"].copy()
    y_attack_type = df["attack_type"].copy()

    return X, y_target, y_attack_type, feature_names


# =========================================================
# SPLIT
# =========================================================


def split_data(X, y_target, y_attack_type, df):

    print_header("3. TRAIN / VALIDATION / TEST SPLIT")

    print(
        "NOTE: this is a random stratified split across ALL days "
        "(Monday-Friday pooled). It is NOT an unseen-day split."
    )

    indices = np.arange(len(X))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(VAL_FRACTION + TEST_FRACTION),
        random_state=RANDOM_STATE,
        stratify=y_target.iloc[indices],
    )

    relative_test_fraction = TEST_FRACTION / (VAL_FRACTION + TEST_FRACTION)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_fraction,
        random_state=RANDOM_STATE,
        stratify=y_target.iloc[temp_idx],
    )

    print(f"Train: {len(train_idx):,} rows")
    print(f"Validation: {len(val_idx):,} rows")
    print(f"Test: {len(test_idx):,} rows")

    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    return splits


# =========================================================
# BINARY MODEL
# =========================================================


def train_binary_model(X, y_target, splits):

    print_header("4. TRAINING PRODUCTION BINARY MODEL")

    y_binary = (y_target != "BENIGN").astype(int)

    X_train = X.iloc[splits["train"]]
    y_train = y_binary.iloc[splits["train"]]

    X_val = X.iloc[splits["val"]]
    y_val = y_binary.iloc[splits["val"]]

    X_test = X.iloc[splits["test"]]
    y_test = y_binary.iloc[splits["test"]]

    print(f"Train BENIGN/ATTACK: {(y_train == 0).sum():,} / {(y_train == 1).sum():,}")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
    )

    print("\nTraining...")
    model.fit(X_train, y_train)
    print("Training complete.")

    # -----------------------------------------------------
    # Threshold selection on VALIDATION ONLY
    # -----------------------------------------------------

    print_header("5. THRESHOLD SELECTION (VALIDATION SPLIT ONLY)")

    val_probabilities = model.predict_proba(X_val)[:, list(model.classes_).index(1)]

    best_threshold = 0.50
    best_f1 = -1.0
    threshold_table = []

    for threshold in THRESHOLDS:
        preds = (val_probabilities >= threshold).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        precision = precision_score(y_val, preds, zero_division=0)
        recall = recall_score(y_val, preds, zero_division=0)

        threshold_table.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
            marker = "  <- best so far"

        print(
            f"threshold={threshold:.2f}  "
            f"precision={precision:.4f}  "
            f"recall={recall:.4f}  "
            f"f1={f1:.4f}{marker}"
        )

    print(f"\nSelected threshold (validation F1): {best_threshold:.2f}")

    # -----------------------------------------------------
    # Final, single evaluation on TEST split
    # -----------------------------------------------------

    print_header("6. FINAL BINARY EVALUATION (HELD-OUT TEST SPLIT)")

    test_probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    test_preds = (test_probabilities >= best_threshold).astype(int)

    test_accuracy = accuracy_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds, zero_division=0)
    test_recall = recall_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)

    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1:        {test_f1:.4f}")

    print("\nConfusion matrix (rows=true, cols=pred, order=[BENIGN, ATTACK]):")
    print(confusion_matrix(y_test, test_preds, labels=[0, 1]))

    metrics = {
        "binary_val_threshold_table": threshold_table,
        "binary_selected_threshold": best_threshold,
        "binary_test_accuracy": float(test_accuracy),
        "binary_test_precision": float(test_precision),
        "binary_test_recall": float(test_recall),
        "binary_test_f1": float(test_f1),
    }

    return model, best_threshold, metrics, test_preds


# =========================================================
# BALANCED TRAIN SAMPLE FOR MULTICLASS (TRAIN SPLIT ONLY)
# =========================================================


def create_balanced_train_sample(X_train, y_train_type, max_per_class, random_state):

    print(
        "\nBalancing multiclass TRAINING split only "
        "(validation/test stay untouched)..."
    )

    combined = X_train.copy()
    combined["__attack_type__"] = y_train_type.values

    samples = []

    for _, group in combined.groupby("__attack_type__"):
        n = min(len(group), max_per_class)
        samples.append(group.sample(n=n, random_state=random_state))

    result = pd.concat(samples, ignore_index=True).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)

    y_balanced = result.pop("__attack_type__")

    print(f"Balanced training rows: {len(result):,}")
    print(y_balanced.value_counts())

    return result, y_balanced


# =========================================================
# MULTICLASS MODEL
# =========================================================


def train_multiclass_model(X, y_attack_type, splits):

    print_header("7. TRAINING PRODUCTION MULTICLASS MODEL")

    X_train_full = X.iloc[splits["train"]]
    y_train_full = y_attack_type.iloc[splits["train"]]

    X_val = X.iloc[splits["val"]]
    y_val = y_attack_type.iloc[splits["val"]]

    X_test = X.iloc[splits["test"]]
    y_test = y_attack_type.iloc[splits["test"]]

    print("\nTraining split class distribution (before balancing):")
    print(y_train_full.value_counts())

    X_train, y_train = create_balanced_train_sample(
        X_train_full, y_train_full, MAX_PER_CLASS, RANDOM_STATE
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
    )

    print("\nTraining...")
    model.fit(X_train, y_train)
    print("Training complete.")

    print_header("8. MULTICLASS EVALUATION (HELD-OUT TEST SPLIT, NATURAL DISTRIBUTION)")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nOverall accuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=model.classes_))

    print("\nFocus: previously-unseen Friday attack families")
    focus_classes = ["DDoS", "PortScan", "Bot"]

    focus_recall = {}

    for attack_class in focus_classes:
        mask = y_test == attack_class
        n = int(mask.sum())

        if n == 0:
            print(f"  {attack_class:<10} not present in test split")
            continue

        correct = int((predictions[mask.to_numpy()] == attack_class).sum())
        recall = correct / n
        focus_recall[attack_class] = recall

        print(f"  {attack_class:<10} recall={recall:.4f}  ({correct}/{n})")

    metrics = {
        "multiclass_test_accuracy": float(accuracy),
        "multiclass_focus_recall": focus_recall,
        "multiclass_classes": [str(c) for c in model.classes_],
    }

    return model, metrics


# =========================================================
# SAVE HOLDOUT TEST CSV (for validate_production.py)
# =========================================================


def save_holdout_csv(df, splits):

    print_header("9. SAVING HELD-OUT TEST CSV")

    holdout_df = df.iloc[splits["test"]].copy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    holdout_df.to_csv(HOLDOUT_PATH, index=False)

    print(f"Saved {len(holdout_df):,} untouched test rows to:")
    print(HOLDOUT_PATH)

    print("\nHeld-out attack_type distribution:")
    print(holdout_df["attack_type"].value_counts())


# =========================================================
# MAIN
# =========================================================


def main():

    print_header("NETSENTRY - PRODUCTION MODEL TRAINING")

    print("Training data: Monday-Friday (all CICIDS2017 days pooled).")
    print("Split: 60% train / 20% validation / 20% test (random, stratified).")
    print("This is NOT the strict unseen-day benchmark. That benchmark")
    print("(train.py / evaluate.py) is left untouched.")

    df = load_processed_data()

    df["attack_type"] = df["attack_type"].apply(normalize_attack_label)

    X, y_target, y_attack_type, feature_names = build_feature_matrix(df)

    splits = split_data(X, y_target, y_attack_type, df)

    binary_model, threshold, binary_metrics, _ = train_binary_model(
        X, y_target, splits
    )

    multiclass_model, multiclass_metrics = train_multiclass_model(
        X, y_attack_type, splits
    )

    save_holdout_csv(df, splits)

    # -----------------------------------------------------
    # Save models + shared feature list
    # -----------------------------------------------------

    print_header("10. SAVING PRODUCTION ARTIFACTS")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(binary_model, BINARY_MODEL_PATH)
    joblib.dump(multiclass_model, MULTICLASS_MODEL_PATH)
    joblib.dump(feature_names, FEATURE_PATH)

    config = {
        "binary_attack_threshold": threshold,
        "feature_count": len(feature_names),
        "trained_on": "Monday-Friday pooled, random stratified 60/20/20 split",
        **binary_metrics,
        **multiclass_metrics,
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Binary model saved:      {BINARY_MODEL_PATH}")
    print(f"Multiclass model saved:  {MULTICLASS_MODEL_PATH}")
    print(f"Feature names saved:     {FEATURE_PATH}")
    print(f"Config saved:            {CONFIG_PATH}")

    print_header("NETSENTRY PRODUCTION TRAINING COMPLETE")

    print(
        "\nNext steps:\n"
        "  1. Start the API in production mode (default):\n"
        "       python src/app.py\n"
        "  2. Compare against the benchmark by starting it as:\n"
        "       NETSENTRY_MODEL_PROFILE=benchmark python src/app.py\n"
        "  3. Run validate_production.py against the held-out test CSV\n"
        "     for a genuine, leakage-free end-to-end check."
    )


if __name__ == "__main__":
    main()
