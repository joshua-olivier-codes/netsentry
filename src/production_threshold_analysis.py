from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Reuse the exact same feature-building / splitting logic used to
# train the production models. Importing this module does NOT
# retrain anything -- main() is guarded by __name__ == "__main__".
import train_production as tp

# =========================================================
# NETSENTRY - PRODUCTION THRESHOLD / CONFIDENCE ANALYSIS
#
# Goal: find better BINARY_ATTACK_THRESHOLD and
# UNKNOWN_TYPE_THRESHOLD values for the already-trained
# production models, WITHOUT:
#
#   (a) retraining anything, or
#   (b) touching data/processed/production_holdout_test.csv
#       (the untouched TEST split).
#
# How: train_production.py used a fixed random_state and a
# deterministic train_test_split. Re-running that exact same
# split logic against the same network_traffic.csv reproduces
# the identical VALIDATION split without needing to have saved
# it separately. We verify this reconstruction against the
# saved TEST holdout before trusting it -- if the two overlap,
# or don't match expected sizes, we abort rather than silently
# produce misleading numbers.
#
# This script only reads model files and prints analysis. It
# does not modify app.py, production_config.json, or any model
# file. Decide on new threshold values from this output, then
# apply them by hand (or ask for a follow-up patch).
# =========================================================

BASE_DIR = tp.BASE_DIR

BINARY_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

UNKNOWN_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75]

# The classes flagged as weak in the last validation run.
FOCUS_CLASSES = [
    "Bot",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Web Attack - Sql Injection",
    "Infiltration",
]


def print_header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def fail(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


# =========================================================
# LOAD MODELS (NO RETRAINING)
# =========================================================


def load_models():

    print_header("1. LOADING PRODUCTION MODELS (READ-ONLY)")

    for path in (tp.BINARY_MODEL_PATH, tp.MULTICLASS_MODEL_PATH, tp.FEATURE_PATH):
        if not path.exists():
            fail(f"Missing:\n{path}\n\nRun train_production.py first.")

    binary_model = joblib.load(tp.BINARY_MODEL_PATH)
    multiclass_model = joblib.load(tp.MULTICLASS_MODEL_PATH)
    feature_names = list(joblib.load(tp.FEATURE_PATH))

    print(f"Binary model:     {tp.BINARY_MODEL_PATH.name}")
    print(f"Multiclass model: {tp.MULTICLASS_MODEL_PATH.name}")
    print(f"Features:         {len(feature_names)}")

    return binary_model, multiclass_model, feature_names


# =========================================================
# RECONSTRUCT THE VALIDATION SPLIT
# =========================================================


def reconstruct_validation_split():

    print_header("2. RECONSTRUCTING THE VALIDATION SPLIT")

    print(
        "Re-deriving the same train/val/test split train_production.py\n"
        "used (same random_state, same stratify column, same source\n"
        "file). No retraining occurs -- this only reproduces indices."
    )

    df = tp.load_processed_data()
    df["attack_type"] = df["attack_type"].apply(tp.normalize_attack_label)

    X, y_target, y_attack_type, feature_names = tp.build_feature_matrix(df)
    splits = tp.split_data(X, y_target, y_attack_type, df)

    return df, X, y_target, y_attack_type, splits, feature_names


def verify_against_holdout(df, splits):

    print_header("3. VERIFYING RECONSTRUCTION AGAINST SAVED TEST HOLDOUT")

    if not tp.HOLDOUT_PATH.exists():
        fail(
            f"Saved test holdout not found:\n{tp.HOLDOUT_PATH}\n\n"
            "Cannot verify the reconstructed split is correct. "
            "Run train_production.py first."
        )

    saved_holdout = pd.read_csv(tp.HOLDOUT_PATH, low_memory=False)

    reconstructed_test = df.iloc[splits["test"]]

    saved_dist = saved_holdout["attack_type"].value_counts().sort_index()
    reconstructed_dist = reconstructed_test["attack_type"].value_counts().sort_index()

    if len(saved_holdout) != len(reconstructed_test) or not saved_dist.equals(
        reconstructed_dist
    ):
        fail(
            "Reconstructed TEST split does not match the saved holdout "
            "CSV. This means network_traffic.csv changed since training, "
            "or the split logic is no longer reproducible. Refusing to "
            "proceed -- re-run train_production.py to get matching, "
            "trustworthy splits before doing threshold analysis."
        )

    # Overlap check: validation and reconstructed test must be disjoint.
    overlap = set(splits["val"]).intersection(set(splits["test"]))

    if overlap:
        fail(
            f"Validation and test indices overlap ({len(overlap)} rows). "
            "Refusing to proceed."
        )

    print("PASS - reconstructed TEST split exactly matches the saved holdout.")
    print("PASS - validation and test indices are disjoint.")
    print(
        "\nThis confirms the reconstructed VALIDATION split is the same "
        "one used during training, and that this analysis never reads "
        "the untouched test rows."
    )


# =========================================================
# BINARY THRESHOLD SWEEP (VALIDATION ONLY)
# =========================================================


def binary_threshold_sweep(binary_model, X, y_target, splits):

    print_header("4. BINARY THRESHOLD SWEEP (VALIDATION SPLIT)")

    X_val = X.iloc[splits["val"]]
    y_val = (y_target.iloc[splits["val"]] != "BENIGN").astype(int)

    attack_index = list(binary_model.classes_).index(1)
    probabilities = binary_model.predict_proba(X_val)[:, attack_index]

    print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}")

    rows = []

    for threshold in BINARY_THRESHOLDS:
        preds = (probabilities >= threshold).astype(int)

        precision = precision_score(y_val, preds, zero_division=0)
        recall = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)

        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        print(f"{threshold:<12}{precision:<12.4f}{recall:<12.4f}{f1:<12.4f}")

    best = max(rows, key=lambda r: r["f1"])

    print(f"\nBest F1 in this range: threshold={best['threshold']} f1={best['f1']:.4f}")

    print(
        "\nNOTE: this is the same metric train_production.py already "
        "optimized over a wider threshold range (0.05-0.95) on this "
        "same validation split. Use this table to see the shape of the "
        "curve around the currently-selected threshold, not to pick a "
        "materially different one unless the numbers clearly justify it."
    )

    return rows


# =========================================================
# MULTICLASS CONFIDENCE / UNKNOWN THRESHOLD ANALYSIS
# =========================================================


def multiclass_unknown_analysis(multiclass_model, X, y_attack_type, splits):

    print_header("5. MULTICLASS CONFIDENCE / UNKNOWN-THRESHOLD ANALYSIS")

    print(
        "Restricting to validation rows that are actual attacks "
        "(true attack_type != BENIGN), mirroring Stage 2 of the API, "
        "which only ever runs on samples Stage 1 already flagged ATTACK."
    )

    val_idx = splits["val"]
    X_val = X.iloc[val_idx]
    y_val_type = y_attack_type.iloc[val_idx]

    attack_mask = (y_val_type != "BENIGN").to_numpy()

    X_attacks = X_val.iloc[attack_mask]
    y_attacks = y_val_type.iloc[attack_mask]

    probabilities = multiclass_model.predict_proba(X_attacks)
    classes = list(multiclass_model.classes_)

    top1_index = np.argmax(probabilities, axis=1)
    top1_label = np.array([classes[i] for i in top1_index])
    top1_confidence = probabilities[np.arange(len(probabilities)), top1_index]

    results = pd.DataFrame(
        {
            "true_type": y_attacks.to_numpy(),
            "predicted_type": top1_label,
            "confidence": top1_confidence,
        }
    )

    # -----------------------------------------------------
    # Confidence diagnostics per class (threshold-independent)
    # -----------------------------------------------------

    print_header("5a. CONFIDENCE WHEN THE MODEL PICKS THE RIGHT CLASS")

    print(
        f"{'Class':<30}{'n':<8}{'Top-1 acc':<12}"
        f"{'Median conf':<14}{'Mean conf'}"
    )

    for attack_class in sorted(results["true_type"].unique()):

        subset = results[results["true_type"] == attack_class]
        n = len(subset)

        top1_acc = (subset["predicted_type"] == attack_class).mean()

        correct = subset[subset["predicted_type"] == attack_class]

        median_conf = correct["confidence"].median() if len(correct) else float("nan")
        mean_conf = correct["confidence"].mean() if len(correct) else float("nan")

        print(
            f"{attack_class:<30}{n:<8}{top1_acc:<12.4f}"
            f"{median_conf:<14.4f}{mean_conf:.4f}"
        )

    # -----------------------------------------------------
    # Effective type accuracy under each UNKNOWN threshold
    #
    # This mirrors app.py's exact logic:
    #   predicted_type == BENIGN         -> UNKNOWN_UNSEEN
    #   confidence >= UNKNOWN threshold  -> predicted_type
    #   else                             -> UNKNOWN_UNSEEN
    # -----------------------------------------------------

    print_header("5b. EFFECTIVE TYPE ACCURACY PER UNKNOWN THRESHOLD")

    header = f"{'Class':<30}" + "".join(f"t={t:<8.2f}" for t in UNKNOWN_THRESHOLDS)
    print(header)

    table = {}

    for attack_class in sorted(results["true_type"].unique()):

        subset = results[results["true_type"] == attack_class]
        n = len(subset)

        row_values = []

        for threshold in UNKNOWN_THRESHOLDS:

            surfaced = (
                (subset["predicted_type"] == attack_class)
                & (subset["predicted_type"] != "BENIGN")
                & (subset["confidence"] >= threshold)
            )

            effective_accuracy = surfaced.sum() / n if n else 0.0
            row_values.append(effective_accuracy)

        table[attack_class] = row_values

        formatted = "".join(f"{v:<10.3f}" for v in row_values)
        print(f"{attack_class:<30}{formatted}")

    # -----------------------------------------------------
    # Macro average across the previously-weak classes
    # -----------------------------------------------------

    print_header("5c. MACRO AVERAGE ACROSS PREVIOUSLY-WEAK CLASSES")

    print(", ".join(FOCUS_CLASSES))
    print()

    present_focus = [c for c in FOCUS_CLASSES if c in table]

    if not present_focus:
        print("None of the focus classes appear in this validation split.")
    else:
        print(f"{'Threshold':<12}{'Macro effective accuracy (focus classes)'}")

        for i, threshold in enumerate(UNKNOWN_THRESHOLDS):
            values = [table[c][i] for c in present_focus]
            macro = float(np.mean(values))
            print(f"{threshold:<12.2f}{macro:.4f}")

    return table


# =========================================================
# MAIN
# =========================================================


def main():

    print_header("NETSENTRY - PRODUCTION THRESHOLD / CONFIDENCE ANALYSIS")

    print(
        "This script does NOT retrain models and does NOT touch\n"
        "data/processed/production_holdout_test.csv. It reconstructs\n"
        "the VALIDATION split used during training and analyzes the\n"
        "already-trained models against it."
    )

    binary_model, multiclass_model, feature_names = load_models()

    df, X, y_target, y_attack_type, splits, split_feature_names = (
        reconstruct_validation_split()
    )

    if split_feature_names != feature_names:
        fail(
            "Reconstructed feature set does not match the saved "
            "feature_names_production.joblib. Refusing to proceed."
        )

    verify_against_holdout(df, splits)

    binary_threshold_sweep(binary_model, X, y_target, splits)

    multiclass_unknown_analysis(multiclass_model, X, y_attack_type, splits)

    print_header("ANALYSIS COMPLETE")

    print(
        "Use section 5c to pick a new UNKNOWN_TYPE_THRESHOLD, and section "
        "4 to sanity-check (not necessarily change) BINARY_ATTACK_THRESHOLD.\n"
        "The test holdout was never read by this script -- it remains "
        "usable for one more genuinely untouched final check after you "
        "apply new settings."
    )


if __name__ == "__main__":
    main()
