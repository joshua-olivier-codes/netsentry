from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# =========================================================
# NETSENTRY - STRICT UNSEEN-DAY BINARY EVALUATION
#
# TRAIN: Monday -> Thursday
# TEST:  Friday ONLY
#
# IMPORTANT:
# Binary model uses 78 features.
# Multiclass model uses 70 features.
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "intrusion_detector.joblib"
FEATURE_PATH = MODEL_DIR / "feature_names_unseen_day.joblib"

RANDOM_STATE = 42


# =========================================================
# FILE DISCOVERY
# =========================================================

def find_friday_files():

    files = sorted(RAW_DIR.rglob("*.csv"))

    friday_files = [
        file for file in files
        if "friday" in file.name.lower()
    ]

    if not friday_files:
        raise FileNotFoundError(
            f"No Friday CSV files found under:\n{RAW_DIR}"
        )

    return friday_files


# =========================================================
# LOAD FRIDAY DATA
# =========================================================

def load_friday_data(files):

    print("\n" + "=" * 70)
    print("LOADING FRIDAY TEST DATA")
    print("=" * 70)

    frames = []

    for file in files:

        print(f"\nLoading: {file.name}")

        df = pd.read_csv(file, low_memory=False)

        print(f"Original rows: {len(df):,}")

        # Normalize column names
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
        )

        if "Label" not in df.columns:
            raise ValueError(
                f"'Label' column not found in {file.name}"
            )

        # Clean labels
        df["Label"] = (
            df["Label"]
            .astype(str)
            .str.strip()
        )

        # Remove invalid numeric values later
        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    print(
        f"\nTotal Friday rows: "
        f"{len(combined):,}"
    )

    return combined


# =========================================================
# CREATE BINARY TARGET
# =========================================================

def create_target(df):

    df = df.copy()

    df["attack_type"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    df["target"] = (
        df["attack_type"]
        .str.upper()
        .ne("BENIGN")
        .astype(int)
    )

    return df


# =========================================================
# PREPARE BINARY FEATURES
# =========================================================

def prepare_features(df, feature_names):

    print("\n" + "=" * 70)
    print("PREPARING BINARY FEATURES")
    print("=" * 70)

    df = df.copy()

    missing = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing:

        print("\nERROR: Missing model features:")

        for feature in missing:
            print(f"  - {feature}")

        raise ValueError(
            f"{len(missing)} binary model features "
            "are missing from Friday data."
        )

    # Extract EXACT model feature order
    X = df[feature_names].copy()

    # Numeric conversion
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace infinities
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove invalid rows
    valid_rows = X.notna().all(axis=1)

    removed = (~valid_rows).sum()

    if removed > 0:

        print(
            f"\nRemoving {removed:,} "
            "rows containing invalid values."
        )

    X = X.loc[valid_rows].copy()

    y = df.loc[
        valid_rows,
        "target"
    ].copy()

    attack_types = df.loc[
        valid_rows,
        "attack_type"
    ].copy()

    print(
        f"\nSamples: {len(X):,}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    return X, y, attack_types


# =========================================================
# DISTRIBUTION
# =========================================================

def show_distribution(attack_types, y):

    print("\n" + "=" * 70)
    print("FRIDAY CLASS DISTRIBUTION")
    print("=" * 70)

    print("\nAttack types:")

    print(
        attack_types
        .value_counts()
        .to_string()
    )

    print("\nBinary classes:")

    binary_distribution = (
        y.value_counts()
        .sort_index()
        .rename({
            0: "BENIGN",
            1: "ATTACK",
        })
    )

    print(
        binary_distribution.to_string()
    )


# =========================================================
# EVALUATION
# =========================================================

def evaluate(model, X, y, attack_types):

    print("\n" + "=" * 70)
    print("OVERALL UNSEEN-DAY RESULTS")
    print("=" * 70)

    print("\nGenerating predictions...")

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    # Find probability of ATTACK
    classes = list(model.classes_)

    if 1 not in classes:

        raise ValueError(
            f"Binary model does not contain "
            f"class 1. Classes: {classes}"
        )

    attack_index = classes.index(1)

    attack_probabilities = probabilities[:, attack_index]

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y,
        predictions
    )

    roc_auc = roc_auc_score(
        y,
        attack_probabilities
    )

    attack_detection_rate = (
        ((predictions == 1) & (y == 1)).sum()
        / (y == 1).sum()
    )

    print(
        f"\nBinary accuracy "
        f"(BENIGN vs ATTACK): "
        f"{accuracy:.4f}"
    )

    print(
        f"Attack detection rate: "
        f"{attack_detection_rate:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc:.4f}"
    )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    print("\nBinary classification report:")

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "BENIGN",
                "ATTACK",
            ],
            digits=4,
            zero_division=0,
        )
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    )

    print("Confusion Matrix:")

    print(matrix)

    print("\nMatrix format:")

    print(
        "[[True Benign,  False Attack],"
    )

    print(
        " [False Benign, True Attack]]"
    )

    # -----------------------------------------------------
    # Per attack type
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-ATTACK-TYPE DETECTION")
    print("=" * 70)

    print(
        "\n"
        f"{'Attack Type':<35}"
        f"{'Samples':>10}"
        f"{'Detected':>12}"
        f"{'Missed':>10}"
        f"{'Recall':>10}"
    )

    print("-" * 80)

    unique_types = attack_types.unique()

    for attack_type in sorted(
        unique_types,
        key=lambda x: str(x)
    ):

        mask = (
            attack_types == attack_type
        )

        total = int(mask.sum())

        detected = int(
            ((predictions == 1) & mask).sum()
        )

        missed = total - detected

        recall = (
            detected / total
            if total > 0
            else 0.0
        )

        print(
            f"{str(attack_type):<35}"
            f"{total:>10,}"
            f"{detected:>12,}"
            f"{missed:>10,}"
            f"{recall * 100:>9.2f}%"
        )

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "attack_detection_rate":
            attack_detection_rate,
        "confusion_matrix": matrix,
    }


# =========================================================
# SAVE PREDICTIONS
# =========================================================

def save_predictions(
    attack_types,
    y,
    predictions,
    probabilities,
):

    output_dir = (
        BASE_DIR
        / "data"
        / "processed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / "friday_predictions.csv"
    )

    result = pd.DataFrame(
        {
            "attack_type":
                attack_types.values,

            "actual":
                y.values,

            "prediction":
                predictions,

            "attack_probability":
                probabilities,
        }
    )

    result["actual_label"] = (
        result["actual"]
        .map({
            0: "BENIGN",
            1: "ATTACK",
        })
    )

    result["prediction_label"] = (
        result["prediction"]
        .map({
            0: "BENIGN",
            1: "ATTACK",
        })
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        "\nPredictions saved to:"
    )

    print(output_file)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("NETSENTRY - STRICT UNSEEN-DAY BINARY EVALUATION")
    print("=" * 70)

    print(
        "\nTraining:"
        "\nMonday + Tuesday + Wednesday + Thursday"
    )

    print(
        "\nTesting:"
        "\nFriday ONLY"
    )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    print("\nLoading binary model...")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Binary model not found:\n"
            f"{MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    print("Model loaded.")

    print(
        f"Model expects: "
        f"{len(model.feature_names_in_)} features"
    )

    print(
        f"Binary classes: "
        f"{list(model.classes_)}"
    )

    # -----------------------------------------------------
    # Load feature names
    # -----------------------------------------------------

    if not FEATURE_PATH.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n"
            f"{FEATURE_PATH}"
        )

    feature_names = joblib.load(
        FEATURE_PATH
    )

    print(
        f"Feature file contains: "
        f"{len(feature_names)} features"
    )

    # -----------------------------------------------------
    # Compatibility check
    # -----------------------------------------------------

    model_features = list(
        model.feature_names_in_
    )

    if model_features != list(
        feature_names
    ):

        raise ValueError(
            "\nMODEL / FEATURE FILE MISMATCH!\n\n"
            f"Model features: {len(model_features)}\n"
            f"Feature file: {len(feature_names)}\n"
        )

    print(
        "Model / feature schema: "
        "COMPATIBLE"
    )

    # -----------------------------------------------------
    # Friday data
    # -----------------------------------------------------

    friday_files = find_friday_files()

    df = load_friday_data(
        friday_files
    )

    df = create_target(df)

    # -----------------------------------------------------
    # Distribution
    # -----------------------------------------------------

    show_distribution(
        df["attack_type"],
        df["target"]
    )

    # -----------------------------------------------------
    # Features
    # -----------------------------------------------------

    X, y, attack_types = prepare_features(
        df,
        feature_names
    )

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    results = evaluate(
        model,
        X,
        y,
        attack_types
    )

    # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    attack_index = list(
        model.classes_
    ).index(1)

    attack_probabilities = (
        probabilities[:, attack_index]
    )

    save_predictions(
        attack_types,
        y,
        predictions,
        attack_probabilities
    )

    # -----------------------------------------------------
    # Final
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("UNSEEN-DAY EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"\nFinal binary accuracy: "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Final attack detection rate: "
        f"{results['attack_detection_rate']:.4f}"
    )

    print(
        f"Final ROC-AUC: "
        f"{results['roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()