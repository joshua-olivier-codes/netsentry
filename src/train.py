from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# =========================================================
# NETSENTRY - MULTICLASS MODEL TRAINING
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "intrusion_detector_multiclass.joblib"

FEATURE_PATH = MODEL_DIR / "feature_names_multiclass.joblib"


# =========================================================
# TRAINING FILES
# =========================================================

TRAINING_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
]


# =========================================================
# FIND FILE
# =========================================================


def find_file(filename):

    matches = list(RAW_DIR.rglob(filename))

    if not matches:

        raise FileNotFoundError(
            f"Could not find:\n{filename}\n\n" f"Searched under:\n{RAW_DIR}"
        )

    return matches[0]


# =========================================================
# CLEAN DATA
# =========================================================


def clean_dataframe(df):

    df.columns = df.columns.astype(str).str.strip()

    if "Label" not in df.columns:

        raise ValueError("Label column not found.")

    df["Label"] = df["Label"].astype(str).str.strip()

    # Remove empty / invalid labels
    df = df[df["Label"].notna()].copy()

    df = df[df["Label"] != ""].copy()

    return df


# =========================================================
# LOAD TRAINING DATA
# =========================================================


def load_training_data():

    frames = []

    print("\nLoading training data...")

    for filename in TRAINING_FILES:

        file = find_file(filename)

        print(f"\nLoading: {file.name}")

        df = pd.read_csv(file, low_memory=False)

        print(f"Original rows: " f"{len(df):,}")

        df = clean_dataframe(df)

        print(f"Clean rows: " f"{len(df):,}")

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    return combined


# =========================================================
# NORMALIZE ATTACK LABELS
# =========================================================


def normalize_attack_label(label):

    label = str(label).strip()

    if label.upper() == "BENIGN":
        return "BENIGN"

    # Normalize malformed encoding from CICIDS
    replacements = {
        "Web Attack � Brute Force": "Web Attack - Brute Force",
        "Web Attack � XSS": "Web Attack - XSS",
        "Web Attack � Sql Injection": "Web Attack - Sql Injection",
    }

    return replacements.get(label, label)


# =========================================================
# SAMPLE PER CLASS
# =========================================================


def create_balanced_sample(df, max_per_class=20000, random_state=42):

    print("\nCreating balanced training sample...")

    df = df.copy()

    df["attack_type"] = df["Label"].apply(normalize_attack_label)

    print("\nAvailable training classes:")

    print(df["attack_type"].value_counts())

    samples = []

    for label, group in df.groupby("attack_type"):

        n = min(len(group), max_per_class)

        sampled = group.sample(n=n, random_state=random_state)

        samples.append(sampled)

    result = pd.concat(samples, ignore_index=True)

    result = result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"\nBalanced sample: " f"{len(result):,} rows")

    print("\nSample distribution:")

    print(result["attack_type"].value_counts())

    return result


# =========================================================
# PREPARE FEATURES
# =========================================================


def prepare_features(df):

    df = df.copy()

    # Target
    y = df["attack_type"].copy()

    # Remove labels
    X = df.drop(
        columns=[
            "Label",
            "target",
            "attack_type",
        ],
        errors="ignore",
    )

    # Keep only numeric features
    X = X.apply(pd.to_numeric, errors="coerce")

    # Replace infinite values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Median imputation
    X = X.fillna(X.median())

    # Any columns that remain entirely invalid
    X = X.fillna(0)

    # Remove constant columns
    constant_columns = [column for column in X.columns if X[column].nunique() <= 1]

    if constant_columns:

        print(f"\nRemoving " f"{len(constant_columns)} " "constant columns.")

        X = X.drop(columns=constant_columns)

    feature_names = list(X.columns)

    return X, y, feature_names


# =========================================================
# MAIN
# =========================================================


def main():

    print("=" * 70)
    print("NETSENTRY - MULTICLASS MODEL TRAINING")
    print("=" * 70)

    print("\nIMPORTANT:")

    print("Training uses Monday + Tuesday + Wednesday + Thursday.")

    print("Friday traffic is NOT used for training.")

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    df = load_training_data()

    print(f"\nTotal Monday-Thursday rows: " f"{len(df):,}")

    # -----------------------------------------------------
    # Balanced sampling
    # -----------------------------------------------------

    df = create_balanced_sample(df, max_per_class=20000)

    # -----------------------------------------------------
    # Features
    # -----------------------------------------------------

    print("\nPreparing features...")

    X, y, feature_names = prepare_features(df)

    print(f"\nSamples: " f"{len(X):,}")

    print(f"Features: " f"{len(feature_names)}")

    # -----------------------------------------------------
    # Train/test split
    # -----------------------------------------------------

    print("\nCreating stratified train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training samples: " f"{len(X_train):,}")

    print(f"Internal test samples: " f"{len(X_test):,}")

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    print("\n" + "=" * 70)

    print("TRAINING RANDOM FOREST")

    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
    )

    print("\nConfiguration:")

    print("  Algorithm: Random Forest")

    print("  Trees: 200")

    print("  Maximum depth: 25")

    print("  Minimum samples per leaf: 2")

    print("  Class weighting: balanced_subsample")

    print("  Features per split: sqrt")

    print("  CPU workers: all available")

    print("\nTraining...")

    model.fit(X_train, y_train)

    print("Training complete.")

    # -----------------------------------------------------
    # Internal evaluation
    # -----------------------------------------------------

    print("\n" + "=" * 70)

    print("INTERNAL TEST EVALUATION")

    print("=" * 70)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: " f"{accuracy:.4f}")

    print("\nClassification Report:")

    print(classification_report(y_test, predictions, zero_division=0))

    print("Confusion Matrix:")

    print(confusion_matrix(y_test, predictions, labels=model.classes_))

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------

    print("\n" + "=" * 70)

    print("TOP FEATURE IMPORTANCE")

    print("=" * 70)

    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )

    print(importance.head(20).to_string())

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    joblib.dump(feature_names, FEATURE_PATH)

    print("\nModel saved to:")

    print(MODEL_PATH)

    print("\nFeature names saved to:")

    print(FEATURE_PATH)

    print("\n" + "=" * 70)

    print("NETSENTRY MULTICLASS TRAINING COMPLETE")

    print("=" * 70)


if __name__ == "__main__":
    main()
