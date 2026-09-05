from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# =========================================================
# NETSENTRY - PRODUCTION MODEL EVALUATION
#
# Evaluates the SAVED production models against the
# untouched production holdout.
#
# IMPORTANT:
# - Does NOT retrain anything.
# - Does NOT modify thresholds.
# - Does NOT use the training/validation split.
# - Uses the exact production feature schema.
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

HOLDOUT_PATH = PROCESSED_DIR / "production_holdout_test.csv"

BINARY_MODEL_PATH = MODEL_DIR / "intrusion_detector_production.joblib"
MULTICLASS_MODEL_PATH = MODEL_DIR / "intrusion_detector_multiclass_production.joblib"
FEATURE_PATH = MODEL_DIR / "feature_names_production.joblib"
CONFIG_PATH = MODEL_DIR / "production_config.json"

OUTPUT_DIR = PROCESSED_DIR / "production_evaluation"

RANDOM_STATE = 42


def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def normalize_attack_label(label):
    label = str(label).strip()

    if label.upper() == "BENIGN":
        return "BENIGN"

    replacements = {
        "Web Attack \ufffd Brute Force": "Web Attack - Brute Force",
        "Web Attack \ufffd XSS": "Web Attack - XSS",
        "Web Attack \ufffd Sql Injection": "Web Attack - Sql Injection",
    }

    return replacements.get(label, label)


# =========================================================
# LOAD ARTIFACTS
# =========================================================

def load_artifacts():

    header("1. LOADING PRODUCTION ARTIFACTS")

    required = [
        HOLDOUT_PATH,
        BINARY_MODEL_PATH,
        MULTICLASS_MODEL_PATH,
        FEATURE_PATH,
        CONFIG_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found:\n{path}")

    binary_model = joblib.load(BINARY_MODEL_PATH)
    multiclass_model = joblib.load(MULTICLASS_MODEL_PATH)
    feature_names = list(joblib.load(FEATURE_PATH))

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    print(f"Binary model:      {BINARY_MODEL_PATH.name}")
    print(f"Multiclass model:  {MULTICLASS_MODEL_PATH.name}")
    print(f"Feature file:      {FEATURE_PATH.name}")
    print(f"Feature count:     {len(feature_names)}")
    print(f"Binary threshold:  {config.get('binary_attack_threshold')}")

    return binary_model, multiclass_model, feature_names, config


# =========================================================
# LOAD HOLDOUT
# =========================================================

def load_holdout():

    header("2. LOADING UNTOUCHED PRODUCTION HOLDOUT")

    df = pd.read_csv(
        HOLDOUT_PATH,
        low_memory=False,
    )

    print(f"Holdout rows: {len(df):,}")
    print(f"Holdout columns: {len(df.columns)}")

    if "target" not in df.columns:
        raise ValueError("Holdout is missing 'target' column.")

    if "attack_type" not in df.columns:
        raise ValueError("Holdout is missing 'attack_type' column.")

    df["attack_type"] = df["attack_type"].apply(normalize_attack_label)

    print("\nAttack distribution:")
    print(df["attack_type"].value_counts().to_string())

    return df


# =========================================================
# BUILD FEATURES
# =========================================================

def build_features(df, feature_names):

    header("3. BUILDING PRODUCTION FEATURE MATRIX")

    missing = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing production features:\n"
            + "\n".join(missing)
        )

    X = df[feature_names].copy()

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        print("WARNING: NaN values detected after conversion.")
        print("Filling NaN values with zero.")

        X = X.fillna(0)

    y_binary = (df["target"] != "BENIGN").astype(int)
    y_attack_type = df["attack_type"].copy()

    print(f"Features supplied: {X.shape[1]}")
    print(f"Rows supplied:     {X.shape[0]:,}")

    return X, y_binary, y_attack_type


# =========================================================
# BINARY EVALUATION
# =========================================================

def evaluate_binary(model, X, y_true, threshold):

    header("4. PRODUCTION BINARY EVALUATION")

    probabilities = model.predict_proba(X)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            f"Binary model does not contain attack class 1. "
            f"Classes: {classes}"
        )

    attack_index = classes.index(1)

    attack_probabilities = probabilities[:, attack_index]

    predictions = (
        attack_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    try:
        auc = roc_auc_score(
            y_true,
            attack_probabilities,
        )
    except ValueError:
        auc = None

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    print(f"Threshold:  {threshold:.2f}")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1:         {f1:.4f}")

    if auc is not None:
        print(f"ROC-AUC:    {auc:.4f}")

    print("\nConfusion matrix")
    print("Rows=true, columns=predicted")
    print("             BENIGN  ATTACK")
    print(f"BENIGN       {matrix[0,0]:6d}  {matrix[0,1]:6d}")
    print(f"ATTACK       {matrix[1,0]:6d}  {matrix[1,1]:6d}")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": None if auc is None else float(auc),
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
        "predictions": predictions,
        "probabilities": attack_probabilities,
    }


# =========================================================
# MULTICLASS EVALUATION
# =========================================================

def evaluate_multiclass(model, X, y_true):

    header("5. PRODUCTION MULTICLASS EVALUATION")

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    print(f"Overall accuracy: {accuracy:.4f}")

    labels = list(model.classes_)

    report_dict = classification_report(
        y_true,
        predictions,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )

    report_text = classification_report(
        y_true,
        predictions,
        labels=labels,
        zero_division=0,
    )

    print("\nClassification report:")
    print(report_text)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=labels,
    )

    print("Confusion matrix:")
    print(matrix)

    print("\nPer-attack recall:")

    per_class = {}

    for label in labels:

        row = report_dict.get(str(label), {})

        precision = float(row.get("precision", 0.0))
        recall = float(row.get("recall", 0.0))
        f1 = float(row.get("f1-score", 0.0))
        support = int(row.get("support", 0))

        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

        print(
            f"{str(label):35s} "
            f"precision={precision:.4f} "
            f"recall={recall:.4f} "
            f"f1={f1:.4f} "
            f"support={support:,}"
        )

    return {
        "accuracy": float(accuracy),
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": matrix.tolist(),
        "labels": [str(label) for label in labels],
        "per_class": per_class,
        "predictions": predictions,
    }


# =========================================================
# FOCUS CLASSES
# =========================================================

def evaluate_focus_classes(y_true, predictions):

    header("6. IMPORTANT ATTACK-FAMILY PERFORMANCE")

    focus_classes = [
        "DDoS",
        "PortScan",
        "Bot",
        "Web Attack - Brute Force",
        "Web Attack - XSS",
        "Web Attack - Sql Injection",
        "Infiltration",
        "Heartbleed",
        "DoS Hulk",
        "DoS GoldenEye",
        "FTP-Patator",
        "SSH-Patator",
        "DoS Slowhttptest",
        "DoS slowloris",
    ]

    results = {}

    for attack_type in focus_classes:

        mask = y_true == attack_type
        total = int(mask.sum())

        if total == 0:
            continue

        actual = y_true[mask]
        predicted = predictions[mask.to_numpy()]

        correct = int(
            (predicted == attack_type).sum()
        )

        recall = correct / total

        results[attack_type] = {
            "correct": correct,
            "total": total,
            "recall": float(recall),
        }

        print(
            f"{attack_type:35s} "
            f"{correct:6d}/{total:<6d} "
            f"recall={recall:.4f}"
        )

    return results


# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(
    binary_results,
    multiclass_results,
    focus_results,
    holdout_rows,
):

    header("7. SAVING PRODUCTION EVALUATION RESULTS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_binary = {
        key: value
        for key, value in binary_results.items()
        if key not in ("predictions", "probabilities")
    }

    clean_multiclass = {
        key: value
        for key, value in multiclass_results.items()
        if key != "predictions"
    }

    results = {
        "evaluation": "production_holdout",
        "holdout_rows": int(holdout_rows),
        "binary": clean_binary,
        "multiclass": clean_multiclass,
        "focus_attack_families": focus_results,
    }

    json_path = OUTPUT_DIR / "production_evaluation.json"

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    # Save per-row predictions
    predictions_path = OUTPUT_DIR / "production_predictions.csv"

    prediction_df = pd.DataFrame(
        {
            "binary_probability": binary_results["probabilities"],
            "binary_prediction": binary_results["predictions"],
            "multiclass_prediction": multiclass_results["predictions"],
        }
    )

    prediction_df.to_csv(
        predictions_path,
        index=False,
    )

    print(f"Saved evaluation JSON:")
    print(json_path)

    print(f"\nSaved per-row predictions:")
    print(predictions_path)


# =========================================================
# MAIN
# =========================================================

def main():

    header("NETSENTRY - PRODUCTION HOLDOUT EVALUATION")

    print(
        "This evaluation uses the untouched production TEST split."
    )

    print(
        "No retraining, threshold selection, or model modification "
        "occurs here."
    )

    binary_model, multiclass_model, feature_names, config = (
        load_artifacts()
    )

    df = load_holdout()

    X, y_binary, y_attack_type = build_features(
        df,
        feature_names,
    )

    # -----------------------------------------------------
    # Schema verification
    # -----------------------------------------------------

    header("SCHEMA VERIFICATION")

    binary_features = list(
        binary_model.feature_names_in_
    )

    multiclass_features = list(
        multiclass_model.feature_names_in_
    )

    if binary_features != feature_names:
        raise ValueError(
            "Binary model feature schema does not match "
            "production feature file."
        )

    if multiclass_features != feature_names:
        raise ValueError(
            "Multiclass model feature schema does not match "
            "production feature file."
        )

    print(f"Binary features:     {len(binary_features)}")
    print(f"Multiclass features: {len(multiclass_features)}")
    print("Schema compatibility: PASS")

    # -----------------------------------------------------
    # Threshold
    # -----------------------------------------------------

    threshold = float(
        config["binary_attack_threshold"]
    )

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    binary_results = evaluate_binary(
        binary_model,
        X,
        y_binary,
        threshold,
    )

    multiclass_results = evaluate_multiclass(
        multiclass_model,
        X,
        y_attack_type,
    )

    focus_results = evaluate_focus_classes(
        y_attack_type,
        multiclass_results["predictions"],
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_results(
        binary_results,
        multiclass_results,
        focus_results,
        len(df),
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    header("NETSENTRY PRODUCTION EVALUATION SUMMARY")

    print(
        f"Holdout rows:          {len(df):,}"
    )

    print(
        f"Feature count:         {len(feature_names)}"
    )

    print(
        f"Binary threshold:      {threshold:.2f}"
    )

    print(
        f"Binary accuracy:       "
        f"{binary_results['accuracy']:.4f}"
    )

    print(
        f"Binary precision:      "
        f"{binary_results['precision']:.4f}"
    )

    print(
        f"Binary recall:         "
        f"{binary_results['recall']:.4f}"
    )

    print(
        f"Binary F1:             "
        f"{binary_results['f1']:.4f}"
    )

    if binary_results["roc_auc"] is not None:
        print(
            f"Binary ROC-AUC:        "
            f"{binary_results['roc_auc']:.4f}"
        )

    print(
        f"Multiclass accuracy:   "
        f"{multiclass_results['accuracy']:.4f}"
    )

    print("\nEvaluation complete.")

    print(
        "\nIMPORTANT: These numbers are for the untouched "
        "production holdout."
    )


if __name__ == "__main__":
    main()
