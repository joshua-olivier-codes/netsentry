from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# =========================================================
# NETSENTRY - BINARY THRESHOLD ANALYSIS
# =========================================================
#
# Purpose:
# Determine whether the poor Friday attack detection rate
# can be improved by changing the binary ATTACK threshold.
#
# Training:
#   Monday + Tuesday + Wednesday + Thursday
#
# Testing:
#   Friday ONLY
#
# IMPORTANT:
# Friday data is used ONLY for evaluation.
# It is NOT used to train the model.
# =========================================================


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "intrusion_detector.joblib"

FEATURE_PATH = BASE_DIR / "models" / "feature_names_unseen_day.joblib"

DATA_DIR = BASE_DIR / "data" / "raw"


FRIDAY_FILES = [
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


# =========================================================
# THRESHOLDS TO TEST
# =========================================================

THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


# =========================================================
# HELPERS
# =========================================================


def print_header(title):

    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def fail(message):

    print(f"\nERROR: {message}")
    raise SystemExit(1)


# =========================================================
# LOAD MODEL
# =========================================================


def load_model():

    print_header("1. LOADING BINARY MODEL")

    if not MODEL_PATH.exists():

        fail(f"Binary model not found:\n" f"{MODEL_PATH}")

    print(f"Model: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    print("Model loaded.")

    print(f"Classes: {list(model.classes_)}")

    return model


# =========================================================
# LOAD FEATURES
# =========================================================


def load_features():

    print_header("2. LOADING FEATURES")

    if not FEATURE_PATH.exists():

        fail(f"Feature file not found:\n" f"{FEATURE_PATH}")

    feature_names = list(joblib.load(FEATURE_PATH))

    print(f"Features: {len(feature_names)}")

    if len(feature_names) != 78:

        print(f"WARNING: Expected 78 features, " f"found {len(feature_names)}.")

    return feature_names


# =========================================================
# LOAD FRIDAY DATA
# =========================================================


def load_friday_data():

    print_header("3. LOADING FRIDAY TEST DATA")

    frames = []

    for filename in FRIDAY_FILES:

        matches = list(DATA_DIR.rglob(filename))

        if not matches:

            fail(f"Friday dataset not found under:\n{DATA_DIR}\n{filename}")

        path = matches[0]

        if not path.exists():

            fail(f"Friday dataset not found:\n" f"{path}")

        print(f"\nLoading: {filename}")

        df = pd.read_csv(
            path,
            low_memory=False,
        )

        print(f"Original rows: {len(df):,}")

        frames.append(df)

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    print(f"\nTotal Friday rows: " f"{len(df):,}")

    return df


# =========================================================
# FIND LABEL COLUMN
# =========================================================


def get_label_column(df):

    if " Label" in df.columns:

        return " Label"

    if "Label" in df.columns:

        return "Label"

    fail("Could not find CICIDS label column.")


# =========================================================
# CREATE BINARY LABEL
# =========================================================


def create_binary_labels(df):

    print_header("4. CREATING BINARY LABELS")

    label_column = get_label_column(df)

    print(f"Label column: {label_column}")

    labels = df[label_column].astype(str).str.strip()

    y = labels.str.upper().ne("BENIGN").astype(int).to_numpy()

    print("\nBinary distribution:")

    benign = int(np.sum(y == 0))
    attack = int(np.sum(y == 1))

    print(f"BENIGN: {benign:,}")
    print(f"ATTACK: {attack:,}")

    return labels, y


# =========================================================
# PREPARE FEATURES
# =========================================================


def prepare_features(
    df,
    feature_names,
):

    print_header("5. PREPARING FEATURES")

    print(f"Preparing {len(df):,} Friday samples...")

    cleaned_columns = {}

    for column in df.columns:

        cleaned_columns[str(column).strip()] = column

    missing = []

    X = pd.DataFrame(index=df.index)

    for feature in feature_names:

        clean_feature = str(feature).strip()

        if clean_feature not in cleaned_columns:

            missing.append(clean_feature)

            continue

        original_column = cleaned_columns[clean_feature]

        values = pd.to_numeric(
            df[original_column],
            errors="coerce",
        )

        values = values.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        values = values.fillna(0.0)

        X[feature] = values

    if missing:

        print("\nMissing features:")

        for feature in missing:

            print(f"  - {feature}")

        fail(f"{len(missing)} required features missing.")

    print(f"Prepared features: " f"{X.shape[1]}")

    print(f"Prepared samples: " f"{X.shape[0]:,}")

    return X


# =========================================================
# GENERATE ATTACK PROBABILITIES
# =========================================================


def generate_probabilities(
    model,
    X,
):

    print_header("6. GENERATING ATTACK PROBABILITIES")

    print("Generating probability predictions...")

    probabilities = model.predict_proba(X)

    classes = list(model.classes_)

    print(f"Model classes: {classes}")

    # -----------------------------------------------------
    # Find ATTACK class
    # -----------------------------------------------------

    if 1 in classes:

        attack_index = classes.index(1)

    elif "ATTACK" in classes:

        attack_index = classes.index("ATTACK")

    else:

        fail("Could not find ATTACK class " "in model.classes_.")

    attack_probabilities = probabilities[:, attack_index]

    print("Probability generation complete.")

    print(f"Minimum attack probability: " f"{attack_probabilities.min():.6f}")

    print(f"Maximum attack probability: " f"{attack_probabilities.max():.6f}")

    print(f"Mean attack probability: " f"{attack_probabilities.mean():.6f}")

    print(f"Median attack probability: " f"{np.median(attack_probabilities):.6f}")

    return attack_probabilities


# =========================================================
# PER-TRAFFIC-TYPE PROBABILITY SUMMARY
# =========================================================


def analyze_probability_distribution(
    labels,
    attack_probabilities,
):

    print_header("7. ATTACK PROBABILITY DISTRIBUTION")

    unique_labels = sorted(labels.unique())

    rows = []

    for label in unique_labels:

        mask = labels.to_numpy() == label

        probs = attack_probabilities[mask]

        if len(probs) == 0:

            continue

        rows.append(
            {
                "Traffic Type": label,
                "Samples": len(probs),
                "Min": np.min(probs),
                "P10": np.percentile(probs, 10),
                "P25": np.percentile(probs, 25),
                "Median": np.median(probs),
                "P75": np.percentile(probs, 75),
                "P90": np.percentile(probs, 90),
                "P95": np.percentile(probs, 95),
                "P99": np.percentile(probs, 99),
                "Max": np.max(probs),
                "Mean": np.mean(probs),
            }
        )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return result


# =========================================================
# THRESHOLD METRICS
# =========================================================


def calculate_threshold_metrics(
    y_true,
    attack_probabilities,
    threshold,
):

    y_pred = (attack_probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "true_negative_rate": true_negative_rate,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


# =========================================================
# RUN THRESHOLD ANALYSIS
# =========================================================


def run_threshold_analysis(
    y_true,
    attack_probabilities,
):

    print_header("8. THRESHOLD ANALYSIS")

    rows = []

    for threshold in THRESHOLDS:

        metrics = calculate_threshold_metrics(
            y_true,
            attack_probabilities,
            threshold,
        )

        rows.append(metrics)

    results = pd.DataFrame(rows)

    display = results.copy()

    display["threshold"] = display["threshold"].map(lambda x: f"{x:.2f}")

    display["accuracy"] = display["accuracy"].map(lambda x: f"{x:.4f}")

    display["precision"] = display["precision"].map(lambda x: f"{x:.4f}")

    display["recall"] = display["recall"].map(lambda x: f"{x:.4f}")

    display["f1"] = display["f1"].map(lambda x: f"{x:.4f}")

    display["false_positive_rate"] = display["false_positive_rate"].map(
        lambda x: f"{x:.4f}"
    )

    display["true_negative_rate"] = display["true_negative_rate"].map(
        lambda x: f"{x:.4f}"
    )

    print(display.to_string(index=False))

    return results


# =========================================================
# PER-ATTACK-TYPE RECALL
# =========================================================


def calculate_per_type_recall(
    labels,
    attack_probabilities,
    threshold,
):

    rows = []

    predicted_attack = attack_probabilities >= threshold

    for traffic_type in sorted(labels.unique()):

        mask = labels.to_numpy() == traffic_type

        total = int(np.sum(mask))

        detected = int(np.sum(predicted_attack[mask]))

        missed = total - detected

        recall = detected / total if total > 0 else 0.0

        rows.append(
            {
                "Traffic Type": traffic_type,
                "Samples": total,
                "Detected": detected,
                "Missed": missed,
                "Recall": recall,
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# SHOW PER-TYPE RESULTS FOR KEY THRESHOLDS
# =========================================================


def show_per_type_analysis(
    labels,
    attack_probabilities,
):

    print_header("9. PER-ATTACK-TYPE DETECTION")

    selected_thresholds = [
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]

    for threshold in selected_thresholds:

        print(f"\n" f"{'-' * 78}")

        print(f"THRESHOLD = {threshold:.2f}")

        print(f"{'-' * 78}")

        result = calculate_per_type_recall(
            labels,
            attack_probabilities,
            threshold,
        )

        result["Recall"] = result["Recall"].map(lambda x: f"{x:.2%}")

        print(result.to_string(index=False))


# =========================================================
# FIND BEST THRESHOLDS
# =========================================================


def find_best_thresholds(results):

    print_header("10. BEST THRESHOLD CANDIDATES")

    # -----------------------------------------------------
    # Best F1
    # -----------------------------------------------------

    best_f1 = results.loc[results["f1"].idxmax()]

    print("Best F1 threshold:")

    print(f"  Threshold: " f"{best_f1['threshold']:.2f}")

    print(f"  Precision: " f"{best_f1['precision']:.4f}")

    print(f"  Recall: " f"{best_f1['recall']:.4f}")

    print(f"  F1: " f"{best_f1['f1']:.4f}")

    print(f"  Accuracy: " f"{best_f1['accuracy']:.4f}")

    # -----------------------------------------------------
    # Best recall with FPR <= 5%
    # -----------------------------------------------------

    low_fpr = results[results["false_positive_rate"] <= 0.05]

    if len(low_fpr) > 0:

        best_low_fpr = low_fpr.loc[low_fpr["recall"].idxmax()]

        print("\nBest recall with " "false-positive rate <= 5%:")

        print(f"  Threshold: " f"{best_low_fpr['threshold']:.2f}")

        print(f"  Precision: " f"{best_low_fpr['precision']:.4f}")

        print(f"  Recall: " f"{best_low_fpr['recall']:.4f}")

        print(f"  F1: " f"{best_low_fpr['f1']:.4f}")

        print(f"  False Positive Rate: " f"{best_low_fpr['false_positive_rate']:.4f}")

    else:

        print("\nNo threshold achieved " "FPR <= 5%.")

    # -----------------------------------------------------
    # Best recall with FPR <= 1%
    # -----------------------------------------------------

    very_low_fpr = results[results["false_positive_rate"] <= 0.01]

    if len(very_low_fpr) > 0:

        best_very_low_fpr = very_low_fpr.loc[very_low_fpr["recall"].idxmax()]

        print("\nBest recall with " "false-positive rate <= 1%:")

        print(f"  Threshold: " f"{best_very_low_fpr['threshold']:.2f}")

        print(f"  Precision: " f"{best_very_low_fpr['precision']:.4f}")

        print(f"  Recall: " f"{best_very_low_fpr['recall']:.4f}")

        print(f"  F1: " f"{best_very_low_fpr['f1']:.4f}")

        print(
            f"  False Positive Rate: " f"{best_very_low_fpr['false_positive_rate']:.4f}"
        )

    else:

        print("\nNo threshold achieved " "FPR <= 1%.")


# =========================================================
# SAVE RESULTS
# =========================================================


def save_results(
    results,
    probability_distribution,
):

    print_header("11. SAVING ANALYSIS RESULTS")

    output_dir = BASE_DIR / "data" / "processed"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_path = output_dir / "friday_threshold_analysis.csv"

    distribution_path = output_dir / "friday_probability_distribution.csv"

    results.to_csv(
        threshold_path,
        index=False,
    )

    probability_distribution.to_csv(
        distribution_path,
        index=False,
    )

    print(f"Threshold results:\n" f"{threshold_path}")

    print(f"\nProbability distribution:\n" f"{distribution_path}")


# =========================================================
# FINAL INTERPRETATION
# =========================================================


def final_interpretation(
    results,
    labels,
    attack_probabilities,
):

    print_header("12. NETSENTRY THRESHOLD ANALYSIS")

    current = results[
        np.isclose(
            results["threshold"],
            0.50,
        )
    ].iloc[0]

    best = results.loc[results["f1"].idxmax()]

    print("Current threshold: 0.50")

    print(f"  Accuracy:  " f"{current['accuracy']:.4f}")

    print(f"  Precision: " f"{current['precision']:.4f}")

    print(f"  Recall:    " f"{current['recall']:.4f}")

    print(f"  F1:        " f"{current['f1']:.4f}")

    print("\nBest F1 threshold:")

    print(f"  Threshold: " f"{best['threshold']:.2f}")

    print(f"  Accuracy:  " f"{best['accuracy']:.4f}")

    print(f"  Precision: " f"{best['precision']:.4f}")

    print(f"  Recall:    " f"{best['recall']:.4f}")

    print(f"  F1:        " f"{best['f1']:.4f}")

    # -----------------------------------------------------
    # Determine whether threshold is likely the issue
    # -----------------------------------------------------

    attack_mask = labels.astype(str).str.upper().ne("BENIGN").to_numpy()

    attack_probs = attack_probabilities[attack_mask]

    benign_mask = ~attack_mask

    benign_probs = attack_probabilities[benign_mask]

    attack_median = np.median(attack_probs)

    benign_median = np.median(benign_probs)

    print("\nProbability separation:")

    print(f"  Median attack probability: " f"{attack_median:.4f}")

    print(f"  Median benign probability: " f"{benign_median:.4f}")

    if attack_median < 0.50:

        print("\nCONCLUSION:")

        print(
            "The majority of unseen Friday attacks "
            "receive an attack probability below 0.50."
        )

        print(
            "This suggests the problem is deeper "
            "than simply choosing a 0.50 threshold."
        )

    else:

        print("\nCONCLUSION:")

        print(
            "A substantial portion of unseen attacks "
            "receive attack probabilities above 0.50."
        )

        print("Threshold tuning may significantly improve " "Friday attack detection.")


# =========================================================
# MAIN
# =========================================================


def main():

    print_header("NETSENTRY - BINARY THRESHOLD ANALYSIS")

    print("Purpose:")

    print(
        "Determine whether changing the binary "
        "ATTACK threshold can improve unseen-day "
        "Friday attack detection."
    )

    print("\nIMPORTANT:")

    print("Friday data is evaluation-only.")

    print("No retraining occurs in this script.")

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = load_model()

    # -----------------------------------------------------
    # Load feature names
    # -----------------------------------------------------

    feature_names = load_features()

    # -----------------------------------------------------
    # Load Friday
    # -----------------------------------------------------

    df = load_friday_data()

    # -----------------------------------------------------
    # Labels
    # -----------------------------------------------------

    labels, y_true = create_binary_labels(df)

    # -----------------------------------------------------
    # Features
    # -----------------------------------------------------

    X = prepare_features(
        df,
        feature_names,
    )

    # -----------------------------------------------------
    # Probabilities
    # -----------------------------------------------------

    attack_probabilities = generate_probabilities(
        model,
        X,
    )

    # -----------------------------------------------------
    # Distribution
    # -----------------------------------------------------

    probability_distribution = analyze_probability_distribution(
        labels,
        attack_probabilities,
    )

    # -----------------------------------------------------
    # Threshold analysis
    # -----------------------------------------------------

    results = run_threshold_analysis(
        y_true,
        attack_probabilities,
    )

    # -----------------------------------------------------
    # Per-type detection
    # -----------------------------------------------------

    show_per_type_analysis(
        labels,
        attack_probabilities,
    )

    # -----------------------------------------------------
    # Best thresholds
    # -----------------------------------------------------

    find_best_thresholds(results)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_results(
        results,
        probability_distribution,
    )

    # -----------------------------------------------------
    # Final interpretation
    # -----------------------------------------------------

    final_interpretation(
        results,
        labels,
        attack_probabilities,
    )

    print_header("THRESHOLD ANALYSIS COMPLETE")


if __name__ == "__main__":

    main()
