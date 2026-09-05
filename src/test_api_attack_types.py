from pathlib import Path

import joblib
import pandas as pd
import requests


# =========================================================
# NETSENTRY - MULTI-ATTACK API VALIDATION
#
# DEPRECATED: loads feature_names_unseen_day.joblib, which is not
# confirmed identical to the production schema
# (feature_names_production.joblib). Use test_api_final.py or
# validate_production.py instead -- both verify against the actual
# 70-feature production schema the API serves.
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"

FEATURE_PATH = (
    BASE_DIR
    / "models"
    / "feature_names_unseen_day.joblib"
)

API_URL = "http://127.0.0.1:8085/api/predict"


# =========================================================
# FRIDAY FILES
# =========================================================

FRIDAY_FILES = [
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


# =========================================================
# FIND FILES
# =========================================================

def find_friday_files():

    print("\nSearching for Friday datasets...")

    found_files = []

    for filename in FRIDAY_FILES:

        matches = list(
            RAW_DIR.rglob(filename)
        )

        if matches:

            found_files.append(
                matches[0]
            )

            print(
                f"  FOUND: {matches[0].name}"
            )

        else:

            print(
                f"  MISSING: {filename}"
            )

    if not found_files:

        raise FileNotFoundError(
            "No Friday CSV files were found."
        )

    return found_files


# =========================================================
# LOAD DATA
# =========================================================

def load_friday_data(files):

    print("\nLoading Friday traffic...")

    frames = []

    for file in files:

        print(
            f"\nLoading: {file.name}"
        )

        df = pd.read_csv(file)

        print(
            f"Rows: {len(df):,}"
        )

        # Normalize column names
        df.columns = (
            df.columns
            .str.strip()
        )

        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    print(
        f"\nCombined rows: "
        f"{len(combined):,}"
    )

    return combined


# =========================================================
# PREPARE FEATURES
# =========================================================

def prepare_features(row, feature_names):

    # Remove labels
    X = row.drop(
        labels=[
            "Label",
            "target",
            "attack_type",
        ],
        errors="ignore",
    )

    # Make sure all expected features exist
    missing = [
        feature
        for feature in feature_names
        if feature not in X.index
    ]

    if missing:

        raise ValueError(
            "Missing features: "
            + ", ".join(missing)
        )

    # Exact model feature order
    X = X[
        feature_names
    ]

    # Convert to numeric
    X = pd.to_numeric(
        X,
        errors="coerce"
    )

    if X.isna().any():

        bad = X.index[
            X.isna()
        ].tolist()

        raise ValueError(
            "Invalid values in features: "
            + ", ".join(bad)
        )

    # JSON-compatible dictionary
    return {
        feature: float(X[feature])
        for feature in feature_names
    }


# =========================================================
# SEND PREDICTION
# =========================================================

def predict(features):

    payload = {
        "features": features
    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=30
        )

    except requests.exceptions.ConnectionError:

        raise RuntimeError(
            "\nCould not connect to NETSENTRY API.\n"
            "Make sure this is running in another terminal:\n\n"
            "python src\\app.py"
        )

    except requests.exceptions.Timeout:

        raise RuntimeError(
            "NETSENTRY API request timed out."
        )

    if response.status_code != 200:

        try:
            error = response.json()
        except ValueError:
            error = response.text

        raise RuntimeError(
            f"API returned HTTP "
            f"{response.status_code}: "
            f"{error}"
        )

    return response.json()


# =========================================================
# EXPECTED BINARY LABEL
# =========================================================

def expected_prediction(attack_type):

    if attack_type == "BENIGN":

        return "BENIGN"

    return "ATTACK"


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("NETSENTRY - MULTI-ATTACK API VALIDATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Load feature names
    # -----------------------------------------------------

    print(
        "\nLoading model feature names..."
    )

    if not FEATURE_PATH.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n"
            f"{FEATURE_PATH}"
        )

    feature_names = joblib.load(
        FEATURE_PATH
    )

    print(
        f"Model expects "
        f"{len(feature_names)} features."
    )

    # -----------------------------------------------------
    # Find datasets
    # -----------------------------------------------------

    files = find_friday_files()

    # -----------------------------------------------------
    # Load datasets
    # -----------------------------------------------------

    df = load_friday_data(files)

    # -----------------------------------------------------
    # Check label column
    # -----------------------------------------------------

    if "Label" not in df.columns:

        raise ValueError(
            "The Friday dataset does not contain "
            "a 'Label' column."
        )

    # Normalize labels
    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    # -----------------------------------------------------
    # Show available attack types
    # -----------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "AVAILABLE TRAFFIC TYPES"
    )

    print(
        "=" * 70
    )

    distribution = (
        df["Label"]
        .value_counts()
    )

    for label, count in distribution.items():

        print(
            f"{label:<35} {count:>10,}"
        )

    # -----------------------------------------------------
    # Target classes
    # -----------------------------------------------------

    target_types = [
        "BENIGN",
        "DDoS",
        "PortScan",
        "Bot",
    ]

    # -----------------------------------------------------
    # Find one sample per class
    # -----------------------------------------------------

    samples = {}

    print(
        "\n" + "=" * 70
    )

    print(
        "SELECTING TEST RECORDS"
    )

    print(
        "=" * 70
    )

    for attack_type in target_types:

        matches = df[
            df["Label"]
            .str.lower()
            == attack_type.lower()
        ]

        if len(matches) == 0:

            print(
                f"\nWARNING: "
                f"{attack_type} not found."
            )

            continue

        # Take the first real dataset record
        samples[attack_type] = (
            matches.iloc[0]
        )

        print(
            f"\n{attack_type}:"
        )

        print(
            f"  Samples available: "
            f"{len(matches):,}"
        )

        print(
            "  Selected: 1"
        )

    if not samples:

        raise RuntimeError(
            "No target traffic samples were found."
        )

    # -----------------------------------------------------
    # Test API
    # -----------------------------------------------------

    results = []

    print(
        "\n" + "=" * 70
    )

    print(
        "RUNNING API PREDICTIONS"
    )

    print(
        "=" * 70
    )

    for attack_type in target_types:

        if attack_type not in samples:

            continue

        row = samples[
            attack_type
        ]

        print(
            "\n" + "-" * 70
        )

        print(
            f"TESTING: {attack_type}"
        )

        print(
            "-" * 70
        )

        print(
            f"Ground truth: "
            f"{attack_type}"
        )

        expected = expected_prediction(
            attack_type
        )

        print(
            f"Expected API class: "
            f"{expected}"
        )

        # Prepare features
        features = prepare_features(
            row,
            feature_names
        )

        print(
            f"Features sent: "
            f"{len(features)}"
        )

        # Send API request
        result = predict(
            features
        )

        prediction = result.get(
            "prediction"
        )

        confidence = result.get(
            "confidence_percent"
        )

        risk = result.get(
            "risk_level"
        )

        probabilities = result.get(
            "probabilities",
            {}
        )

        correct = (
            prediction
            == expected
        )

        print(
            f"\nPrediction: "
            f"{prediction}"
        )

        print(
            f"Confidence: "
            f"{confidence}%"
        )

        print(
            f"Risk level: "
            f"{risk}"
        )

        print(
            f"Benign probability: "
            f"{probabilities.get('benign')}"
        )

        print(
            f"Attack probability: "
            f"{probabilities.get('attack')}"
        )

        if correct:

            print(
                "\nResult: CORRECT"
            )

        else:

            print(
                "\nResult: INCORRECT"
            )

        results.append(
            {
                "attack_type": attack_type,
                "expected": expected,
                "prediction": prediction,
                "confidence": confidence,
                "risk": risk,
                "correct": correct,
            }
        )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print(
        "\n\n" + "=" * 70
    )

    print(
        "API VALIDATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"{'Traffic Type':<18}"
        f"{'Expected':<12}"
        f"{'Prediction':<12}"
        f"{'Confidence':<12}"
        f"{'Result'}"
    )

    print(
        "-" * 70
    )

    for result in results:

        status = (
            "PASS"
            if result["correct"]
            else "FAIL"
        )

        print(
            f"{result['attack_type']:<18}"
            f"{result['expected']:<12}"
            f"{result['prediction']:<12}"
            f"{str(result['confidence']) + '%':<12}"
            f"{status}"
        )

    # -----------------------------------------------------
    # Overall API validation
    # -----------------------------------------------------

    passed = sum(
        1
        for result in results
        if result["correct"]
    )

    total = len(results)

    print(
        "\n" + "-" * 70
    )

    print(
        f"API TESTS PASSED: "
        f"{passed}/{total}"
    )

    if total > 0:

        accuracy = (
            passed / total
        ) * 100

        print(
            f"Validation accuracy: "
            f"{accuracy:.2f}%"
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "NETSENTRY API VALIDATION COMPLETE"
    )

    print(
        "=" * 70
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()