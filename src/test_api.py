from pathlib import Path

import joblib
import pandas as pd
import requests

# =========================================================
# NETSENTRY - API VALIDATION CLIENT
#
# DEPRECATED: loads feature_names_multiclass.joblib, which is not
# confirmed identical to the production schema
# (feature_names_production.joblib). Use test_api_final.py or
# validate_production.py instead -- both verify against the actual
# 70-feature production schema the API serves.
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"

FEATURE_PATH = BASE_DIR / "models" / "feature_names_multiclass.joblib"

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
# FIND FILE
# =========================================================


def find_file(filename):

    matches = list(RAW_DIR.rglob(filename))

    if not matches:

        raise FileNotFoundError(f"Could not find:\n{filename}")

    return matches[0]


# =========================================================
# NORMALIZE LABEL
# =========================================================


def normalize_label(label):

    label = str(label).strip()

    if label.upper() == "BENIGN":
        return "BENIGN"

    replacements = {
        "Web Attack � Brute Force": "Web Attack - Brute Force",
        "Web Attack � XSS": "Web Attack - XSS",
        "Web Attack � Sql Injection": "Web Attack - Sql Injection",
    }

    return replacements.get(label, label)


# =========================================================
# PREPARE FEATURES
# =========================================================


def prepare_features(row, feature_names):

    X = row.drop(
        labels=[
            "Label",
            "target",
            "attack_type",
        ],
        errors="ignore",
    )

    X = X[feature_names]

    X = pd.to_numeric(X, errors="coerce")

    X = X.fillna(0)

    return {feature: float(X[feature]) for feature in feature_names}


# =========================================================
# MAIN
# =========================================================


def main():

    print("=" * 70)
    print("NETSENTRY - MULTICLASS API VALIDATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Load features
    # -----------------------------------------------------

    print("\nLoading model feature names...")

    feature_names = joblib.load(FEATURE_PATH)

    print(f"Model expects " f"{len(feature_names)} features.")

    # -----------------------------------------------------
    # Load Friday datasets
    # -----------------------------------------------------

    frames = []

    print("\nLoading Friday traffic...")

    for filename in FRIDAY_FILES:

        file = find_file(filename)

        print(f"  Loading: " f"{file.name}")

        df = pd.read_csv(file, low_memory=False)

        df.columns = df.columns.astype(str).str.strip()

        df["attack_type"] = df["Label"].apply(normalize_label)

        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    print(f"\nCombined rows: " f"{len(df):,}")

    # -----------------------------------------------------
    # Desired test classes
    # -----------------------------------------------------

    target_types = [
        "BENIGN",
        "DDoS",
        "PortScan",
        "Bot",
    ]

    # -----------------------------------------------------
    # Select records
    # -----------------------------------------------------

    print("\n" + "=" * 70)

    print("SELECTING TEST RECORDS")

    print("=" * 70)

    samples = {}

    for attack_type in target_types:

        matches = df[df["attack_type"].str.lower() == attack_type.lower()]

        if len(matches) == 0:

            print(f"\nWARNING: " f"{attack_type} not found.")

            continue

        samples[attack_type] = matches.iloc[0]

        print(f"\n{attack_type}: " f"{len(matches):,} available")

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    results = []

    print("\n" + "=" * 70)

    print("RUNNING API TESTS")

    print("=" * 70)

    for attack_type in target_types:

        if attack_type not in samples:
            continue

        row = samples[attack_type]

        print("\n" + "-" * 70)

        print(f"GROUND TRUTH: " f"{attack_type}")

        print("-" * 70)

        features = prepare_features(row, feature_names)

        try:

            response = requests.post(API_URL, json={"features": features}, timeout=30)

        except Exception as error:

            print(f"API CONNECTION ERROR: " f"{error}")

            return

        print(f"HTTP Status: " f"{response.status_code}")

        if response.status_code != 200:

            print("API ERROR:")

            print(response.text)

            continue

        result = response.json()

        prediction = result["prediction"]

        confidence = result["confidence_percent"]

        risk = result["risk_level"]

        is_attack = result["is_attack"]

        # Expected
        expected = "BENIGN" if attack_type == "BENIGN" else attack_type

        correct = prediction == expected

        print(f"\nPrediction: " f"{prediction}")

        print(f"Confidence: " f"{confidence}%")

        print(f"Attack: " f"{is_attack}")

        print(f"Risk: " f"{risk}")

        print("\nProbabilities:")

        for label, probability in result["probabilities"].items():

            print(f"  {label:<30}" f"{probability * 100:>7.2f}%")

        if correct:

            print("\nResult: PASS")

        else:

            print("\nResult: FAIL")

        results.append(
            {
                "ground_truth": attack_type,
                "prediction": prediction,
                "confidence": confidence,
                "risk": risk,
                "correct": correct,
            }
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n\n" + "=" * 70)

    print("API VALIDATION SUMMARY")

    print("=" * 70)

    print()

    print(f"{'Traffic':<20}" f"{'Prediction':<25}" f"{'Confidence':<12}" f"Result")

    print("-" * 70)

    for result in results:

        status = "PASS" if result["correct"] else "FAIL"

        print(
            f"{result['ground_truth']:<20}"
            f"{result['prediction']:<25}"
            f"{str(result['confidence']) + '%':<12}"
            f"{status}"
        )

    passed = sum(result["correct"] for result in results)

    total = len(results)

    print("\n" + "-" * 70)

    print(f"Tests passed: " f"{passed}/{total}")

    if total:

        print(f"API sample accuracy: " f"{passed / total:.2%}")

    print("\n" + "=" * 70)

    print("NETSENTRY API VALIDATION COMPLETE")

    print("=" * 70)


if __name__ == "__main__":
    main()
