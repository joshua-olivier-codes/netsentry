from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import requests

# =========================================================
# NETSENTRY - FINAL API VALIDATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

API_URL = "http://127.0.0.1:8085"

BINARY_FEATURE_PATH = BASE_DIR / "models" / "feature_names_production.joblib"

MULTICLASS_FEATURE_PATH = BASE_DIR / "models" / "feature_names_multiclass.joblib"

DATA_DIR = BASE_DIR / "data" / "raw" / "MachineLearningCVE"


# =========================================================
# FRIDAY DATASETS
# =========================================================

FRIDAY_FILES = [
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


# =========================================================
# HELPERS
# =========================================================


def print_header(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def fail(message):

    print(f"\nERROR: {message}")

    sys.exit(1)


# =========================================================
# API GET
# =========================================================


def call_get(endpoint):

    try:

        response = requests.get(
            API_URL + endpoint,
            timeout=15,
        )

        try:
            body = response.json()

        except Exception:
            body = response.text

        return response.status_code, body

    except requests.exceptions.ConnectionError:

        fail(
            "Could not connect to NETSENTRY API.\n"
            f"Start it with:\n"
            f"python src\\app.py"
        )

    except requests.exceptions.Timeout:

        fail("API request timed out.")

    except Exception as error:

        fail(f"GET request failed: {error}")


# =========================================================
# API POST
# =========================================================


def call_api(features):

    try:

        response = requests.post(
            f"{API_URL}/api/predict",
            json={"features": features},
            timeout=30,
        )

        try:

            body = response.json()

        except Exception:

            body = response.text

        return response.status_code, body

    except requests.exceptions.ConnectionError:

        fail(
            "Could not connect to NETSENTRY API.\n"
            f"Make sure the API is running on {API_URL}"
        )

    except requests.exceptions.Timeout:

        fail("API request timed out.")

    except Exception as error:

        fail(f"API request failed: {error}")


# =========================================================
# HEALTH CHECK
# =========================================================


def test_health():

    print_header("1. HEALTH CHECK")

    status, body = call_get("/health")

    print(f"HTTP Status: {status}")

    print(
        json.dumps(
            body,
            indent=2,
        )
    )

    if status != 200:

        fail("Health endpoint failed.")

    if not isinstance(body, dict):

        fail("Health response is not JSON.")

    if body.get("status") != "healthy":

        fail("API did not report healthy status.")

    print("\nPASS - API is healthy.")


# =========================================================
# MODEL INFORMATION
# =========================================================


def test_model_info():

    print_header("2. MODEL INFORMATION")

    status, body = call_get("/api/model")

    print(f"HTTP Status: {status}")

    if status != 200:

        fail("/api/model failed.")

    print(
        json.dumps(
            body,
            indent=2,
        )
    )

    # -----------------------------------------------------
    # Validate architecture
    # -----------------------------------------------------

    binary_info = body.get(
        "binary_model",
        {},
    )

    multiclass_info = body.get(
        "multiclass_model",
        {},
    )

    binary_count = binary_info.get("feature_count")

    multiclass_count = multiclass_info.get("feature_count")

    print("\nArchitecture validation:")

    print(f"  Binary features:     {binary_count}")

    print(f"  Multiclass features: {multiclass_count}")

    if binary_count != 70:

        fail("API binary feature count should be 70.")

    if multiclass_count != 70:

        fail("API multiclass feature count should be 70.")

    print("\nPASS - Model information endpoint works.")


# =========================================================
# LOAD FEATURES
# =========================================================


def load_feature_names():

    print_header("3. LOADING MODEL FEATURES")

    # -----------------------------------------------------
    # Binary features
    # -----------------------------------------------------

    if not BINARY_FEATURE_PATH.exists():

        fail("Binary feature file not found:\n" f"{BINARY_FEATURE_PATH}")

    binary_feature_names = list(joblib.load(BINARY_FEATURE_PATH))

    # -----------------------------------------------------
    # Multiclass features
    # -----------------------------------------------------

    if not MULTICLASS_FEATURE_PATH.exists():

        fail("Multiclass feature file not found:\n" f"{MULTICLASS_FEATURE_PATH}")

    multiclass_feature_names = list(joblib.load(MULTICLASS_FEATURE_PATH))

    # -----------------------------------------------------
    # Print counts
    # -----------------------------------------------------

    print(f"Binary features:     " f"{len(binary_feature_names)}")

    print(f"Multiclass features: " f"{len(multiclass_feature_names)}")

    # -----------------------------------------------------
    # Validate expected architecture
    # -----------------------------------------------------

    if len(binary_feature_names) != 70:

        fail(
            "Expected exactly 70 binary features.\n"
            f"Found: {len(binary_feature_names)}"
        )

    if len(multiclass_feature_names) != 70:

        fail(
            "Expected exactly 70 multiclass features.\n"
            f"Found: {len(multiclass_feature_names)}"
        )

    # -----------------------------------------------------
    # Verify every multiclass feature exists in binary
    # -----------------------------------------------------

    missing_from_binary = [
        feature
        for feature in multiclass_feature_names
        if feature not in binary_feature_names
    ]

    if missing_from_binary:

        fail(
            "Multiclass features missing from binary "
            "feature set:\n" + "\n".join(missing_from_binary)
        )

    # -----------------------------------------------------
    # Determine binary-only features
    # -----------------------------------------------------

    binary_only_features = [
        feature
        for feature in binary_feature_names
        if feature not in multiclass_feature_names
    ]

    print(f"Binary-only features: " f"{len(binary_only_features)}")

    if len(binary_only_features) != 0:

        fail(
            "Expected exactly 0 binary-only features.\n"
            f"Found: {len(binary_only_features)}"
        )

    print("\nBinary-only features:")

    for feature in binary_only_features:

        print(f"  - {feature}")

    print("\nPASS - Feature architecture validated.")

    return (
        binary_feature_names,
        multiclass_feature_names,
    )


# =========================================================
# LOAD FRIDAY DATA
# =========================================================


def load_friday_data():

    print_header("4. LOADING FRIDAY TEST DATA")

    frames = []

    for filename in FRIDAY_FILES:

        path = DATA_DIR / filename

        if not path.exists():

            print(f"WARNING: Missing:\n" f"  {path}")

            continue

        print(f"\nLoading: {filename}")

        df = pd.read_csv(
            path,
            low_memory=False,
        )

        print(f"Rows: {len(df):,}")

        frames.append(df)

    if not frames:

        fail("No Friday datasets were found.")

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    print(f"\nCombined rows: " f"{len(df):,}")

    # -----------------------------------------------------
    # Find label column
    # -----------------------------------------------------

    if " Label" in df.columns:

        label_column = " Label"

    elif "Label" in df.columns:

        label_column = "Label"

    else:

        fail("Could not find dataset label column.")

    # -----------------------------------------------------
    # Normalize labels
    # -----------------------------------------------------

    df["attack_type"] = df[label_column].astype(str).str.strip()

    print("\nAvailable Friday traffic:")

    print(df["attack_type"].value_counts().to_string())

    return df


# =========================================================
# FEATURE EXTRACTION
# =========================================================


def prepare_record(
    row,
    feature_names,
):
    """
    Converts one CICIDS2017 network-flow row into
    the exact feature format expected by the model.

    The API binary stage requires 70 features.

    The API multiclass stage requires 70 features.

    This function can therefore be called with either
    feature list.
    """

    # -----------------------------------------------------
    # Normalize dataset column names
    # -----------------------------------------------------

    cleaned = {}

    for key, value in row.items():

        clean_key = str(key).strip()

        cleaned[clean_key] = value

    # -----------------------------------------------------
    # Extract features
    # -----------------------------------------------------

    values = {}

    missing = []

    for feature in feature_names:

        clean_feature = str(feature).strip()

        if clean_feature not in cleaned:

            missing.append(clean_feature)

            continue

        value = cleaned[clean_feature]

        # -------------------------------------------------
        # Convert numeric
        # -------------------------------------------------

        try:

            value = float(value)

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                f"Feature '{clean_feature}' "
                f"contains a non-numeric value: "
                f"{value!r}"
            )

        # -------------------------------------------------
        # Validate finite
        # -------------------------------------------------

        if not np.isfinite(value):

            raise ValueError(f"Feature '{clean_feature}' " f"is not finite.")

        values[clean_feature] = value

    # -----------------------------------------------------
    # Missing feature check
    # -----------------------------------------------------

    if missing:

        raise ValueError("Missing features:\n" + "\n".join(missing))

    return values


# =========================================================
# SELECT REPRESENTATIVE RECORDS
# =========================================================


def select_records(df):

    print_header("5. SELECTING REPRESENTATIVE TRAFFIC")

    selected = {}

    traffic_types = [
        "BENIGN",
        "DDoS",
        "PortScan",
        "Bot",
    ]

    for traffic_type in traffic_types:

        subset = df[df["attack_type"] == traffic_type]

        if len(subset) == 0:

            print(f"{traffic_type:<12} " f"NOT FOUND")

            continue

        # -------------------------------------------------
        # Deterministic middle record
        # -------------------------------------------------

        record = subset.iloc[len(subset) // 2]

        selected[traffic_type] = record

        print(f"{traffic_type:<12} " f"{len(subset):,} available")

    return selected


# =========================================================
# VALIDATE TRAFFIC TYPES
# =========================================================


def validate_attack_types(
    records,
    binary_feature_names,
):

    print_header("6. ATTACK-TYPE API VALIDATION")

    results = []

    for traffic_type, row in records.items():

        print("\n" + "-" * 70)

        print(f"TESTING: {traffic_type}")

        # -------------------------------------------------
        # Prepare FULL 70-feature binary input
        # -------------------------------------------------

        try:

            features = prepare_record(
                row,
                binary_feature_names,
            )

        except Exception as error:

            print(f"Feature preparation failed: " f"{error}")

            results.append(
                {
                    "traffic_type": traffic_type,
                    "expected": ("BENIGN" if traffic_type == "BENIGN" else "ATTACK"),
                    "prediction": "ERROR",
                    "confidence": 0.0,
                    "attack_type": "ERROR",
                    "result": "FAIL",
                }
            )

            continue

        # -------------------------------------------------
        # Expected binary class
        # -------------------------------------------------

        expected = "BENIGN" if traffic_type == "BENIGN" else "ATTACK"

        print(f"Ground truth: " f"{traffic_type}")

        print(f"Expected API class: " f"{expected}")

        print(f"Features sent: " f"{len(features)}")

        # -------------------------------------------------
        # Verify we really have 70 features
        # -------------------------------------------------

        if len(features) != 70:

            print("\nFAIL - Expected " "70 features.")

            results.append(
                {
                    "traffic_type": traffic_type,
                    "expected": expected,
                    "prediction": "ERROR",
                    "confidence": 0.0,
                    "attack_type": "ERROR",
                    "result": "FAIL",
                }
            )

            continue

        # -------------------------------------------------
        # API request
        # -------------------------------------------------

        status, body = call_api(features)

        print(f"HTTP Status: {status}")

        if status != 200:

            print(
                json.dumps(
                    body,
                    indent=2,
                )
            )

            results.append(
                {
                    "traffic_type": traffic_type,
                    "expected": expected,
                    "prediction": "ERROR",
                    "confidence": 0.0,
                    "attack_type": "ERROR",
                    "result": "FAIL",
                }
            )

            continue

        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        prediction = body.get("prediction")

        confidence = body.get(
            "confidence_percent",
            0.0,
        )

        attack_type = body.get("attack_type")

        attack_type_prediction = body.get("attack_type_prediction")

        attack_type_confidence = body.get("attack_type_confidence")

        risk = body.get("risk_level")

        is_attack = body.get("is_attack")

        # -------------------------------------------------
        # Print results
        # -------------------------------------------------

        print(f"\nPrediction: " f"{prediction}")

        print(f"Is attack: " f"{is_attack}")

        print(f"Attack type: " f"{attack_type}")

        print(f"Attack type prediction: " f"{attack_type_prediction}")

        print(f"Binary confidence: " f"{confidence}%")

        if attack_type_confidence is not None:

            print(
                f"Attack type confidence: "
                f"{round(float(attack_type_confidence) * 100, 2)}%"
            )

        print(f"Risk level: " f"{risk}")

        # -------------------------------------------------
        # Binary correctness
        # -------------------------------------------------

        correct = prediction == expected

        result = "PASS" if correct else "FAIL"

        print(f"\nResult: {result}")

        results.append(
            {
                "traffic_type": traffic_type,
                "expected": expected,
                "prediction": prediction,
                "confidence": confidence,
                "attack_type": attack_type,
                "risk": risk,
                "result": result,
            }
        )

    return results


# =========================================================
# ERROR HANDLING TESTS
# =========================================================


def validate_bad_requests():

    print_header("7. API ERROR-HANDLING TESTS")

    tests = [
        (
            "Missing features",
            {},
        ),
        (
            "Features is not an object",
            {"features": []},
        ),
        (
            "Missing feature",
            {"features": {"Destination Port": 80}},
        ),
    ]

    passed = 0

    for name, payload in tests:

        print(f"\nTesting: {name}")

        try:

            response = requests.post(
                f"{API_URL}/api/predict",
                json=payload,
                timeout=15,
            )

            print(f"HTTP Status: " f"{response.status_code}")

            try:

                body = response.json()

                print(
                    json.dumps(
                        body,
                        indent=2,
                    )
                )

            except Exception:

                body = None

            if response.status_code == 400:

                print("PASS")

                passed += 1

            else:

                print("FAIL - expected " "HTTP 400")

        except Exception as error:

            print(f"FAIL: {error}")

    print(f"\nError handling: " f"{passed}/{len(tests)} passed")

    return passed == len(tests)


# =========================================================
# SUMMARY
# =========================================================


def print_summary(
    results,
    error_tests_passed,
):

    print_header("NETSENTRY FINAL VALIDATION SUMMARY")

    print(
        f"{'Traffic Type':<15}"
        f"{'Expected':<12}"
        f"{'Prediction':<12}"
        f"{'Confidence':<12}"
        f"{'Attack Type':<25}"
        f"Result"
    )

    print("-" * 90)

    for result in results:

        print(
            f"{result['traffic_type']:<15}"
            f"{result['expected']:<12}"
            f"{result['prediction']:<12}"
            f"{str(result['confidence']) + '%':<12}"
            f"{str(result['attack_type']):<25}"
            f"{result['result']}"
        )

    passed = sum(1 for result in results if result["result"] == "PASS")

    total = len(results)

    print("\n" + "-" * 90)

    print(f"Traffic tests: " f"{passed}/{total}")

    print("Error handling: " + ("PASS" if error_tests_passed else "FAIL"))

    if total:

        print(f"Traffic validation accuracy: " f"{(passed / total) * 100:.2f}%")

    print("\n" + "=" * 70)

    if passed == total and error_tests_passed:

        print("NETSENTRY API VALIDATION: PASS")

    else:

        print("NETSENTRY API VALIDATION: PARTIAL")

    print("=" * 70)


# =========================================================
# MAIN
# =========================================================


def main():

    print_header("NETSENTRY - FINAL API VALIDATION")

    print(f"API: {API_URL}")

    print("\nRequired architecture:")

    print("Binary model -> " "BENIGN / ATTACK")

    print("Multiclass model -> " "attack type")

    print("\nFeature architecture:")

    print("Binary model:     70 features")

    print("Multiclass model: 70 features")

    # -----------------------------------------------------
    # Health
    # -----------------------------------------------------

    test_health()

    # -----------------------------------------------------
    # Model information
    # -----------------------------------------------------

    test_model_info()

    # -----------------------------------------------------
    # Features
    # -----------------------------------------------------

    (
        binary_feature_names,
        multiclass_feature_names,
    ) = load_feature_names()

    # -----------------------------------------------------
    # Friday data
    # -----------------------------------------------------

    df = load_friday_data()

    # -----------------------------------------------------
    # Representative records
    # -----------------------------------------------------

    records = select_records(df)

    # -----------------------------------------------------
    # Attack type API tests
    #
    # IMPORTANT:
    # Send the full 70 binary features.
    # -----------------------------------------------------

    results = validate_attack_types(
        records,
        binary_feature_names,
    )

    # -----------------------------------------------------
    # Error handling
    # -----------------------------------------------------

    error_tests_passed = validate_bad_requests()

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print_summary(
        results,
        error_tests_passed,
    )


# =========================================================
# ENTRY POINT
# =========================================================


if __name__ == "__main__":

    main()
