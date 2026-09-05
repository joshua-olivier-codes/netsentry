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

FEATURE_PATH = (
    BASE_DIR
    / "models"
    / "feature_names_multiclass.joblib"
)

DATA_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "MachineLearningCVE"
)


FRIDAY_FILES = [

    "Friday-WorkingHours-Morning.pcap_ISCX.csv",

    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",

    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


# Number of samples used for the API smoke test
SAMPLES_PER_CLASS = 10


# =========================================================
# HELPERS
# =========================================================

def print_header(title):

    print(
        "\n"
        + "=" * 70
    )

    print(title)

    print(
        "=" * 70
    )


def fail(message):

    print(
        f"\nERROR: {message}"
    )

    sys.exit(1)


# =========================================================
# GET REQUEST
# =========================================================

def call_get(endpoint):

    try:

        response = requests.get(
            API_URL + endpoint,
            timeout=15
        )


        try:

            body = response.json()

        except Exception:

            body = response.text


        return (
            response.status_code,
            body
        )


    except requests.exceptions.ConnectionError:

        fail(
            "Could not connect to API.\n"
            f"Start it with:\n"
            f"python src\\app.py"
        )


    except Exception as error:

        fail(
            f"GET request failed: {error}"
        )


# =========================================================
# POST REQUEST
# =========================================================

def call_api(features):

    try:

        response = requests.post(

            f"{API_URL}/api/predict",

            json={
                "features": features
            },

            timeout=30
        )


        try:

            body = response.json()

        except Exception:

            body = response.text


        return (
            response.status_code,
            body
        )


    except requests.exceptions.ConnectionError:

        fail(
            "Could not connect to NETSENTRY API.\n"
            f"Make sure the API is running on {API_URL}"
        )


    except requests.exceptions.Timeout:

        fail(
            "API request timed out."
        )


    except Exception as error:

        fail(
            f"API request failed: {error}"
        )


# =========================================================
# HEALTH CHECK
# =========================================================

def test_health():

    print_header(
        "1. HEALTH CHECK"
    )


    status, body = call_get(
        "/health"
    )


    print(
        f"HTTP Status: {status}"
    )


    print(
        json.dumps(
            body,
            indent=2
        )
    )


    if status != 200:

        fail(
            "Health endpoint failed."
        )


    if not isinstance(
        body,
        dict
    ):

        fail(
            "Health response is not JSON."
        )


    if body.get(
        "status"
    ) != "healthy":

        fail(
            "API did not report healthy status."
        )


    print(
        "\nPASS - API is healthy."
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

def test_model_info():

    print_header(
        "2. MODEL INFORMATION"
    )


    status, body = call_get(
        "/api/model"
    )


    print(
        f"HTTP Status: {status}"
    )


    if status != 200:

        fail(
            "/api/model failed."
        )


    print(
        json.dumps(
            body,
            indent=2
        )
    )


    if body.get(
        "binary_model",
        {}
    ).get(
        "feature_count"
    ) != 78:

        fail(
            "Binary model does not report 78 features."
        )


    if body.get(
        "multiclass_model",
        {}
    ).get(
        "feature_count"
    ) != 70:

        fail(
            "Multiclass model does not report 70 features."
        )


    print(
        "\nPASS - Model information endpoint works."
    )


# =========================================================
# LOAD FEATURES
# =========================================================

def load_feature_names():

    print_header(
        "3. LOADING MODEL FEATURES"
    )


    if not FEATURE_PATH.exists():

        fail(
            f"Feature file not found:\n"
            f"{FEATURE_PATH}"
        )


    feature_names = joblib.load(
        FEATURE_PATH
    )


    feature_names = list(
        feature_names
    )


    print(
        f"Features loaded: "
        f"{len(feature_names)}"
    )


    if len(feature_names) != 70:

        fail(
            "Expected exactly 70 multiclass features."
        )


    return feature_names


# =========================================================
# LOAD FRIDAY DATA
# =========================================================

def load_friday_data():

    print_header(
        "4. LOADING FRIDAY TEST DATA"
    )


    frames = []


    for filename in FRIDAY_FILES:

        path = DATA_DIR / filename


        if not path.exists():

            print(
                f"WARNING: Missing:\n"
                f"  {path}"
            )

            continue


        print(
            f"\nLoading: {filename}"
        )


        df = pd.read_csv(
            path,
            low_memory=False
        )


        print(
            f"Rows: {len(df):,}"
        )


        frames.append(
            df
        )


    if not frames:

        fail(
            "No Friday datasets were found."
        )


    df = pd.concat(
        frames,
        ignore_index=True
    )


    print(
        f"\nCombined rows: "
        f"{len(df):,}"
    )


    if " Label" in df.columns:

        label_column = " Label"

    elif "Label" in df.columns:

        label_column = "Label"

    else:

        fail(
            "Could not find dataset label column."
        )


    df["attack_type"] = (
        df[label_column]
        .astype(str)
        .str.strip()
    )


    return df


# =========================================================
# FEATURE PREPARATION
# =========================================================

def prepare_record(
    row,
    feature_names
):

    cleaned = {}


    for key, value in row.items():

        clean_key = str(
            key
        ).strip()

        cleaned[
            clean_key
        ] = value


    values = []


    missing = []


    for feature in feature_names:

        clean_feature = str(
            feature
        ).strip()


        if clean_feature not in cleaned:

            missing.append(
                clean_feature
            )

            continue


        value = cleaned[
            clean_feature
        ]


        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            value = 0.0


        if not np.isfinite(
            value
        ):

            value = 0.0


        values.append(
            value
        )


    if missing:

        raise ValueError(
            "Missing features: "
            + ", ".join(missing)
        )


    return {
        feature: values[index]
        for index, feature
        in enumerate(feature_names)
    }


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

def print_distribution(df):

    print_header(
        "5. FRIDAY CLASS DISTRIBUTION"
    )


    distribution = (
        df["attack_type"]
        .value_counts()
    )


    print(
        distribution.to_string()
    )


# =========================================================
# SAMPLE RECORDS
# =========================================================

def select_records(df):

    print_header(
        "6. SELECTING API TEST SAMPLES"
    )


    traffic_types = [
        "BENIGN",
        "DDoS",
        "PortScan",
        "Bot"
    ]


    selected = {}


    for traffic_type in traffic_types:

        subset = df[
            df["attack_type"]
            == traffic_type
        ]


        if len(subset) == 0:

            print(
                f"{traffic_type:<12}"
                f" NOT FOUND"
            )

            continue


        # Deterministic samples spread throughout
        # the class instead of selecting only one
        # arbitrary middle record.

        if len(subset) <= SAMPLES_PER_CLASS:

            sample = subset

        else:

            indexes = np.linspace(
                0,
                len(subset) - 1,
                SAMPLES_PER_CLASS,
                dtype=int
            )

            sample = subset.iloc[
                indexes
            ]


        selected[
            traffic_type
        ] = sample


        print(
            f"{traffic_type:<12}"
            f" {len(subset):,} available"
            f" -> {len(sample)} API samples"
        )


    return selected


# =========================================================
# TEST BINARY / API BEHAVIOR
# =========================================================

def test_attack_types(
    records,
    feature_names
):

    print_header(
        "7. API TRAFFIC VALIDATION"
    )


    results = []


    for traffic_type, rows in records.items():

        print(
            "\n"
            + "-" * 70
        )


        print(
            f"TESTING: {traffic_type}"
        )


        expected_prediction = (
            "BENIGN"
            if traffic_type == "BENIGN"
            else "ATTACK"
        )


        correct = 0

        total = len(rows)

        attack_detected = 0

        unknown_detected = 0

        known_type_detected = 0


        for sample_number, (_, row) in enumerate(
            rows.iterrows(),
            start=1
        ):

            try:

                features = prepare_record(
                    row,
                    feature_names
                )

            except Exception as error:

                print(
                    f"\nSample {sample_number}: "
                    f"feature preparation failed:"
                )

                print(
                    error
                )

                continue


            status, body = call_api(
                features
            )


            if status != 200:

                print(
                    f"\nSample {sample_number}: "
                    f"HTTP {status}"
                )

                continue


            prediction = body.get(
                "prediction"
            )


            attack_type = body.get(
                "attack_type"
            )


            confidence = body.get(
                "confidence_percent",
                0.0
            )


            attack_probability = (
                body
                .get("probabilities", {})
                .get("1", 0.0)
            )


            if prediction == expected_prediction:

                correct += 1


            if prediction == "ATTACK":

                attack_detected += 1


            if attack_type == "UNKNOWN_UNSEEN":

                unknown_detected += 1


            elif (
                attack_type != "BENIGN"
                and attack_type is not None
            ):

                known_type_detected += 1


            # Print first few samples only
            # to keep the terminal readable.

            if sample_number <= 3:

                print(
                    f"\nSample {sample_number}:"
                )

                print(
                    f"  Expected binary: "
                    f"{expected_prediction}"
                )

                print(
                    f"  Prediction: "
                    f"{prediction}"
                )

                print(
                    f"  Attack probability: "
                    f"{float(attack_probability) * 100:.2f}%"
                )

                print(
                    f"  Confidence: "
                    f"{confidence}%"
                )

                print(
                    f"  Attack type: "
                    f"{attack_type}"
                )


        accuracy = (
            correct / total
            if total
            else 0.0
        )


        detection_rate = (
            attack_detected / total
            if traffic_type != "BENIGN"
            and total
            else None
        )


        print(
            "\nSummary:"
        )


        print(
            f"  Binary correct: "
            f"{correct}/{total}"
        )


        print(
            f"  Binary accuracy: "
            f"{accuracy * 100:.2f}%"
        )


        if detection_rate is not None:

            print(
                f"  Attack detection: "
                f"{detection_rate * 100:.2f}%"
            )


        print(
            f"  UNKNOWN_UNSEEN: "
            f"{unknown_detected}"
        )


        print(
            f"  Known attack types: "
            f"{known_type_detected}"
        )


        results.append(
            {
                "traffic_type": traffic_type,
                "expected": expected_prediction,
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "attack_detected": attack_detected,
                "unknown_unseen": unknown_detected,
                "known_attack_type": known_type_detected
            }
        )


    return results


# =========================================================
# ERROR HANDLING
# =========================================================

def test_bad_requests():

    print_header(
        "8. API ERROR-HANDLING TESTS"
    )


    tests = [

        (
            "Missing features",
            {}
        ),

        (
            "Features is not an object",
            {
                "features": []
            }
        ),

        (
            "Missing feature",
            {
                "features": {
                    "Destination Port": 80
                }
            }
        )
    ]


    passed = 0


    for name, payload in tests:

        print(
            f"\nTesting: {name}"
        )


        try:

            response = requests.post(

                f"{API_URL}/api/predict",

                json=payload,

                timeout=15
            )


            print(
                f"HTTP Status: "
                f"{response.status_code}"
            )


            try:

                body = response.json()

                print(
                    json.dumps(
                        body,
                        indent=2
                    )
                )

            except Exception:

                body = None


            if response.status_code == 400:

                print(
                    "PASS"
                )

                passed += 1

            else:

                print(
                    "FAIL - expected HTTP 400"
                )


        except Exception as error:

            print(
                f"FAIL: {error}"
            )


    print(
        f"\nError handling: "
        f"{passed}/{len(tests)} passed"
    )


    return (
        passed == len(tests)
    )


# =========================================================
# SUMMARY
# =========================================================

def print_summary(
    results,
    error_tests_passed
):

    print_header(
        "NETSENTRY FINAL API VALIDATION SUMMARY"
    )


    print(
        f"{'Traffic Type':<15}"
        f"{'Expected':<12}"
        f"{'Accuracy':<12}"
        f"{'Attack Det.':<15}"
        f"{'Unknown':<12}"
        f"{'Known Type'}"
    )


    print(
        "-" * 85
    )


    for result in results:

        detection = "-"

        if (
            result["traffic_type"]
            != "BENIGN"
        ):

            detection = (
                f"{result['attack_detected']}/"
                f"{result['total']}"
            )


        print(
            f"{result['traffic_type']:<15}"
            f"{result['expected']:<12}"
            f"{result['accuracy'] * 100:>6.2f}%"
            f"{detection:<15}"
            f"{result['unknown_unseen']:<12}"
            f"{result['known_attack_type']}"
        )


    print(
        "\n"
        + "-" * 85
    )


    total_samples = sum(
        result["total"]
        for result in results
    )


    total_correct = sum(
        result["correct"]
        for result in results
    )


    overall_accuracy = (
        total_correct
        / total_samples
        if total_samples
        else 0.0
    )


    print(
        f"Total API samples: "
        f"{total_samples}"
    )


    print(
        f"Binary validation accuracy: "
        f"{overall_accuracy * 100:.2f}%"
    )


    print(
        "Error handling: "
        + (
            "PASS"
            if error_tests_passed
            else "FAIL"
        )
    )


    print(
        "\n"
        + "=" * 70
    )


    # -----------------------------------------------------
    # IMPORTANT
    # -----------------------------------------------------
    #
    # We intentionally do NOT require DDoS/PortScan/Bot
    # to be classified as their true names.
    #
    # Those attack families are unseen by the multiclass
    # training model.
    #
    # Correct behavior is:
    #
    # Stage 1 -> ATTACK
    # Stage 2 -> UNKNOWN_UNSEEN
    #
    # -----------------------------------------------------

    print(
        "NETSENTRY API VALIDATION COMPLETE"
    )


    print(
        "=" * 70
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print_header(
        "NETSENTRY - FINAL API VALIDATION"
    )


    print(
        f"API: {API_URL}"
    )


    print(
        "\nRequired architecture:"
    )


    print(
        "Stage 1: Binary model -> BENIGN / ATTACK"
    )


    print(
        "Stage 2: Multiclass model -> known attack / UNKNOWN_UNSEEN"
    )


    # -----------------------------------------------------
    # Health
    # -----------------------------------------------------

    test_health()


    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    test_model_info()


    # -----------------------------------------------------
    # Features
    # -----------------------------------------------------

    feature_names = (
        load_feature_names()
    )


    # -----------------------------------------------------
    # Friday data
    # -----------------------------------------------------

    df = load_friday_data()


    # -----------------------------------------------------
    # Distribution
    # -----------------------------------------------------

    print_distribution(
        df
    )


    # -----------------------------------------------------
    # Select samples
    # -----------------------------------------------------

    records = select_records(
        df
    )


    # -----------------------------------------------------
    # Traffic tests
    # -----------------------------------------------------

    results = test_attack_types(
        records,
        feature_names
    )


    # -----------------------------------------------------
    # Error handling
    # -----------------------------------------------------

    error_tests_passed = (
        test_bad_requests()
    )


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print_summary(
        results,
        error_tests_passed
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()