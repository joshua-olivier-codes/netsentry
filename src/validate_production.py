from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import requests


# =========================================================
# NETSENTRY - PRODUCTION API VALIDATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

API_URL = "http://127.0.0.1:8085"

FEATURE_PATH = (
    BASE_DIR
    / "models"
    / "feature_names_multiclass.joblib"
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "test.csv"
)

# ---------------------------------------------------------
# FALLBACK TEST DATA PATHS
# ---------------------------------------------------------
#
# If test.csv does not exist, the validator will try these.
#

FALLBACK_DATA_PATHS = [
    BASE_DIR / "data" / "processed" / "held_out_test.csv",
    BASE_DIR / "data" / "processed" / "test_data.csv",
    BASE_DIR / "data" / "processed" / "test_set.csv",
    BASE_DIR / "data" / "raw" / "MachineLearningCVE",
]


# Number of samples per class used for API smoke validation
SAMPLES_PER_CLASS = 15


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


def normalize_attack_label(label):
    """
    Same normalization used in evaluate_production.py / test_api.py.

    The raw CICIDS2017 CSVs encode the "Web Attack" labels with a
    mangled dash character (mojibake). Without this normalization,
    exact-match lookups for "Web Attack - Brute Force", "Web Attack
    - XSS", and "Web Attack - Sql Injection" silently find nothing,
    even though the rows are present in the data.
    """

    label = str(label).strip()

    if label.upper() == "BENIGN":
        return "BENIGN"

    replacements = {
        "Web Attack \ufffd Brute Force": "Web Attack - Brute Force",
        "Web Attack \ufffd XSS": "Web Attack - XSS",
        "Web Attack \ufffd Sql Injection": "Web Attack - Sql Injection",
    }

    return replacements.get(label, label)


def find_test_data():
    """
    Locate the held-out test dataset.

    Supports:
        - a single CSV file
        - a directory containing CSV files
    """

    if DATA_PATH.exists():
        return DATA_PATH

    for path in FALLBACK_DATA_PATHS:

        if not path.exists():
            continue

        if path.is_file() and path.suffix.lower() == ".csv":
            return path

        if path.is_dir():

            csv_files = sorted(
                path.glob("*.csv")
            )

            if csv_files:
                return csv_files

    fail(
        "Could not locate held-out test data.\n"
        f"Expected something like:\n"
        f"  {DATA_PATH}"
    )


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
            "Could not connect to NETSENTRY API.\n"
            f"Make sure the API is running on:\n"
            f"{API_URL}\n\n"
            f"Start it with:\n"
            f"python src\\app.py"
        )

    except requests.exceptions.Timeout:

        fail(
            "GET request timed out."
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

    if not isinstance(body, dict):
        fail(
            "Health response is not JSON."
        )

    if body.get("status") != "healthy":
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

    if not isinstance(body, dict):
        fail(
            "/api/model did not return JSON."
        )

    print(
        json.dumps(
            body,
            indent=2
        )
    )

    binary_features = (
        body
        .get("binary_model", {})
        .get("feature_count")
    )

    multiclass_features = (
        body
        .get("multiclass_model", {})
        .get("feature_count")
    )

    print(
        f"\nBinary feature count reported by API: "
        f"{binary_features}"
    )

    print(
        f"Multiclass feature count reported by API: "
        f"{multiclass_features}"
    )

    if binary_features != 70:

        fail(
            "Binary model does not report 70 features."
        )

    if multiclass_features != 70:

        fail(
            "Multiclass model does not report 70 features."
        )

    print(
        "\nPASS - Model information endpoint works."
    )

    return body


# =========================================================
# LOAD FEATURES
# =========================================================

def load_feature_names():

    print_header(
        "3. LOADING PRODUCTION MODEL FEATURES"
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
        f"Features loaded: {len(feature_names)}"
    )

    if len(feature_names) != 70:

        fail(
            "Expected exactly 70 production features."
        )

    return feature_names


# =========================================================
# LOAD HELD-OUT DATA
# =========================================================

def load_held_out_data():

    print_header(
        "4. LOADING HELD-OUT TEST DATA"
    )

    data_source = find_test_data()

    frames = []

    if isinstance(
        data_source,
        list
    ):

        for path in data_source:

            print(
                f"\nLoading: {path.name}"
            )

            try:

                df = pd.read_csv(
                    path,
                    low_memory=False
                )

            except Exception as error:

                print(
                    f"WARNING: Could not read {path}"
                )

                print(error)

                continue

            print(
                f"Rows: {len(df):,}"
            )

            frames.append(
                df
            )

        if not frames:

            fail(
                "None of the discovered CSV files could be loaded."
            )

        df = pd.concat(
            frames,
            ignore_index=True
        )

    else:

        print(
            f"Loading: {data_source}"
        )

        try:

            df = pd.read_csv(
                data_source,
                low_memory=False
            )

        except Exception as error:

            fail(
                f"Could not read test data:\n{error}"
            )

    print(
        f"Rows: {len(df):,}"
    )

    if len(df) == 0:

        fail(
            "Held-out test dataset is empty."
        )

    # -----------------------------------------------------
    # FIND LABEL COLUMN
    # -----------------------------------------------------

    label_column = None

    for candidate in [
        " Label",
        "Label",
        "label",
        "attack_type",
        "Attack Type"
    ]:

        if candidate in df.columns:

            label_column = candidate

            break

    if label_column is None:

        fail(
            "Could not find dataset label column.\n"
            "Expected one of: Label, attack_type, Attack Type"
        )

    df["attack_type"] = (
        df[label_column]
        .astype(str)
        .str.strip()
        .apply(normalize_attack_label)
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
        "5. HELD-OUT CLASS DISTRIBUTION"
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
        "DoS Hulk",
        "PortScan",
        "DDoS",
        "DoS GoldenEye",
        "FTP-Patator",
        "SSH-Patator",
        "DoS Slowhttptest",
        "DoS slowloris",
        "Bot",
        "Web Attack - Brute Force",
        "Web Attack - XSS",
        "Infiltration",
        "Heartbleed",
        "Web Attack - Sql Injection"
    ]

    selected = {}

    for traffic_type in traffic_types:

        subset = df[
            df["attack_type"]
            == traffic_type
        ]

        if len(subset) == 0:

            print(
                f"{traffic_type:<30} NOT FOUND"
            )

            continue

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
            f"{traffic_type:<30}"
            f" {len(subset):,} available"
            f" -> {len(sample)} API samples"
        )

    return selected


# =========================================================
# API TRAFFIC VALIDATION
# =========================================================

def test_attack_types(
    records,
    feature_names
):

    print_header(
        "7. API TRAFFIC VALIDATION (HELD-OUT, LEAKAGE-FREE)"
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

        expected_binary = (
            "BENIGN"
            if traffic_type == "BENIGN"
            else "ATTACK"
        )

        total = len(rows)

        binary_correct = 0

        type_correct = 0

        type_evaluated = 0

        attack_detected = 0

        unknown_detected = 0

        known_type_detected = 0

        api_errors = 0

        # -------------------------------------------------
        # Keep individual observations for audit
        # -------------------------------------------------

        sample_results = []

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

                print(error)

                api_errors += 1

                sample_results.append(
                    {
                        "sample": sample_number,
                        "error": str(error)
                    }
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

                api_errors += 1

                sample_results.append(
                    {
                        "sample": sample_number,
                        "http_status": status,
                        "response": body
                    }
                )

                continue

            if not isinstance(body, dict):

                print(
                    f"\nSample {sample_number}: "
                    f"API returned invalid JSON object."
                )

                api_errors += 1

                continue

            prediction = body.get(
                "prediction"
            )

            attack_type = body.get(
                "attack_type"
            )

            # NOTE: app.py's /api/predict response uses the keys
            # "confidence" (binary) and "attack_type_confidence"
            # (Stage 2), not "binary_confidence"/"type_confidence".
            # Reading the wrong keys is why these printed as None.
            binary_confidence = body.get(
                "confidence"
            )

            type_confidence = body.get(
                "attack_type_confidence"
            )

            risk_level = body.get(
                "risk_level"
            )

            # -------------------------------------------------
            # BINARY EVALUATION
            # -------------------------------------------------

            if prediction == expected_binary:

                binary_correct += 1

            if (
                expected_binary == "ATTACK"
                and prediction == "ATTACK"
            ):

                attack_detected += 1

            # -------------------------------------------------
            # ATTACK TYPE EVALUATION
            # -------------------------------------------------
            #
            # BENIGN does not get type accuracy because
            # Stage 2 is not supposed to classify it as
            # an attack family.
            #

            if expected_binary == "ATTACK":

                if attack_type == traffic_type:

                    type_correct += 1

                if attack_type is not None:

                    type_evaluated += 1

                if attack_type == "UNKNOWN_UNSEEN":

                    unknown_detected += 1

                elif (
                    attack_type is not None
                    and attack_type != "BENIGN"
                ):

                    known_type_detected += 1

            # -------------------------------------------------
            # PRINT FIRST THREE
            # -------------------------------------------------

            if sample_number <= 3:

                print(
                    f"\nSample {sample_number}:"
                )

                print(
                    f"  Expected binary: "
                    f"{expected_binary}"
                )

                print(
                    f"  Prediction:      "
                    f"{prediction}"
                )

                print(
                    f"  Attack type:     "
                    f"{attack_type}"
                )

                print(
                    f"  Binary confidence: "
                    f"{binary_confidence}"
                )

                print(
                    f"  Type confidence:   "
                    f"{type_confidence}"
                )

                print(
                    f"  Risk level:         "
                    f"{risk_level}"
                )

            sample_results.append(
                {
                    "sample": sample_number,
                    "expected_binary": expected_binary,
                    "expected_attack_type": traffic_type,
                    "prediction": prediction,
                    "attack_type": attack_type,
                    "binary_confidence": binary_confidence,
                    "type_confidence": type_confidence,
                    "risk_level": risk_level
                }
            )

        # -----------------------------------------------------
        # CALCULATE METRICS
        # -----------------------------------------------------

        valid_samples = (
            total - api_errors
        )

        binary_accuracy = (
            binary_correct / valid_samples
            if valid_samples > 0
            else 0.0
        )

        if expected_binary == "ATTACK":

            detection_rate = (
                attack_detected / valid_samples
                if valid_samples > 0
                else 0.0
            )

            type_accuracy = (
                type_correct / valid_samples
                if valid_samples > 0
                else 0.0
            )

        else:

            detection_rate = None

            type_accuracy = None

        # -----------------------------------------------------
        # SUMMARY
        # -----------------------------------------------------

        print(
            "\nSummary:"
        )

        print(
            f"  Binary correct: "
            f"{binary_correct}/{valid_samples} "
            f"({binary_accuracy * 100:.2f}%)"
        )

        if expected_binary == "ATTACK":

            print(
                f"  Attack detection: "
                f"{attack_detected}/{valid_samples} "
                f"({detection_rate * 100:.2f}%)"
            )

            print(
                f"  Correct attack-type name: "
                f"{type_correct}/{valid_samples} "
                f"({type_accuracy * 100:.2f}%)"
            )

            print(
                f"  UNKNOWN_UNSEEN: "
                f"{unknown_detected}"
            )

            print(
                f"  Known type detected: "
                f"{known_type_detected}"
            )

        if api_errors:

            print(
                f"  API/sample errors: "
                f"{api_errors}"
            )

        # -----------------------------------------------------
        # IMPORTANT
        # -----------------------------------------------------
        #
        # The dictionary below contains ALL fields expected
        # by print_summary().
        #
        # This prevents the previous KeyError problems.
        # -----------------------------------------------------

        result = {
            "traffic_type": traffic_type,
            "expected_binary": expected_binary,

            "total": total,
            "valid_samples": valid_samples,
            "api_errors": api_errors,

            "binary_correct": binary_correct,
            "binary_accuracy": binary_accuracy,

            "attack_detected": attack_detected,
            "detection_rate": detection_rate,

            "type_correct": type_correct,
            "type_evaluated": type_evaluated,
            "type_accuracy": type_accuracy,

            "unknown_unseen": unknown_detected,
            "known_type_detected": known_type_detected,

            "samples": sample_results
        }

        results.append(
            result
        )

    return results


# =========================================================
# ERROR HANDLING TESTS
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
# SAVE API AUDIT
# =========================================================

def save_api_audit(results):

    """
    Save the validation results to JSON.

    This is intentionally best-effort so that a failure to
    write the audit file never causes the validation itself
    to crash.
    """

    try:

        audit_dir = (
            BASE_DIR
            / "reports"
        )

        audit_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        audit_path = (
            audit_dir
            / "netsentry_api_validation.json"
        )

        payload = {
            "api_url": API_URL,
            "samples_per_class": SAMPLES_PER_CLASS,
            "results": results
        }

        with open(
            audit_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                payload,
                file,
                indent=2,
                default=str
            )

        print(
            f"\nAudit saved to:\n"
            f"{audit_path}"
        )

    except Exception as error:

        print(
            "\nWARNING: Could not save API audit."
        )

        print(
            error
        )


# =========================================================
# SUMMARY
# =========================================================

def print_summary(
    results,
    error_tests_passed
):

    print_header(
        "NETSENTRY PRODUCTION VALIDATION SUMMARY"
    )

    print(
        f"{'Traffic Type':<30}"
        f"{'Binary Acc.':<15}"
        f"{'Attack Det.':<15}"
        f"{'Known Type':<15}"
        f"{'Unknown':<10}"
    )

    print(
        "-" * 85
    )

    for result in results:

        traffic_type = result.get(
            "traffic_type",
            "UNKNOWN"
        )

        binary_accuracy = result.get(
            "binary_accuracy",
            0.0
        )

        detection_rate = result.get(
            "detection_rate"
        )

        type_accuracy = result.get(
            "type_accuracy"
        )

        unknown_count = result.get(
            "unknown_unseen",
            0
        )

        if detection_rate is None:

            detection_text = "-"

        else:

            detection_text = (
                f"{detection_rate * 100:.2f}%"
            )

        if type_accuracy is None:

            type_text = "-"

        else:

            type_text = (
                f"{type_accuracy * 100:.2f}%"
            )

        print(
            f"{traffic_type:<30}"
            f"{binary_accuracy * 100:>7.2f}%"
            f"{detection_text:>12}"
            f"{type_text:>14}"
            f"{unknown_count:>10}"
        )

    print(
        "\n"
        + "-" * 85
    )

    # -----------------------------------------------------
    # GLOBAL METRICS
    # -----------------------------------------------------

    total_samples = sum(
        result.get(
            "valid_samples",
            0
        )
        for result in results
    )

    total_binary_correct = sum(
        result.get(
            "binary_correct",
            0
        )
        for result in results
    )

    overall_binary_accuracy = (
        total_binary_correct
        / total_samples
        if total_samples
        else 0.0
    )

    # -----------------------------------------------------
    # ATTACK-ONLY METRICS
    # -----------------------------------------------------

    attack_results = [
        result
        for result in results
        if result.get(
            "expected_binary"
        ) == "ATTACK"
    ]

    attack_samples = sum(
        result.get(
            "valid_samples",
            0
        )
        for result in attack_results
    )

    attack_detected = sum(
        result.get(
            "attack_detected",
            0
        )
        for result in attack_results
    )

    attack_type_correct = sum(
        result.get(
            "type_correct",
            0
        )
        for result in attack_results
    )

    overall_attack_detection = (
        attack_detected
        / attack_samples
        if attack_samples
        else 0.0
    )

    overall_attack_type_accuracy = (
        attack_type_correct
        / attack_samples
        if attack_samples
        else 0.0
    )

    # -----------------------------------------------------
    # API ERRORS
    # -----------------------------------------------------

    total_api_errors = sum(
        result.get(
            "api_errors",
            0
        )
        for result in results
    )

    # -----------------------------------------------------
    # ERROR HANDLING
    # -----------------------------------------------------

    print(
        f"Total API samples: "
        f"{total_samples}"
    )

    print(
        f"Binary validation accuracy: "
        f"{overall_binary_accuracy * 100:.2f}%"
    )

    print(
        f"Attack detection rate: "
        f"{overall_attack_detection * 100:.2f}%"
    )

    print(
        f"Attack-type accuracy: "
        f"{overall_attack_type_accuracy * 100:.2f}%"
    )

    print(
        f"API/sample errors: "
        f"{total_api_errors}"
    )

    print(
        "Error handling: "
        + (
            "PASS"
            if error_tests_passed
            else "FAIL"
        )
    )

    # -----------------------------------------------------
    # PER-CLASS DETAILS
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PER-CLASS DETAILS"
    )

    print(
        "=" * 70
    )

    for result in results:

        traffic_type = result.get(
            "traffic_type",
            "UNKNOWN"
        )

        print(
            f"\n{traffic_type}"
        )

        print(
            f"  Binary accuracy: "
            f"{result.get('binary_accuracy', 0.0) * 100:.2f}%"
        )

        if result.get(
            "detection_rate"
        ) is not None:

            print(
                f"  Attack detection: "
                f"{result['detection_rate'] * 100:.2f}%"
            )

        if result.get(
            "type_accuracy"
        ) is not None:

            print(
                f"  Attack-type accuracy: "
                f"{result['type_accuracy'] * 100:.2f}%"
            )

        print(
            f"  Unknown unseen: "
            f"{result.get('unknown_unseen', 0)}"
        )

        print(
            f"  Known type detected: "
            f"{result.get('known_type_detected', 0)}"
        )

    # -----------------------------------------------------
    # FINAL STATUS
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

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
        "NETSENTRY - PRODUCTION API VALIDATION"
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
    # HEALTH
    # -----------------------------------------------------

    test_health()

    # -----------------------------------------------------
    # MODEL INFORMATION
    # -----------------------------------------------------

    test_model_info()

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    feature_names = (
        load_feature_names()
    )

    # -----------------------------------------------------
    # HELD-OUT DATA
    # -----------------------------------------------------

    df = load_held_out_data()

    # -----------------------------------------------------
    # DISTRIBUTION
    # -----------------------------------------------------

    print_distribution(
        df
    )

    # -----------------------------------------------------
    # SAMPLE SELECTION
    # -----------------------------------------------------

    records = select_records(
        df
    )

    if not records:

        fail(
            "No supported traffic classes were found."
        )

    # -----------------------------------------------------
    # API TEST
    # -----------------------------------------------------

    results = test_attack_types(
        records,
        feature_names
    )

    # -----------------------------------------------------
    # SAVE AUDIT
    # -----------------------------------------------------

    save_api_audit(
        results
    )

    # -----------------------------------------------------
    # ERROR HANDLING
    # -----------------------------------------------------

    error_tests_passed = (
        test_bad_requests()
    )

    # -----------------------------------------------------
    # SUMMARY
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