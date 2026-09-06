from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

# =========================================================
# NETSENTRY - PRODUCTION API
# =========================================================
#
# Two-stage intrusion detection architecture
#
# Stage 1:
#   Binary Random Forest
#   BENIGN / ATTACK
#
# Stage 2:
#   Multiclass Random Forest
#   Known attack family / UNKNOWN_UNSEEN
#
# Production artifacts:
#   models/intrusion_detector_production.joblib
#   models/intrusion_detector_multiclass_production.joblib
#   models/feature_names_production.joblib
#
# Thresholds:
#   Binary attack threshold: 0.55
#   Unknown attack-type threshold: 0.30
#
# API:
#   GET  /
#   GET  /health
#   GET  /api/model
#   POST /api/predict
#
# =========================================================


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"


BINARY_MODEL_PATH = MODEL_DIR / "intrusion_detector_production.joblib"


MULTICLASS_MODEL_PATH = MODEL_DIR / "intrusion_detector_multiclass_production.joblib"


FEATURE_PATH = MODEL_DIR / "feature_names_production.joblib"


# =========================================================
# CONFIGURATION
# =========================================================

# Read from the environment so a deployment never needs a code edit.

HOST = os.getenv("NETSENTRY_HOST", "127.0.0.1")

PORT = int(os.getenv("NETSENTRY_PORT", "8086"))


# Werkzeug's debugger runs arbitrary code and puts source in tracebacks.
DEBUG = os.getenv("NETSENTRY_DEBUG", "").strip().lower() in ("1", "true", "yes")


# Stage 1 threshold
BINARY_ATTACK_THRESHOLD = float(os.getenv("NETSENTRY_ATTACK_THRESHOLD", "0.55"))


# Stage 2 threshold
UNKNOWN_TYPE_THRESHOLD = float(os.getenv("NETSENTRY_UNKNOWN_THRESHOLD", "0.30"))


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)


# =========================================================
# GLOBAL MODEL OBJECTS
# =========================================================

binary_model = None

multiclass_model = None

binary_feature_names = []

multiclass_feature_names = []

binary_only_features = []


# =========================================================
# LOGGING
# =========================================================


def log(message):

    print(f"[NETSENTRY] {message}", flush=True)


# =========================================================
# MODEL LOADING
# =========================================================


def load_models():

    global binary_model
    global multiclass_model
    global binary_feature_names
    global multiclass_feature_names
    global binary_only_features

    log("Loading production artifacts...")

    # -----------------------------------------------------
    # Binary model
    # -----------------------------------------------------

    if not BINARY_MODEL_PATH.exists():

        raise FileNotFoundError(f"Binary model not found:\n" f"{BINARY_MODEL_PATH}")

    binary_model = joblib.load(BINARY_MODEL_PATH)

    # -----------------------------------------------------
    # Multiclass model
    # -----------------------------------------------------

    if not MULTICLASS_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Multiclass model not found:\n" f"{MULTICLASS_MODEL_PATH}"
        )

    multiclass_model = joblib.load(MULTICLASS_MODEL_PATH)

    # -----------------------------------------------------
    # Feature names
    # -----------------------------------------------------

    if not FEATURE_PATH.exists():

        raise FileNotFoundError(f"Feature file not found:\n" f"{FEATURE_PATH}")

    production_features = joblib.load(FEATURE_PATH)

    production_features = [str(feature).strip() for feature in production_features]

    binary_feature_names = list(production_features)

    multiclass_feature_names = list(production_features)

    binary_only_features = [
        feature
        for feature in binary_feature_names
        if feature not in multiclass_feature_names
    ]

    # -----------------------------------------------------
    # Validate schema
    # -----------------------------------------------------

    if len(binary_feature_names) == 0:

        raise ValueError("No binary features were loaded.")

    if len(multiclass_feature_names) == 0:

        raise ValueError("No multiclass features were loaded.")

    if len(binary_feature_names) != len(multiclass_feature_names):

        log("WARNING: binary and multiclass feature counts differ.")

    log(f"Binary model loaded: " f"{BINARY_MODEL_PATH.name}")

    log(f"Multiclass model loaded: " f"{MULTICLASS_MODEL_PATH.name}")

    log(f"Production features loaded: " f"{len(production_features)}")

    log(f"Binary features: " f"{len(binary_feature_names)}")

    log(f"Multiclass features: " f"{len(multiclass_feature_names)}")

    log(f"Binary attack threshold: " f"{BINARY_ATTACK_THRESHOLD}")

    log(f"Unknown type threshold: " f"{UNKNOWN_TYPE_THRESHOLD}")


# =========================================================
# MODEL CLASS HELPERS
# =========================================================


def normalize_class(value):
    """
    Convert model class labels into a consistent representation.

    Supports models whose classes are:

        0 / 1

    or:

        "0" / "1"

    or:

        "BENIGN" / "ATTACK"
    """

    text = str(value).strip()

    return text


def get_binary_probabilities(X):
    """
    Return:

        {
            "BENIGN": probability,
            "ATTACK": probability
        }

    regardless of whether the underlying model stores
    classes as integers or strings.
    """

    probabilities = binary_model.predict_proba(X)[0]

    classes = list(binary_model.classes_)

    probability_map = {}

    for cls, probability in zip(classes, probabilities):

        normalized = normalize_class(cls)

        probability_map[normalized] = float(probability)

    benign_probability = 0.0

    attack_probability = 0.0

    # -----------------------------------------------------
    # Numeric class encoding
    # -----------------------------------------------------

    if "0" in probability_map:

        benign_probability = probability_map["0"]

    if "1" in probability_map:

        attack_probability = probability_map["1"]

    # -----------------------------------------------------
    # String class encoding
    # -----------------------------------------------------

    if "BENIGN" in probability_map:

        benign_probability = probability_map["BENIGN"]

    if "ATTACK" in probability_map:

        attack_probability = probability_map["ATTACK"]

    return (benign_probability, attack_probability, probability_map)


# =========================================================
# FEATURE PREPARATION
# =========================================================


def clean_features(features):
    """
    Normalize incoming JSON feature names and values.

    Returns:

        {
            "Feature A": float,
            "Feature B": float,
            ...
        }
    """

    if not isinstance(features, dict):

        raise ValueError("'features' must be a JSON object.")

    cleaned = {}

    for key, value in features.items():

        clean_key = str(key).strip()

        if not clean_key:

            continue

        cleaned[clean_key] = value

    return cleaned


def validate_required_features(features, feature_names):

    missing = []

    for feature in feature_names:

        clean_feature = str(feature).strip()

        if clean_feature not in features:

            missing.append(clean_feature)

    return missing


def prepare_input(features, feature_names):
    """
    Convert API JSON into a model-ready DataFrame.

    Every required production feature must be present.

    Non-numeric values and NaN/Infinity are converted to 0.
    """

    cleaned = clean_features(features)

    missing = validate_required_features(cleaned, feature_names)

    if missing:

        raise ValueError("Missing binary model features: " + ", ".join(missing))

    values = []

    for feature in feature_names:

        value = cleaned[str(feature).strip()]

        try:

            value = float(value)

        except (TypeError, ValueError):

            value = 0.0

        if not np.isfinite(value):

            value = 0.0

        values.append(value)

    X = pd.DataFrame([values], columns=feature_names)

    return X


# =========================================================
# RISK LEVEL
# =========================================================


def get_risk_level(prediction, confidence):
    """
    Convert binary model confidence into an operational
    risk level.

    This is deliberately confidence-driven, not a separate model:

    ATTACK predictions:
        confidence >= 0.90  -> CRITICAL  (act on this now)
        confidence >= 0.75  -> HIGH
        confidence >= 0.50  -> MEDIUM
        else                -> LOW       (flagged ATTACK, but weakly)

    BENIGN predictions:
        confidence >= 0.90  -> LOW       (safe to ignore)
        confidence >= 0.70  -> MEDIUM
        else                -> HIGH      (called BENIGN, but the
                                           model wasn't sure -- worth
                                           a human glancing at)

    A low-confidence BENIGN call is intentionally treated as higher
    risk than a high-confidence one: "probably fine" is a weaker
    signal than "definitely fine," and this keeps that distinction
    visible in the risk level rather than hiding it behind a binary
    label.
    """

    confidence = float(confidence)

    if prediction == "BENIGN":

        if confidence >= 0.90:

            return "LOW"

        if confidence >= 0.70:

            return "MEDIUM"

        return "HIGH"

    # -----------------------------------------------------
    # ATTACK
    # -----------------------------------------------------

    if confidence >= 0.90:

        return "CRITICAL"

    if confidence >= 0.75:

        return "HIGH"

    if confidence >= 0.50:

        return "MEDIUM"

    return "LOW"


# =========================================================
# ATTACK TYPE PREDICTION
# =========================================================


def predict_attack_type(X):
    """
    Stage 2 multiclass prediction.

    Returns:

        attack_type
        attack_type_prediction
        attack_type_confidence
        attack_type_probabilities
    """

    probabilities = multiclass_model.predict_proba(X)[0]

    classes = list(multiclass_model.classes_)

    attack_index = int(np.argmax(probabilities))

    attack_type_prediction = str(classes[attack_index])

    attack_type_confidence = float(probabilities[attack_index])

    attack_type_probabilities = {
        str(cls): round(float(probability), 6)
        for cls, probability in zip(classes, probabilities)
    }

    # -----------------------------------------------------
    # Unknown-class protection
    # -----------------------------------------------------
    #
    # If the multiclass model predicts BENIGN,
    # the traffic has already passed Stage 1 as ATTACK.
    #
    # Therefore we should never call it BENIGN.
    #
    # If confidence is below the configured threshold,
    # do not pretend we know the attack family.
    # -----------------------------------------------------

    if attack_type_prediction.upper() == "BENIGN":

        attack_type = "UNKNOWN_UNSEEN"

    elif attack_type_confidence >= UNKNOWN_TYPE_THRESHOLD:

        attack_type = attack_type_prediction

    else:

        attack_type = "UNKNOWN_UNSEEN"

    return (
        attack_type,
        attack_type_prediction,
        attack_type_confidence,
        attack_type_probabilities,
    )


# =========================================================
# ROOT ENDPOINT
# =========================================================


@app.route("/", methods=["GET"])
def home():

    attack_types = [
        str(cls) for cls in multiclass_model.classes_ if str(cls).upper() != "BENIGN"
    ]

    return jsonify(
        {
            "name": "NETSENTRY",
            "description": (
                "Two-stage machine-learning " "network intrusion detection API."
            ),
            "architecture": {
                "stage_1": ("Binary Random Forest"),
                "stage_2": ("Multiclass Random Forest"),
            },
            "binary_attack_threshold": (BINARY_ATTACK_THRESHOLD),
            "unknown_type_threshold": (UNKNOWN_TYPE_THRESHOLD),
            "binary_feature_count": (len(binary_feature_names)),
            "multiclass_feature_count": (len(multiclass_feature_names)),
            "attack_types": attack_types,
            "endpoints": {
                "health": "/health",
                "model": "/api/model",
                "predict": ("POST /api/predict"),
            },
        }
    )


# =========================================================
# HEALTH
# =========================================================


@app.route("/health", methods=["GET"])
def health():

    return jsonify(
        {
            "status": "healthy",
            "model_loaded": (binary_model is not None and multiclass_model is not None),
            "binary_model_loaded": (binary_model is not None),
            "multiclass_model_loaded": (multiclass_model is not None),
            "binary_feature_count": (len(binary_feature_names)),
            "multiclass_feature_count": (len(multiclass_feature_names)),
            "binary_only_feature_count": (len(binary_only_features)),
            "class_count": 2,
            "attack_class_count": (
                len(multiclass_model.classes_) if multiclass_model is not None else 0
            ),
            "binary_attack_threshold": (BINARY_ATTACK_THRESHOLD),
            "unknown_type_threshold": (UNKNOWN_TYPE_THRESHOLD),
        }
    )


# =========================================================
# MODEL INFORMATION
# =========================================================


@app.route("/api/model", methods=["GET"])
def model_info():

    if binary_model is None or multiclass_model is None:

        return jsonify({"error": "Models are not loaded."}), 503

    return jsonify(
        {
            "model": ("Two-stage Random Forest"),
            "binary_model": {
                "file": (BINARY_MODEL_PATH.name),
                "classes": [str(cls) for cls in binary_model.classes_],
                "feature_count": (len(binary_feature_names)),
                "attack_threshold": (BINARY_ATTACK_THRESHOLD),
            },
            "multiclass_model": {
                "file": (MULTICLASS_MODEL_PATH.name),
                "classes": [str(cls) for cls in multiclass_model.classes_],
                "feature_count": (len(multiclass_feature_names)),
            },
            "binary_only_features": (binary_only_features),
            "unknown_type_threshold": (UNKNOWN_TYPE_THRESHOLD),
        }
    )


# =========================================================
# PREDICTION API
# =========================================================


@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        # =================================================
        # REQUEST VALIDATION
        # =================================================

        if not request.is_json:

            return jsonify({"error": ("Request must contain " "JSON data.")}), 400

        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):

            return jsonify({"error": ("Request body must be " "a JSON object.")}), 400

        if "features" not in payload:

            return jsonify({"error": ("Missing 'features' object.")}), 400

        features = payload["features"]

        if not isinstance(features, dict):

            return jsonify({"error": ("'features' must be " "a JSON object.")}), 400

        # =================================================
        # STAGE 1 INPUT
        # =================================================

        binary_X = prepare_input(features, binary_feature_names)

        # =================================================
        # STAGE 1
        # BINARY DETECTION
        # =================================================

        benign_probability, attack_probability, binary_probability_map = (
            get_binary_probabilities(binary_X)
        )

        # -------------------------------------------------
        # Explicit attack threshold
        # -------------------------------------------------

        is_attack = attack_probability >= BINARY_ATTACK_THRESHOLD

        if is_attack:

            prediction = "ATTACK"

            confidence = attack_probability

        else:

            prediction = "BENIGN"

            confidence = benign_probability

        # =================================================
        # STAGE 2
        # ATTACK TYPE
        # =================================================

        attack_type = "BENIGN"

        attack_type_prediction = None

        attack_type_confidence = None

        attack_type_probabilities = {}

        if is_attack:

            multiclass_X = prepare_input(features, multiclass_feature_names)

            (
                attack_type,
                attack_type_prediction,
                attack_type_confidence,
                attack_type_probabilities,
            ) = predict_attack_type(multiclass_X)

        # =================================================
        # RISK
        # =================================================

        confidence = float(confidence)

        risk_level = get_risk_level(prediction, confidence)

        # =================================================
        # SORT BINARY PROBABILITIES
        # =================================================

        sorted_binary_probabilities = dict(
            sorted(
                binary_probability_map.items(), key=lambda item: item[1], reverse=True
            )
        )

        # =================================================
        # RESPONSE
        # =================================================

        return jsonify(
            {
                # -----------------------------------------
                # Primary prediction
                # -----------------------------------------
                "prediction": prediction,
                "is_attack": is_attack,
                # -----------------------------------------
                # Binary confidence
                # -----------------------------------------
                "confidence": round(confidence, 6),
                "confidence_percent": round(confidence * 100, 2),
                # -----------------------------------------
                # Risk
                # -----------------------------------------
                "risk_level": risk_level,
                # -----------------------------------------
                # Attack type
                # -----------------------------------------
                "attack_type": attack_type,
                "attack_type_prediction": (attack_type_prediction),
                "attack_type_confidence": (
                    round(attack_type_confidence, 6)
                    if attack_type_confidence is not None
                    else None
                ),
                # -----------------------------------------
                # Binary probabilities
                # -----------------------------------------
                "probabilities": {
                    label: round(probability, 6)
                    for (label, probability) in sorted_binary_probabilities.items()
                },
                # -----------------------------------------
                # Multiclass probabilities
                # -----------------------------------------
                "attack_type_probabilities": (attack_type_probabilities),
                # -----------------------------------------
                # Models
                # -----------------------------------------
                "models": {
                    "binary": (BINARY_MODEL_PATH.name),
                    "multiclass": (MULTICLASS_MODEL_PATH.name),
                },
                # -----------------------------------------
                # Thresholds
                # -----------------------------------------
                "thresholds": {
                    "binary_attack": (BINARY_ATTACK_THRESHOLD),
                    "unknown_attack_type": (UNKNOWN_TYPE_THRESHOLD),
                },
                # -----------------------------------------
                # Feature information
                # -----------------------------------------
                "features": {
                    "binary": (len(binary_feature_names)),
                    "multiclass": (len(multiclass_feature_names)),
                    "provided": (len(features)),
                },
            }
        )

    # =====================================================
    # BAD REQUEST
    # =====================================================

    except ValueError as error:

        return jsonify({"error": str(error)}), 400

    # =====================================================
    # INTERNAL ERROR
    # =====================================================

    except Exception as error:

        log(f"Prediction error: {repr(error)}")

        return jsonify({"error": ("Internal prediction error.")}), 500


# =========================================================
# STARTUP
# =========================================================


def initialize():

    print()
    print("=" * 70)
    print("NETSENTRY - PRODUCTION API")
    print("=" * 70)

    print(f"Base directory: {BASE_DIR}")

    print(f"Model directory: {MODEL_DIR}")

    print()

    load_models()

    print()

    print("=" * 70)
    print("NETSENTRY READY")
    print("=" * 70)

    print(f"API: http://{HOST}:{PORT}")

    print(f"Debug: {DEBUG}")

    print("GET  /health")

    print("GET  /api/model")

    print("POST /api/predict")

    print("=" * 70)
    print()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    initialize()

    app.run(host=HOST, port=PORT, debug=DEBUG)
