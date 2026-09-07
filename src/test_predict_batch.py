"""
Tests for POST /api/predict/batch.

The models are stubbed rather than loaded, so this suite runs without the
trained .joblib artifacts and without a server. That is deliberate: the stub
probabilities are known exactly, which is what makes the equivalence checks
below meaningful.

The central claim under test is that batch prediction returns exactly what
calling /api/predict once per record would return.

    pytest src/test_predict_batch.py
"""

import hashlib

import numpy as np
import pytest

import app as netsentry


# =========================================================
# STUB MODELS
# =========================================================

# Feature names are synthetic. These tests are about batching behaviour, not
# about the production schema, so short names keep the fixtures readable.
FEATURE_NAMES = [f"f{index:02d}" for index in range(70)]

ATTACK_CLASSES = [
    "BENIGN",
    "DDoS",
    "PortScan",
    "DoS Hulk",
    "FTP-Patator",
    "Bot",
    "Heartbleed",
]

# Counts predict_proba invocations so the batching claim is measurable.
CALLS = {"binary": 0, "multiclass": 0}


def unit(seed):
    """Deterministic float in [0, 1) so runs are reproducible."""

    digest = hashlib.sha256(seed.encode()).digest()

    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def attack_probability(row):
    # f03 carries an explicit probability when it is set, so boundary cases
    # can be pinned exactly. Hash-derived values never land on a threshold.
    if row["f03"] > 0:
        return float(row["f03"])

    return unit(f"{row['f00']}:{row['f01']}")


class StubBinaryModel:
    """
    Orders probability columns by classes_, exactly as scikit-learn does,
    so the class-encoding handling in app.py is genuinely exercised.
    """

    def __init__(self, classes):
        self.classes_ = np.array(classes)

    def predict_proba(self, X):
        CALLS["binary"] += 1

        rows = []

        for _, row in X.iterrows():
            probability = attack_probability(row)

            columns = []

            for cls in self.classes_:
                name = str(cls).strip().upper()
                columns.append(probability if name in ("ATTACK", "1") else 1.0 - probability)

            rows.append(columns)

        return np.array(rows)


class StubMulticlassModel:
    def __init__(self):
        self.classes_ = np.array(ATTACK_CLASSES)

    def predict_proba(self, X):
        CALLS["multiclass"] += 1

        rows = []

        for _, row in X.iterrows():
            weights = [unit(f"{row['f00']}:{row['f01']}:{cls}") for cls in ATTACK_CLASSES]
            total = sum(weights)
            rows.append([weight / total for weight in weights])

        return np.array(rows)


# =========================================================
# FIXTURES
# =========================================================


@pytest.fixture
def make_client(monkeypatch):
    """Install stub models and return a Flask test client factory."""

    def factory(binary_classes=("ATTACK", "BENIGN")):
        monkeypatch.setattr(netsentry, "binary_model", StubBinaryModel(binary_classes))
        monkeypatch.setattr(netsentry, "multiclass_model", StubMulticlassModel())
        monkeypatch.setattr(netsentry, "binary_feature_names", list(FEATURE_NAMES))
        monkeypatch.setattr(netsentry, "multiclass_feature_names", list(FEATURE_NAMES))
        monkeypatch.setattr(netsentry, "binary_only_features", [])

        netsentry.app.config["TESTING"] = True

        CALLS["binary"] = 0
        CALLS["multiclass"] = 0

        return netsentry.app.test_client()

    return factory


@pytest.fixture
def client(make_client):
    return make_client()


def make_record(index):
    record = {name: 0.0 for name in FEATURE_NAMES}
    record["f00"] = float([80, 443, 22, 445, 53, 3389, 8080][index % 7])
    record["f01"] = float((index * 7919) % 120000)
    record["f02"] = float((index * 31) % 900)
    return record


# Fields a batch result shares with a single-prediction response.
SHARED_FIELDS = [
    "prediction",
    "is_attack",
    "confidence",
    "confidence_percent",
    "risk_level",
    "attack_type",
    "attack_type_prediction",
    "attack_type_confidence",
    "probabilities",
]


# =========================================================
# EQUIVALENCE
# =========================================================


@pytest.mark.parametrize(
    "binary_classes",
    [("ATTACK", "BENIGN"), ("BENIGN", "ATTACK"), (0, 1)],
    ids=["attack-first", "benign-first", "numeric"],
)
def test_batch_matches_single_endpoint(make_client, binary_classes):
    """Batch results must equal calling /api/predict once per record."""

    client = make_client(binary_classes)
    records = [make_record(index) for index in range(40)]

    singles = []

    for record in records:
        response = client.post("/api/predict", json={"features": record})
        assert response.status_code == 200
        singles.append(response.get_json())

    response = client.post(
        "/api/predict/batch",
        json={"records": records, "include_type_probabilities": True},
    )
    assert response.status_code == 200

    batch = response.get_json()

    assert batch["count"] == len(records)

    for position, (single, result) in enumerate(zip(singles, batch["results"])):
        assert result["index"] == position

        for field in SHARED_FIELDS:
            assert result[field] == single[field], f"record {position}, field {field}"

        if single["is_attack"]:
            assert result["attack_type_probabilities"] == single["attack_type_probabilities"]

    attacks = sum(1 for single in singles if single["is_attack"])

    assert batch["attack_count"] == attacks
    assert batch["benign_count"] == len(records) - attacks


def test_batch_makes_two_model_calls_regardless_of_size(client):
    """The point of the endpoint: prediction cost stops scaling with N."""

    records = [make_record(index) for index in range(40)]

    client.post("/api/predict", json={"features": records[0]})
    assert CALLS["binary"] == 1

    CALLS["binary"] = 0
    CALLS["multiclass"] = 0

    response = client.post("/api/predict/batch", json={"records": records})
    assert response.status_code == 200

    assert CALLS["binary"] == 1
    assert CALLS["multiclass"] <= 1


# =========================================================
# THRESHOLD BOUNDARY
# =========================================================


def test_threshold_is_inclusive(client):
    """
    A record landing exactly on the attack threshold classifies as ATTACK.

    Randomly distributed probabilities never hit the boundary, so without
    pinning it a >= / > mix-up between the two paths goes unnoticed.
    """

    record = make_record(0)
    record["f03"] = netsentry.BINARY_ATTACK_THRESHOLD

    single = client.post("/api/predict", json={"features": record}).get_json()
    batch = client.post("/api/predict/batch", json={"records": [record]}).get_json()

    assert single["is_attack"] is True
    assert batch["results"][0]["is_attack"] is True
    assert batch["results"][0]["prediction"] == single["prediction"]


def test_just_below_threshold_is_benign(client):
    record = make_record(0)
    record["f03"] = netsentry.BINARY_ATTACK_THRESHOLD - 1e-9

    single = client.post("/api/predict", json={"features": record}).get_json()
    batch = client.post("/api/predict/batch", json={"records": [record]}).get_json()

    assert single["is_attack"] is False
    assert batch["results"][0]["is_attack"] is False
    assert batch["results"][0]["prediction"] == single["prediction"]


# =========================================================
# STAGE 2 BEHAVIOUR
# =========================================================


def test_unknown_unseen_rules_are_preserved(client):
    """
    Stage 2 must resolve to UNKNOWN_UNSEEN when it predicts BENIGN, or when
    confidence falls below the configured threshold.
    """

    records = [make_record(index) for index in range(120)]

    batch = client.post(
        "/api/predict/batch",
        json={"records": records, "include_type_probabilities": True},
    ).get_json()

    threshold = batch["thresholds"]["unknown_attack_type"]
    attacks = [result for result in batch["results"] if result["is_attack"]]

    assert attacks, "fixture produced no attack rows"

    for result in attacks:
        prediction = result["attack_type_prediction"]
        confidence = result["attack_type_confidence"]

        if str(prediction).upper() == "BENIGN":
            expected = "UNKNOWN_UNSEEN"
        elif confidence >= threshold:
            expected = prediction
        else:
            expected = "UNKNOWN_UNSEEN"

        assert result["attack_type"] == expected


def test_benign_records_skip_stage_two(client):
    records = [make_record(index) for index in range(60)]

    batch = client.post("/api/predict/batch", json={"records": records}).get_json()

    for result in batch["results"]:
        if not result["is_attack"]:
            assert result["attack_type"] == "BENIGN"
            assert result["attack_type_prediction"] is None
            assert result["attack_type_confidence"] is None


# =========================================================
# RESPONSE SHAPE
# =========================================================


def test_type_probabilities_are_opt_in(client):
    records = [make_record(index) for index in range(30)]

    default = client.post("/api/predict/batch", json={"records": records}).get_json()
    requested = client.post(
        "/api/predict/batch",
        json={"records": records, "include_type_probabilities": True},
    ).get_json()

    assert all("attack_type_probabilities" not in r for r in default["results"])
    assert all("attack_type_probabilities" in r for r in requested["results"])


def test_shared_metadata_is_reported_once(client):
    """Model, threshold and feature metadata is identical per record."""

    records = [make_record(index) for index in range(10)]

    batch = client.post("/api/predict/batch", json={"records": records}).get_json()

    assert "models" in batch
    assert "thresholds" in batch
    assert "features" in batch

    for result in batch["results"]:
        assert "models" not in result
        assert "thresholds" not in result


# =========================================================
# REQUEST VALIDATION
# =========================================================


def test_missing_records_key_is_rejected(client):
    response = client.post("/api/predict/batch", json={})

    assert response.status_code == 400
    assert "records" in response.get_json()["error"]


@pytest.mark.parametrize("records", ["not-a-list", 5, {}], ids=["string", "int", "object"])
def test_non_array_records_is_rejected(client, records):
    response = client.post("/api/predict/batch", json={"records": records})

    assert response.status_code == 400


def test_empty_records_is_rejected(client):
    response = client.post("/api/predict/batch", json={"records": []})

    assert response.status_code == 400


def test_incomplete_record_names_its_index(client):
    """A large batch is undebuggable if the error does not say which record."""

    good = make_record(1)

    response = client.post("/api/predict/batch", json={"records": [good, {"f00": 80}]})

    assert response.status_code == 400
    assert "Record 1" in response.get_json()["error"]


def test_non_object_record_is_rejected(client):
    good = make_record(1)

    response = client.post("/api/predict/batch", json={"records": [good, "nope"]})

    assert response.status_code == 400
    assert "Record 1" in response.get_json()["error"]


def test_oversized_batch_is_rejected(client):
    records = [make_record(1)] * (netsentry.MAX_BATCH_RECORDS + 1)

    response = client.post("/api/predict/batch", json={"records": records})

    assert response.status_code == 413


def test_single_prediction_endpoint_is_unchanged(client):
    """The existing endpoint must keep its full per-record response shape."""

    response = client.post("/api/predict", json={"features": make_record(3)})

    assert response.status_code == 200

    body = response.get_json()

    for field in ("prediction", "is_attack", "confidence", "risk_level", "models",
                  "thresholds", "features", "probabilities"):
        assert field in body
