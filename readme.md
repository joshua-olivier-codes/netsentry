# NETSENTRY

**NETSENTRY** is a machine-learning network intrusion detection API built with Python, Flask, and scikit-learn using the **CICIDS2017** network-flow dataset.

It uses a **two-stage Random Forest architecture**:

1. A binary classifier determines whether network traffic is `BENIGN` or `ATTACK`.
2. When traffic is classified as an attack, a multiclass classifier identifies the attack type.

The project is intentionally **backend/API only**. There is no frontend.

---

## Architecture

```text
                    CICIDS2017 CSV Files
                            |
                            v
                   src/preprocess.py
                            |
             Cleaning / Label Normalization
                            |
                            v
                 70 Production Features
                            |
              +-------------+-------------+
              |                           |
              v                           v
       Binary Random Forest        Multiclass Random Forest
              |                           |
       BENIGN / ATTACK              Attack classification
              |                           |
       Attack threshold: 0.55       15 known classes
              |                     Unknown threshold: 0.30
              +-------------+-------------+
                            |
                            v
                    Flask REST API
                   127.0.0.1:8086
```

### Two-stage detection

```text
Network Flow
     |
     v
Binary Classifier
     |
     +---- BENIGN ----> LOW risk
     |
     +---- ATTACK ----> Multiclass Classifier
                              |
                              +--> Known attack type
                              |
                              +--> UNKNOWN_UNSEEN
```

The production models use:

* **Random Forest**
* 200 trees
* `max_depth=25`
* `min_samples_leaf=2`
* `max_features="sqrt"`
* `class_weight="balanced_subsample"`
* `random_state=42`

---

# Dataset

NETSENTRY uses the **CICIDS2017** intrusion-detection dataset.

The project works with CICFlowMeter-derived network-flow records containing traffic statistics such as:

* destination ports
* packet counts
* packet lengths
* flow duration
* inter-arrival times
* packet rates
* TCP flags
* header lengths
* TCP window values
* active/idle statistics
* subflow statistics

The raw CSV files are stored under:

```text
data/raw/MachineLearningCVE/
```

`src/preprocess.py` performs the main preprocessing operations:

* trims dataset column names;
* normalizes labels;
* removes rows without labels;
* creates the binary `target` value;
* preserves the original attack label as `attack_type`;
* combines the source CSV files into the processed dataset.

The production training pipeline creates a stratified train/validation/test split and saves the production artifacts used by the API.

---

# Production Feature Schema

The production API expects exactly **70 numeric features**.

The production feature schema is stored in:

```text
models/feature_names_production.joblib
```

The multiclass schema is stored in:

```text
models/feature_names_multiclass.joblib
```

Both production schemas contain exactly **70 features**.

The API validates the required feature set before prediction and preserves the model-defined feature ordering.

The production feature set is:

```text
Destination Port
Flow Duration
Total Fwd Packets
Total Backward Packets
Total Length of Fwd Packets
Total Length of Bwd Packets
Fwd Packet Length Max
Fwd Packet Length Min
Fwd Packet Length Mean
Fwd Packet Length Std
Bwd Packet Length Max
Bwd Packet Length Min
Bwd Packet Length Mean
Bwd Packet Length Std
Flow Bytes/s
Flow Packets/s
Flow IAT Mean
Flow IAT Std
Flow IAT Max
Flow IAT Min
Fwd IATTotal
Fwd IAT Mean
Fwd IAT Std
Fwd IAT Max
Fwd IAT Min
Bwd IAT Total
Bwd IAT Mean
Bwd IAT Std
Bwd IATMax
Bwd IAT Min
Fwd PSH Flags
Fwd URG Flags
Fwd Header Length
Bwd Header Length
Fwd Packets/s
Bwd Packets/s
Min Packet Length
Max Packet Length
Packet Length Mean
Packet Length Std
Packet Length Variance
FIN Flag Count
SYN Flag Count
RST Flag Count
PSH Flag Count
ACK Flag Count
URG Flag Count
CWE Flag Count
ECE Flag Count
Down/Up Ratio
Average Packet Size
Avg Fwd Segment Size
Avg Bwd Segment Size
Fwd Header Length.1
Subflow Fwd Packets
Subflow Fwd Bytes
Subflow Bwd Packets
Subflow Bwd Bytes
Init_Win_bytes_forward
Init_Win_bytes_backward
act_data_pkt_fwd
min_seg_size_forward
Active Mean
Active Std
Active Max
Active Min
Idle Mean
Idle Std
Idle Max
Idle Min
```

Missing required features are rejected by the API.

Values that cannot be converted to finite numeric values cause feature preparation to fail rather than silently producing a prediction.

---

# Models

## Binary Intrusion Detector

Production model:

```text
models/intrusion_detector_production.joblib
```

Model schema:

```text
Input features: 70
Classes: 2
Classes: [0, 1]
```

The API maps these classes to:

```text
0 -> BENIGN
1 -> ATTACK
```

The attack probability threshold is:

```text
0.55
```

Therefore:

```text
P(ATTACK) >= 0.55
        |
        +----> ATTACK

P(ATTACK) < 0.55
        |
        +----> BENIGN
```

---

## Multiclass Attack Classifier

Production model:

```text
models/intrusion_detector_multiclass_production.joblib
```

The model contains 15 classes:

```text
BENIGN
Bot
DDoS
DoS GoldenEye
DoS Hulk
DoS Slowhttptest
DoS slowloris
FTP-Patator
Heartbleed
Infiltration
PortScan
SSH-Patator
Web Attack - Brute Force
Web Attack - Sql Injection
Web Attack - XSS
```

The multiclass classifier is used after the binary stage identifies traffic as an attack.

An attack type is considered unknown when:

* the multiclass model predicts `BENIGN`, despite the binary classifier detecting an attack; or
* the predicted attack type probability is below `0.30`.

In these cases the API can return:

```text
UNKNOWN_UNSEEN
```

---

# Model Evaluation

The production model evaluation is separate from the API smoke/validation tests.

The saved production evaluation should be treated as the primary measure of aggregate model performance.

## Binary Classification

Production holdout results:

| Metric           |   Score |
| ---------------- | ------: |
| Accuracy         | 0.99899 |
| Precision        | 0.99627 |
| Recall           | 0.99859 |
| F1               | 0.99743 |
| ROC AUC          | 0.99993 |
| Attack threshold |    0.55 |

Confusion matrix:

```text
                 Predicted
                 BENIGN   ATTACK

True BENIGN      454202      417
True ATTACK         157   111373
```

These results indicate very strong binary separation on the production holdout.

However, these are benchmark-dataset results and should not be interpreted as proof of equivalent performance on real-world network traffic.

---

## Multiclass Classification

Production holdout results:

| Metric             |   Score |
| ------------------ | ------: |
| Accuracy           | 0.99503 |
| Macro Precision    | 0.84681 |
| Macro Recall       | 0.85986 |
| Macro F1           | 0.80851 |
| Weighted Precision | 0.99725 |
| Weighted Recall    | 0.99503 |
| Weighted F1        | 0.99592 |

The difference between weighted and macro metrics is important.

The weighted metrics are strongly influenced by common classes, particularly `BENIGN`. Rare attack classes are substantially less reliable.

Examples include:

| Attack Type                |   F1 |
| -------------------------- | ---: |
| Bot                        | 0.38 |
| Web Attack - XSS           | 0.34 |
| Web Attack - Sql Injection | 0.33 |

Consequently, the multiclass **macro metrics and individual class metrics are more informative than overall accuracy when considering deployment**.

---

# API

The backend is implemented using Flask.

Start the API from the repository root:

```bash
python src/app.py
```

Default address:

```text
http://127.0.0.1:8086
```

The API is designed to run locally and does not include a frontend.

---

## Endpoints

| Method | Endpoint       | Description                     |
| ------ | -------------- | ------------------------------- |
| `GET`  | `/`            | Service information             |
| `GET`  | `/health`      | API/model health check          |
| `GET`  | `/api/model`   | Model and schema information    |
| `POST` | `/api/predict` | Predict one network-flow record |

---

## Health Check

```http
GET /health
```

A healthy response reports that:

* the binary model is loaded;
* the multiclass model is loaded;
* the binary feature count is 70;
* the multiclass feature count is 70;
* the API is healthy.

Example:

```json
{
  "status": "healthy",
  "binary_model_loaded": true,
  "multiclass_model_loaded": true,
  "binary_feature_count": 70,
  "multiclass_feature_count": 70,
  "binary_attack_threshold": 0.55,
  "unknown_type_threshold": 0.3
}
```

---

# Prediction Request

`POST /api/predict`

The request body must contain a `features` JSON object containing all required production features.

Example structure:

```json
{
  "features": {
    "Destination Port": 80,
    "Flow Duration": 1000,
    "Total Fwd Packets": 4,
    "Total Backward Packets": 3,
    "Total Length of Fwd Packets": 500,
    "Total Length of Bwd Packets": 400
  }
}
```

The example above is intentionally incomplete and will fail validation because the API requires all 70 production features.

A complete request can be generated from a CICIDS2017 flow record using the same feature schema used during training.

---

# Prediction Response

A successful attack prediction contains information from both stages of the pipeline.

Example response structure:

```json
{
  "prediction": "ATTACK",
  "is_attack": true,
  "confidence": 0.998,
  "confidence_percent": 99.8,
  "risk_level": "CRITICAL",
  "attack_type": "DoS Hulk",
  "attack_type_prediction": "DoS Hulk",
  "attack_type_confidence": 0.991,
  "models": {
    "binary": "intrusion_detector_production.joblib",
    "multiclass": "intrusion_detector_multiclass_production.joblib"
  },
  "thresholds": {
    "binary_attack": 0.55,
    "unknown_attack_type": 0.30
  }
}
```

The exact probability fields depend on the prediction returned by the trained models.

---

# Risk Levels

The API derives a risk level from the binary model confidence.

For attack predictions:

```text
>= 0.90  -> CRITICAL
>= 0.75  -> HIGH
>= 0.50  -> MEDIUM
```

For benign predictions:

```text
>= 0.90  -> LOW
>= 0.70  -> MEDIUM
<  0.70  -> HIGH
```

These risk levels are application-level interpretations of model probability and are **not calibrated guarantees of real-world risk**.

---

# Error Handling

The API validates malformed requests and missing features.

Examples include:

```text
Missing 'features' object
```

```text
'features' must be a JSON object
```

```text
Missing binary model features: ...
```

The final API validation confirmed that these error cases correctly returned HTTP `400`.

---

# Final API Validation

The final validation was executed against the running API:

```text
http://127.0.0.1:8086
```

Command:

```bash
python src/test_api_final.py
```

The validation checked:

1. API health
2. model information
3. binary/multiclass feature schemas
4. CICIDS2017 Friday traffic
5. representative traffic predictions
6. malformed request handling

## Architecture validation

```text
Binary features:     70
Multiclass features: 70
Binary-only features: 0
```

The production API therefore uses a consistent 70-feature schema for both stages.

---

## Representative Traffic Validation

The validation tested representative records from:

```text
BENIGN
DDoS
PortScan
Bot
```

Results:

| Traffic  | Expected | Prediction | Binary Confidence | Attack Type | Result |
| -------- | -------- | ---------- | ----------------: | ----------- | ------ |
| BENIGN   | BENIGN   | BENIGN     |           100.00% | BENIGN      | PASS   |
| DDoS     | ATTACK   | ATTACK     |           100.00% | DDoS        | PASS   |
| PortScan | ATTACK   | ATTACK     |           100.00% | PortScan    | PASS   |
| Bot      | ATTACK   | ATTACK     |            70.61% | Bot         | PASS   |

Final result:

```text
Traffic tests: 4/4
Error handling: PASS
Traffic validation accuracy: 100.00%

NETSENTRY API VALIDATION: PASS
```

The API also correctly identified:

```text
DDoS       -> DDoS
PortScan   -> PortScan
Bot        -> Bot
```

with the multiclass classifier.

For the tested `Bot` record, for example:

```text
Binary confidence:       70.61%
Attack type confidence:  99.22%
Risk level:              MEDIUM
```

This API validation is an end-to-end integration check using representative records. It is **not a replacement for the aggregate production holdout evaluation**.

---

# Training

Preprocess the raw CICIDS2017 data:

```bash
python src/preprocess.py
```

Train the production models:

```bash
python src/train_production.py
```

Evaluate the production holdout:

```bash
python src/evaluate_production.py
```

The repository includes trained production model artifacts, so retraining is not required simply to run the API.

---

# Testing

Start the API:

```bash
python src/app.py
```

Then run the available validation scripts:

```bash
python src/test_api.py
```

```bash
python src/test_api_attack_types.py
```

```bash
python src/test_api_final.py
```

Production validation:

```bash
python src/validate_production.py
```

Production evaluation:

```bash
python src/evaluate_production.py
```

---

# Repository Structure

```text
NETSENTRY/
|
|-- data/
|   |-- raw/
|   |   `-- MachineLearningCVE/
|   |
|   `-- processed/
|
|-- models/
|   |-- intrusion_detector_production.joblib
|   |-- intrusion_detector_multiclass_production.joblib
|   |-- feature_names_production.joblib
|   |-- feature_names_multiclass.joblib
|   `-- production_config.json
|
|-- reports/
|   `-- netsentry_api_validation.json
|
|-- src/
|   |-- app.py
|   |-- preprocess.py
|   |-- train.py
|   |-- train_production.py
|   |-- evaluate.py
|   |-- evaluate_production.py
|   |-- production_threshold_analysis.py
|   |-- threshold_analysis.py
|   |-- validate_production.py
|   |-- train_unseen_day.py
|   |-- test_api.py
|   |-- test_api_attack_types.py
|   `-- test_api_final.py
|
|-- requirements.txt
`-- README.md
```

---

# Technology Stack

* **Python**
* **Flask**
* **pandas**
* **NumPy**
* **scikit-learn**
* **Random Forest**
* **joblib**
* **CICIDS2017**

---

# Limitations

NETSENTRY is a machine-learning research/portfolio project and has several important limitations.

### Dataset limitations

CICIDS2017 is a benchmark dataset and does not necessarily represent:

* modern enterprise networks;
* current applications;
* encrypted traffic patterns;
* current attacker behavior;
* current network infrastructure.

### Evaluation limitations

The production evaluation uses a stratified split of pooled data. Similar traffic conditions can therefore exist across training and test partitions, potentially producing more optimistic results than a true time-separated or environment-separated evaluation.

### Class imbalance

The multiclass model performs much better on common classes than rare classes.

The high weighted F1 should therefore not be interpreted as equally strong performance across all attack categories.

### API limitations

The API currently processes one network-flow record per request.

It does not provide:

* packet capture;
* live traffic ingestion;
* flow extraction;
* authentication;
* authorization;
* rate limiting;
* persistent storage;
* alert delivery;
* frontend visualization.

### Probability limitations

Random Forest probabilities are model outputs, not guaranteed calibrated probabilities.

A confidence of 99% should not automatically be interpreted as a 99% probability that a real-world event is malicious.

---

# Security and Ethical Considerations

NETSENTRY should only be used with network traffic and systems for which the operator has authorization.

Network-flow records can contain sensitive operational information and should be protected appropriately.

Before exposing the API outside localhost, an operational deployment should consider:

* authentication;
* authorization;
* TLS;
* network access controls;
* rate limiting;
* logging safeguards;
* monitoring;
* input validation.

Predictions should support security investigation and defense-in-depth rather than automatically triggering high-impact actions without human review.

---

# Project Status

**NETSENTRY production API validation: PASS**

The current implementation has:

* a trained binary intrusion detector;
* a trained multiclass attack classifier;
* a consistent 70-feature production schema;
* configurable detection thresholds;
* unknown attack-type handling;
* a Flask REST API;
* model health checks;
* model information endpoints;
* input validation;
* malformed-request handling;
* production holdout evaluation;
* end-to-end API validation.

The project is **functionally complete as an ML/API portfolio project**.

Further work should focus on documentation, reproducibility, deployment hardening, and improved evaluation on temporally or operationally separated data rather than adding features solely for the sake of complexity.
