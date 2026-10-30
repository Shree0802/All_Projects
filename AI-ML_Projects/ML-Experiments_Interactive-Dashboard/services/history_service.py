import json
import os
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_FILE = BASE_DIR / "models" / "training_history.json"

def _ensure_history_file():
    if not HISTORY_FILE.exists():
        initial_records = [
            {
                "id": "base-slr",
                "timestamp": "Pre-trained baseline",
                "experiment_key": "slr",
                "experiment_title": "Simple Linear Regression",
                "category": "Regression",
                "dataset": "Placement.csv",
                "algorithm": "LinearRegression (OLS)",
                "parameters": {"fit_intercept": True},
                "metrics": {"r2_score": 0.7731, "mae": 0.2315, "rmse": 0.2901},
                "artifact_path": "models/salary_slr.pkl",
                "status": "Existing saved model",
                "is_baseline": True
            },
            {
                "id": "base-mlr",
                "timestamp": "Pre-trained baseline",
                "experiment_key": "mlr",
                "experiment_title": "Multiple Linear Regression",
                "category": "Regression",
                "dataset": "Cgpa_iq_package.csv",
                "algorithm": "Multiple Linear Regression",
                "parameters": {"features": ["cgpa", "iq"]},
                "metrics": {"r2_score": 0.8668, "mae": 0.2114, "rmse": 0.2642},
                "artifact_path": "models/package_mlr.pkl",
                "status": "Existing saved model",
                "is_baseline": True
            },
            {
                "id": "base-placement",
                "timestamp": "Pre-trained baseline",
                "experiment_key": "placement",
                "experiment_title": "Logistic Regression: Binary Classification",
                "category": "Classification",
                "dataset": "Placement_Y_N.csv",
                "algorithm": "LogisticRegression",
                "parameters": {"solver": "lbfgs"},
                "metrics": {"accuracy": 85.0, "precision": 85.71, "recall": 85.71, "f1_score": 85.71},
                "artifact_path": "models/placement_lr.pkl",
                "status": "Existing saved model",
                "is_baseline": True
            },
            {
                "id": "base-heart",
                "timestamp": "Pre-trained baseline",
                "experiment_key": "heart",
                "experiment_title": "Decision Tree Classification",
                "category": "Classification",
                "dataset": "heart_disease_dataset.csv",
                "algorithm": "DecisionTreeClassifier (Gini)",
                "parameters": {"criterion": "gini", "max_depth": 5},
                "metrics": {"accuracy": 70.49, "precision": 70.27, "recall": 78.79, "f1_score": 74.29},
                "artifact_path": "models/heart_dt.pkl",
                "status": "Existing saved model",
                "is_baseline": True
            },
            {
                "id": "base-banknote",
                "timestamp": "Pre-trained baseline",
                "experiment_key": "linear_svm",
                "experiment_title": "Linear Support Vector Machine",
                "category": "Classification",
                "dataset": "data_banknote_authentication.csv",
                "algorithm": "SVC (linear kernel)",
                "parameters": {"kernel": "linear", "C": 1.0},
                "metrics": {"accuracy": 98.55, "precision": 98.56, "recall": 98.55, "f1_score": 98.55},
                "artifact_path": "models/banknote_linear_svm.pkl",
                "status": "Existing saved model",
                "is_baseline": True
            }
        ]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(initial_records, f, indent=2)

def get_all_history():
    _ensure_history_file()
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def add_history_entry(entry):
    _ensure_history_file()
    records = get_all_history()
    if "id" not in entry:
        entry["id"] = f"run-{int(time.time()*1000)}"
    if "timestamp" not in entry:
        entry["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    entry["is_baseline"] = False
    records.insert(0, entry) # Most recent first
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return entry

def delete_history_entry(entry_id):
    """Deletes an experiment record from history and removes its artifact if safe."""
    _ensure_history_file()
    records = get_all_history()
    remaining = []
    deleted = False
    for r in records:
        if r.get("id") == entry_id:
            deleted = True
            # Check if artifact path is inside models/trained/ and remove
            artifact = r.get("artifact_path")
            if artifact and "models/trained" in str(artifact):
                p = BASE_DIR / artifact
                if p.exists() and p.is_file():
                    try:
                        p.unlink()
                    except Exception:
                        pass
        else:
            remaining.append(r)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(remaining, f, indent=2)
    return deleted

def get_history_by_id(entry_id):
    for item in get_all_history():
        if item.get("id") == entry_id:
            return item
    return None

def get_session_stats():
    records = get_all_history()
    session_train_count = sum(1 for r in records if not r.get("is_baseline", False))
    last_activity = records[0]["timestamp"] if len(records) > 0 else "None"
    return {
        "total_records": len(records),
        "session_train_count": session_train_count,
        "last_activity": last_activity
    }
