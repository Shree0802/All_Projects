import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
RETRAIN_DIR = BASE_DIR / "models" / "retrained"
RETRAIN_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_CONFIGS = {
    "slr": {
        "title": "Simple Linear Regression",
        "category": "Regression",
        "dataset_file": "Placement.csv",
        "target": "package",
        "features": ["cgpa"],
        "model_type": "LinearRegression",
        "description": "Trained on CGPA to predict placement package (LPA)."
    },
    "mlr": {
        "title": "Multiple Linear Regression",
        "category": "Regression",
        "dataset_file": "Cgpa_iq_package.csv",
        "target": "package",
        "features": ["cgpa", "iq"],
        "model_type": "LinearRegression",
        "description": "Trained on CGPA and IQ to predict placement package (LPA)."
    },
    "placement": {
        "title": "Logistic Regression: Binary Classification",
        "category": "Classification",
        "dataset_file": "Placement_Y_N.csv",
        "target": "placement",
        "features": ["cgpa", "iq"],
        "model_type": "LogisticRegression",
        "description": "Binary classification predicting placement likelihood from CGPA and IQ."
    },
    "heart": {
        "title": "Decision Tree Classification",
        "category": "Classification",
        "dataset_file": "heart_disease_dataset.csv",
        "target": "target",
        "features": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
        "model_type": "DecisionTreeClassifier",
        "description": "Gini-based Decision Tree classifier on 13 clinical cardiovascular features."
    },
    "breast": {
        "title": "K-Nearest Neighbors (KNN)",
        "category": "Classification",
        "dataset_file": "breast-cancer.csv",
        "target": "diagnosis",
        "features": [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
            "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
            "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
            "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
        ],
        "model_type": "KNeighborsClassifier",
        "description": "KNN classifier on 30 diagnostic cell measurements."
    },
    "message": {
        "title": "Naive Bayes Classification",
        "category": "Classification",
        "dataset_file": "college_messages_222.csv",
        "target": "label",
        "features": ["message"],
        "model_type": "CountVectorizer + MultinomialNB",
        "description": "Text vectorization and Multinomial Naive Bayes pipeline on college notifications."
    },
    "diabetes": {
        "title": "Random Forest Classification",
        "category": "Classification",
        "dataset_file": "diabetes.csv",
        "target": "Outcome",
        "features": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        "model_type": "RandomForestClassifier",
        "description": "Ensemble Random Forest classifier on 8 diabetes diagnostic attributes."
    },
    "glucose": {
        "title": "Random Forest Regression",
        "category": "Regression",
        "dataset_file": "diabetes.csv",
        "target": "Glucose",
        "features": ["Pregnancies", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        "model_type": "RandomForestRegressor",
        "description": "Ensemble Random Forest regressor estimating blood glucose levels from demographic/clinical factors."
    },
    "linear_svm": {
        "title": "Linear Support Vector Machine (SVM)",
        "category": "Classification",
        "dataset_file": "data_banknote_authentication.csv",
        "target": "class",
        "features": ["variance", "skewness", "curtosis", "entropy"],
        "model_type": "Linear SVM (SVC)",
        "description": "Linear maximum-margin Support Vector Machine classifying banknote authenticity."
    },
    "rbf_svm": {
        "title": "Nonlinear Support Vector Machine (RBF SVM)",
        "category": "Classification",
        "dataset_file": "banknote_authentication_nonlinear_svm.csv",
        "target": "class",
        "features": ["variance", "skewness", "curtosis", "entropy"],
        "model_type": "RBF Kernel SVM (SVC)",
        "description": "Radial Basis Function (RBF) nonlinear SVM classifier for complex decision boundaries."
    }
}

def get_dataset_inspection(key):
    """Returns dataset summary including row count, col count, missing values, and top records."""
    if key not in TRAINING_CONFIGS:
        return {"error": f"No training configuration for experiment '{key}'."}

    cfg = TRAINING_CONFIGS[key]
    csv_path = DATASET_DIR / cfg["dataset_file"]
    if not csv_path.exists():
        return {"error": f"Dataset file '{cfg['dataset_file']}' not found."}

    df = pd.read_csv(csv_path)
    # Check for unnamed index columns
    cleaned_cols = [c for c in df.columns if not c.startswith("Unnamed")]
    df_clean = df[cleaned_cols]

    preview_rows = df_clean.head(5).to_dict(orient="records")
    missing_counts = df_clean.isnull().sum().to_dict()

    return {
        "key": key,
        "title": cfg["title"],
        "category": cfg["category"],
        "dataset_file": cfg["dataset_file"],
        "rows": len(df_clean),
        "columns": len(df_clean.columns),
        "column_names": list(df_clean.columns),
        "target": cfg["target"],
        "features": cfg["features"],
        "missing_values": int(df_clean.isnull().sum().sum()),
        "missing_per_col": missing_counts,
        "preview": preview_rows,
        "model_type": cfg["model_type"],
        "description": cfg["description"]
    }

def train_and_evaluate(key, test_size=0.2, random_state=42, hyperparams=None):
    """
    Splits the genuine dataset into train/test sets, fits the model on train,
    evaluates strictly on the held-out test set, and returns verified metrics.
    """
    if key not in TRAINING_CONFIGS:
        raise ValueError(f"Experiment '{key}' is not configurable for dynamic training.")

    cfg = TRAINING_CONFIGS[key]
    csv_path = DATASET_DIR / cfg["dataset_file"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset {cfg['dataset_file']} not found.")

    df = pd.read_csv(csv_path)
    hyperparams = hyperparams or {}

    is_regression = cfg["category"] == "Regression"

    # Preprocessing / Feature Extraction
    if key == "message":
        X = df["message"].astype(str)
        y = df["label"].astype(str)
        # Stratified train-test split for text classification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        vec = CountVectorizer()
        clf = MultinomialNB()
        pipeline = Pipeline([("vectorizer", vec), ("classifier", clf)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        trained_artifact = pipeline
        classes = sorted(list(set(y_train)))
    elif key == "breast":
        y_raw = df["diagnosis"]
        # Map 'M'/'B' or string to numeric or keep classes
        X = df[cfg["features"]].astype(float)
        y = y_raw
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        n_neighbors = int(hyperparams.get("n_neighbors", 5))
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        trained_artifact = model
        classes = sorted(list(set(y_train)))
    else:
        # Numeric models
        X = df[cfg["features"]].astype(float)
        y = df[cfg["target"]]

        if is_regression:
            y = y.astype(float)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            if key in ("slr", "mlr"):
                model = LinearRegression()
            elif key == "glucose":
                n_est = int(hyperparams.get("n_estimators", 100))
                max_d = hyperparams.get("max_depth", None)
                if max_d: max_d = int(max_d)
                model = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=random_state)
            else:
                model = LinearRegression()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            trained_artifact = model
        else:
            # Classification
            y = y.astype(int) if pd.api.types.is_numeric_dtype(y) else y.astype(str)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            if key == "placement":
                model = LogisticRegression(random_state=random_state)
            elif key == "heart":
                max_d = hyperparams.get("max_depth", None)
                if max_d: max_d = int(max_d)
                model = DecisionTreeClassifier(criterion="gini", max_depth=max_d, random_state=random_state)
            elif key == "diabetes":
                n_est = int(hyperparams.get("n_estimators", 100))
                max_d = hyperparams.get("max_depth", None)
                if max_d: max_d = int(max_d)
                model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=random_state)
            elif key == "linear_svm":
                c_val = float(hyperparams.get("C", 1.0))
                model = SVC(kernel="linear", C=c_val, random_state=random_state)
            elif key == "rbf_svm":
                c_val = float(hyperparams.get("C", 1.0))
                model = SVC(kernel="rbf", C=c_val, gamma="scale", random_state=random_state)
            else:
                model = LogisticRegression(random_state=random_state)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            trained_artifact = model
            classes = sorted(list(set(y_train)))

    # Compute Verified Metrics
    if is_regression:
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        # Sample comparison data for plotting
        sample_indices = np.linspace(0, len(y_test) - 1, min(20, len(y_test)), dtype=int)
        chart_data = {
            "labels": [f"Test #{i+1}" for i in range(len(sample_indices))],
            "actual": [round(float(list(y_test)[i]), 2) for i in sample_indices],
            "predicted": [round(float(list(y_pred)[i]), 2) for i in sample_indices],
        }

        metrics = {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4),
        }
    else:
        acc = float(accuracy_score(y_test, y_pred))
        # Handle binary vs multiclass averaging
        avg_method = "binary" if len(classes) == 2 else "weighted"
        pos_label = classes[1] if len(classes) == 2 else None
        
        try:
            if pos_label is not None:
                prec = float(precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0))
                rec = float(recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0))
                f1 = float(f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0))
            else:
                prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
                rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
                f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        except Exception:
            prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()

        chart_data = {
            "classes": [str(c) for c in classes],
            "confusion_matrix": cm,
        }

        metrics = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "confusion_matrix": cm,
            "classes": [str(c) for c in classes],
        }

    training_summary = {
        "key": key,
        "title": cfg["title"],
        "category": cfg["category"],
        "model_type": cfg["model_type"],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_size_percent": int(test_size * 100),
        "metrics": metrics,
        "chart_data": chart_data,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return training_summary, trained_artifact

def save_retrained_model(key, trained_artifact):
    """Saves newly trained model to separate output directory models/retrained/ safely."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{key}_retrained_{timestamp}.joblib"
    save_path = RETRAIN_DIR / filename
    joblib.dump(trained_artifact, save_path)
    return str(save_path.name)
