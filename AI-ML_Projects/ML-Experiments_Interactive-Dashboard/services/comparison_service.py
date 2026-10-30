import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from services import training_service
from services import evaluation_service as evals
from services import visualization_service as viz

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"

COMPARISON_GROUPS = {
    "banknote_svm": {
        "title": "Banknote Authenticity: Linear SVM vs. Nonlinear RBF SVM",
        "category": "Classification",
        "description": "Compares linear maximum-margin hyperplane partitioning against Gaussian RBF kernel projection on continuous wavelet features.",
        "models": [
            {"key": "linear_svm", "name": "Linear SVM (C=1.0)", "dataset": "data_banknote_authentication.csv"},
            {"key": "rbf_svm", "name": "Nonlinear SVM (RBF Kernel)", "dataset": "banknote_authentication_nonlinear_svm.csv"}
        ],
        "metric_keys": ["accuracy", "precision", "recall", "f1_score"]
    },
    "placement_regression": {
        "title": "Placement Compensation: Simple vs. Multiple Linear Regression",
        "category": "Regression",
        "description": "Compares ordinary least-squares regression with single feature (CGPA) against joint multi-variable feature modeling (CGPA + IQ score).",
        "models": [
            {"key": "slr", "name": "Simple Linear Regression (CGPA only)", "dataset": "Placement.csv"},
            {"key": "mlr", "name": "Multiple Linear Regression (CGPA + IQ)", "dataset": "Cgpa_iq_package.csv"}
        ],
        "metric_keys": ["r2_score", "mae", "rmse"]
    },
    "clinical_decision": {
        "title": "Clinical Diagnostic Risk: Decision Tree vs. Random Forest",
        "category": "Classification",
        "description": "Compares a single interpretable decision tree with an ensemble of de-correlated bagged decision trees on diagnostic screening data.",
        "models": [
            {"key": "heart", "name": "Decision Tree (Heart Disease)", "dataset": "heart_disease_dataset.csv"},
            {"key": "diabetes", "name": "Random Forest (Diabetes)", "dataset": "diabetes.csv"}
        ],
        "metric_keys": ["accuracy", "precision", "recall", "f1_score"],
        "note": "Evaluated on their respective verified clinical diagnostic benchmarks. Metrics reflect individual task performance."
    }
}

def get_comparison_data(group_key="banknote_svm"):
    if group_key not in COMPARISON_GROUPS:
        group_key = "banknote_svm"

    group = COMPARISON_GROUPS[group_key]
    comparison_results = []

    for item in group["models"]:
        key = item["key"]
        run = training_service.get_latest_run(key)
        if not run:
            run = training_service.run_experiment_pipeline(key)

        m = run["metrics"]
        comparison_results.append({
            "key": key,
            "name": item["name"],
            "dataset": item["dataset"],
            "metrics": m
        })

    # Build Comparison Chart
    fig = go.Figure()
    if group["category"] == "Regression":
        for res in comparison_results:
            fig.add_trace(go.Bar(
                name=res["name"],
                x=["R² Score", "MAE", "RMSE"],
                y=[res["metrics"].get("r2_score", 0), res["metrics"].get("mae", 0), res["metrics"].get("rmse", 0)],
                text=[res["metrics"].get("r2_score", 0), res["metrics"].get("mae", 0), res["metrics"].get("rmse", 0)],
                textposition="auto"
            ))
        fig.update_layout(
            barmode="group",
            title=f"Performance Comparison: {group['title']}",
            yaxis=dict(title="Score / Error Magnitude"),
            height=360,
            **viz.THEME_LAYOUT
        )
    else:
        for res in comparison_results:
            m = res["metrics"]
            fig.add_trace(go.Bar(
                name=res["name"],
                x=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"],
                y=[m.get("accuracy", 0), m.get("precision", 0), m.get("recall", 0), m.get("f1_score", 0)],
                text=[f"{m.get('accuracy', 0)}%", f"{m.get('precision', 0)}%", f"{m.get('recall', 0)}%", f"{m.get('f1_score', 0)}%"],
                textposition="auto"
            ))
        fig.update_layout(
            barmode="group",
            title=f"Performance Comparison: {group['title']}",
            yaxis=dict(title="Percentage (%)", range=[0, 110]),
            height=360,
            **viz.THEME_LAYOUT
        )

    return {
        "group_key": group_key,
        "title": group["title"],
        "category": group["category"],
        "description": group["description"],
        "results": comparison_results,
        "chart_json": json.loads(fig.to_json()),
        "note": group.get("note", "")
    }

def run_custom_comparison(dataset_filename, target_col, feature_cols, selected_algorithms, test_size=0.2, random_state=42):
    """
    Executes a scientifically fair multi-model comparison on the exact same dataset,
    features, train/test split, and random seed.
    """
    file_path = DATASET_DIR / dataset_filename
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_filename} not found.")

    df = pd.read_csv(file_path)
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]].dropna(subset=[target_col])

    if len(feature_cols) == 0:
        feature_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Determine if problem is regression or classification
    is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > 10

    # Handle missing values in features
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Train/test split with zero leakage
    stratify = y if (not is_regression and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y.to_numpy(), test_size=test_size, random_state=random_state, stratify=stratify
    )

    results = []

    for algo_id in selected_algorithms:
        start_t = time.perf_counter()
        pipeline = None
        algo_name = algo_id

        if is_regression:
            if algo_id == "linear_regression":
                pipeline = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
                algo_name = "Linear Regression"
            elif algo_id == "random_forest_regressor":
                pipeline = Pipeline([("model", RandomForestRegressor(n_estimators=100, random_state=random_state))])
                algo_name = "Random Forest Regressor"
            else:
                continue

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)
            m = evals.evaluate_regression(y_test, y_pred)
            results.append({
                "id": algo_id,
                "name": algo_name,
                "metrics": m,
                "duration_ms": elapsed_ms
            })

        else:
            # Classification algorithms
            classes = sorted(list(set(y_test)))
            if algo_id == "logistic_regression":
                pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=random_state, max_iter=1000))])
                algo_name = "Logistic Regression"
            elif algo_id == "decision_tree":
                pipeline = Pipeline([("model", DecisionTreeClassifier(max_depth=5, random_state=random_state))])
                algo_name = "Decision Tree (Gini)"
            elif algo_id == "knn":
                pipeline = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))])
                algo_name = "K-Nearest Neighbors (k=5)"
            elif algo_id == "random_forest":
                pipeline = Pipeline([("model", RandomForestClassifier(n_estimators=100, random_state=random_state))])
                algo_name = "Random Forest Classifier"
            elif algo_id == "linear_svm":
                pipeline = Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="linear", C=1.0, probability=True, random_state=random_state))])
                algo_name = "Linear SVM"
            elif algo_id == "rbf_svm":
                pipeline = Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state))])
                algo_name = "Nonlinear RBF SVM"
            else:
                continue

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
            elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)
            m = evals.evaluate_classification(y_test, y_pred, y_prob=y_prob, classes=classes)
            results.append({
                "id": algo_id,
                "name": algo_name,
                "metrics": m,
                "duration_ms": elapsed_ms
            })

    # Sort results by primary metric (F1-score for classification, R² for regression)
    if is_regression:
        results = sorted(results, key=lambda x: x["metrics"].get("r2_score", -999), reverse=True)
    else:
        results = sorted(results, key=lambda x: x["metrics"].get("f1_score", 0), reverse=True)

    # Build Comparison Chart
    fig = go.Figure()
    if is_regression:
        for res in results:
            fig.add_trace(go.Bar(
                name=res["name"],
                x=["R² Score", "MAE", "RMSE"],
                y=[res["metrics"].get("r2_score", 0), res["metrics"].get("mae", 0), res["metrics"].get("rmse", 0)],
                text=[res["metrics"].get("r2_score", 0), res["metrics"].get("mae", 0), res["metrics"].get("rmse", 0)],
                textposition="auto"
            ))
        fig.update_layout(
            barmode="group",
            title=f"Custom Benchmark Comparison: {dataset_filename} (Target: {target_col})",
            yaxis=dict(title="Metric Score"),
            height=380,
            **viz.THEME_LAYOUT
        )
    else:
        for res in results:
            m = res["metrics"]
            fig.add_trace(go.Bar(
                name=res["name"],
                x=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"],
                y=[m.get("accuracy", 0), m.get("precision", 0), m.get("recall", 0), m.get("f1_score", 0)],
                text=[f"{m.get('accuracy', 0)}%", f"{m.get('precision', 0)}%", f"{m.get('recall', 0)}%", f"{m.get('f1_score', 0)}%"],
                textposition="auto"
            ))
        fig.update_layout(
            barmode="group",
            title=f"Custom Benchmark Comparison: {dataset_filename} (Target: {target_col})",
            yaxis=dict(title="Percentage (%)", range=[0, 110]),
            height=380,
            **viz.THEME_LAYOUT
        )

    return {
        "dataset": dataset_filename,
        "target": target_col,
        "is_regression": is_regression,
        "results": results,
        "chart_json": json.loads(fig.to_json())
    }
