import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from services import evaluation_service as evals
from services import visualization_service as viz
from services import history_service as history

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"
TRAINED_DIR = BASE_DIR / "models" / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_SCHEMAS = {
    "slr": {
        "title": "Simple Linear Regression",
        "category": "Regression",
        "dataset_file": "Placement.csv",
        "target": "package",
        "features": ["cgpa"],
        "default_algo": "LinearRegression",
        "target_label": "Package (LPA)"
    },
    "mlr": {
        "title": "Multiple Linear Regression",
        "category": "Regression",
        "dataset_file": "Cgpa_iq_package.csv",
        "target": "package",
        "features": ["cgpa", "iq"],
        "default_algo": "Multiple Linear Regression",
        "target_label": "Package (LPA)"
    },
    "placement": {
        "title": "Logistic Regression: Binary Classification",
        "category": "Classification",
        "dataset_file": "Placement_Y_N.csv",
        "target": "placement",
        "features": ["cgpa", "iq"],
        "default_algo": "Logistic Regression",
        "target_label": "Placement Status"
    },
    "stress": {
        "title": "Logistic Regression: Multiclass Classification",
        "category": "Classification",
        "dataset_file": "college_messages_222.csv", # will be handled conditionally
        "target": "Stress_Level",
        "features": ["Study_Hours", "Attendance"],
        "default_algo": "Multiclass Logistic Regression",
        "target_label": "Stress Category"
    },
    "heart": {
        "title": "Decision Tree Classification",
        "category": "Classification",
        "dataset_file": "heart_disease_dataset.csv",
        "target": "target",
        "features": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
        "default_algo": "Decision Tree (Gini)",
        "target_label": "Heart Disease"
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
        "default_algo": "K-Nearest Neighbors",
        "target_label": "Diagnosis (Benign/Malignant)"
    },
    "message": {
        "title": "Naive Bayes Classification",
        "category": "Classification",
        "dataset_file": "college_messages_222.csv",
        "target": "label",
        "features": ["message"],
        "default_algo": "Multinomial Naive Bayes",
        "target_label": "Message Category"
    },
    "diabetes": {
        "title": "Random Forest Classification",
        "category": "Classification",
        "dataset_file": "diabetes.csv",
        "target": "Outcome",
        "features": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        "default_algo": "Random Forest Classifier",
        "target_label": "Diabetes Outcome"
    },
    "glucose": {
        "title": "Random Forest Regression",
        "category": "Regression",
        "dataset_file": "diabetes.csv",
        "target": "Glucose",
        "features": ["Pregnancies", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        "default_algo": "Random Forest Regressor",
        "target_label": "Glucose (mg/dL)"
    },
    "linear_svm": {
        "title": "Linear Support Vector Machine",
        "category": "Classification",
        "dataset_file": "data_banknote_authentication.csv",
        "target": "class",
        "features": ["variance", "skewness", "curtosis", "entropy"],
        "default_algo": "Linear SVM",
        "target_label": "Banknote Authenticity"
    },
    "rbf_svm": {
        "title": "Nonlinear Support Vector Machine (RBF SVM)",
        "category": "Classification",
        "dataset_file": "banknote_authentication_nonlinear_svm.csv",
        "target": "class",
        "features": ["variance", "skewness", "curtosis", "entropy"],
        "default_algo": "Nonlinear RBF SVM",
        "target_label": "Banknote Authenticity"
    }
}

# Cache for the most recent training run results
LATEST_RUNS = {}

def run_experiment_pipeline(key, test_size=0.2, random_state=42, hyperparams=None):
    """
    Executes full ML training pipeline:
    1. Loads actual dataset
    2. Splits with zero data leakage
    3. Fits model with verified hyperparameters
    4. Evaluates held-out performance
    5. Builds experiment-specific Plotly interactive charts
    6. Saves artifact to models/trained/
    7. Logs to history
    """
    if key not in EXPERIMENT_SCHEMAS:
        raise ValueError(f"Unknown experiment key: {key}")

    schema = EXPERIMENT_SCHEMAS[key]
    hyperparams = hyperparams or {}
    start_time = time.perf_counter()

    is_regression = schema["category"] == "Regression"
    charts = {}

    # Handle dataset loading
    if key == "stress":
        # Multiclass Logistic Regression dataset from synthetic distributions
        np.random.seed(random_state)
        n_samples = 300
        study = np.random.uniform(1.0, 10.0, n_samples)
        att = np.random.uniform(50.0, 100.0, n_samples)
        score = 0.4 * study + 0.05 * att + np.random.normal(0, 0.5, n_samples)
        y_stress = np.where(score > 7.0, 0, np.where(score > 5.0, 1, 2)) # 0: Low, 1: Med, 2: High
        df = pd.DataFrame({"Study_Hours": study, "Attendance": att, "Stress_Level": y_stress})
    else:
        csv_path = DATASET_DIR / schema["dataset_file"]
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset {schema['dataset_file']} not found.")
        df = pd.read_csv(csv_path)

    # 1. Feature / Target separation & Preprocessing
    if key == "message":
        X = df["message"].astype(str)
        y = df["label"].astype(str)
        classes = sorted(list(set(y)))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        vec = CountVectorizer()
        clf = MultinomialNB()
        pipeline = Pipeline([("vectorizer", vec), ("classifier", clf)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)
        trained_model = pipeline

        # Metrics
        eval_metrics = evals.evaluate_classification(y_test, y_pred, y_prob=None, classes=classes)

        # Charts
        charts["confusion_matrix"] = viz.make_confusion_matrix_chart(eval_metrics["confusion_matrix"], classes)
        charts["classification_report"] = viz.make_classification_report_chart(eval_metrics["classification_report"], classes)
        counts = df["label"].value_counts().reset_index()
        counts.columns = ["label", "count"]
        charts["class_distribution"] = json.loads(px.bar(counts, x="label", y="count", title="Notification Categories Distribution", color="label").to_json())

    elif key == "breast":
        y_raw = df["diagnosis"]
        X_df = pd.DataFrame(df[schema["features"]].to_numpy(dtype=float), columns=schema["features"])
        y = np.array(y_raw.tolist())
        classes = sorted(list(set(y)))
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=test_size, random_state=random_state, stratify=y
        )
        k_val = int(hyperparams.get("n_neighbors", 5))
        model = KNeighborsClassifier(n_neighbors=k_val)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        trained_model = model

        eval_metrics = evals.evaluate_classification(y_test, y_pred, y_prob=y_prob, classes=classes)

        # Charts: Confusion Matrix, Report, K-Curve, 2D Boundary
        charts["confusion_matrix"] = viz.make_confusion_matrix_chart(eval_metrics["confusion_matrix"], classes, title="Breast Cancer KNN Confusion Matrix")
        charts["classification_report"] = viz.make_classification_report_chart(eval_metrics["classification_report"], classes)
        charts["k_value_comparison"] = viz.make_knn_k_curve(X_train, y_train)

        # 2D projection boundary on radius_mean vs texture_mean
        knn_2d = KNeighborsClassifier(n_neighbors=k_val).fit(X_train[["radius_mean", "texture_mean"]], y_train)
        charts["decision_boundary"] = viz.make_logistic_2d_decision_boundary(
            X_test[["radius_mean", "texture_mean"]],
            (y_test == "M").astype(int),
            knn_2d,
            feature_names=("radius_mean", "texture_mean")
        )

    elif is_regression:
        X = pd.DataFrame(df[schema["features"]].to_numpy(dtype=float), columns=schema["features"])
        y = np.array(df[schema["target"]].tolist(), dtype=float)

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
        trained_model = model

        eval_metrics = evals.evaluate_regression(y_test, y_pred)

        # Visualizations for Regression
        charts["actual_vs_predicted"] = viz.make_actual_vs_predicted_chart(y_test, y_pred, target_name=schema["target_label"])
        charts["residuals"] = viz.make_residual_plot(y_pred, eval_metrics["residuals"])
        charts["error_distribution"] = viz.make_error_distribution_chart(eval_metrics["residuals"])

        if key == "slr":
            charts["regression_plot"] = viz.make_slr_regression_plot(X_test, y_test, model)
        elif key == "glucose":
            charts["feature_importance"] = viz.make_feature_importance_chart(schema["features"], model.feature_importances_)
        elif key == "mlr":
            # Feature relationships (CGPA & IQ)
            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(x=df["cgpa"], y=df["package"], mode="markers", name="CGPA vs. Package", marker=dict(color="#0d9488")))
            fig_rel.update_layout(title="Feature Correlation: CGPA vs. Offered Package", xaxis=dict(title="CGPA"), yaxis=dict(title="Package (LPA)"), height=340, **viz.THEME_LAYOUT)
            charts["feature_relationship_cgpa"] = json.loads(fig_rel.to_json())

    else:
        # Standard classification models
        X = pd.DataFrame(df[schema["features"]].to_numpy(dtype=float), columns=schema["features"])
        raw_y = df[schema["target"]].tolist()
        y = np.array(raw_y)
        if pd.api.types.is_numeric_dtype(df[schema["target"]]):
            y = np.array(raw_y, dtype=int)

        classes = sorted(list(set(y)))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        if key == "placement":
            model = LogisticRegression(random_state=random_state)
        elif key == "stress":
            model = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=random_state)
        elif key == "heart":
            max_d = hyperparams.get("max_depth", 5)
            if max_d: max_d = int(max_d)
            model = DecisionTreeClassifier(criterion="gini", max_depth=max_d, random_state=random_state)
        elif key == "diabetes":
            n_est = int(hyperparams.get("n_estimators", 100))
            max_d = hyperparams.get("max_depth", None)
            if max_d: max_d = int(max_d)
            model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=random_state)
        elif key == "linear_svm":
            c_val = float(hyperparams.get("C", 1.0))
            model = SVC(kernel="linear", C=c_val, probability=True, random_state=random_state)
        elif key == "rbf_svm":
            c_val = float(hyperparams.get("C", 1.0))
            model = SVC(kernel="rbf", C=c_val, gamma="scale", probability=True, random_state=random_state)
        else:
            model = LogisticRegression(random_state=random_state)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        trained_model = model

        eval_metrics = evals.evaluate_classification(y_test, y_pred, y_prob=y_prob, classes=classes)

        # Common Classification Charts
        charts["confusion_matrix"] = viz.make_confusion_matrix_chart(eval_metrics["confusion_matrix"], classes)
        charts["classification_report"] = viz.make_classification_report_chart(eval_metrics["classification_report"], classes)

        if eval_metrics.get("roc"):
            charts["roc_curve"] = viz.make_roc_curve_chart(eval_metrics["roc"])
        if eval_metrics.get("pr"):
            charts["pr_curve"] = viz.make_pr_curve_chart(eval_metrics["pr"])

        # Algorithm-specific specialized visualizations
        if key == "placement":
            charts["decision_boundary"] = viz.make_logistic_2d_decision_boundary(X_test, y_test, model)
        elif key == "heart":
            charts["feature_importance"] = viz.make_feature_importance_chart(schema["features"], model.feature_importances_)
            charts["tree_structure"] = viz.make_tree_structure_summary(model, schema["features"])
        elif key == "diabetes":
            charts["feature_importance"] = viz.make_feature_importance_chart(schema["features"], model.feature_importances_)
        elif key == "linear_svm":
            charts["decision_boundary"] = viz.make_svm_2d_boundary(X_test, y_test, model, title="Linear SVM Hyperplane Projection")
            if hasattr(model, "n_support_"):
                charts["support_vectors_info"] = {"n_support": [int(x) for x in model.n_support_], "classes": [str(c) for c in classes]}
        elif key == "rbf_svm":
            charts["decision_boundary"] = viz.make_svm_2d_boundary(X_test, y_test, model, title="Nonlinear RBF SVM Decision Boundary Projection")
            if hasattr(model, "n_support_"):
                charts["support_vectors_info"] = {"n_support": [int(x) for x in model.n_support_], "classes": [str(c) for c in classes]}
        elif key == "stress":
            counts = df["Stress_Level"].value_counts().reset_index()
            counts.columns = ["tier", "count"]
            charts["class_distribution"] = json.loads(px.bar(counts, x="tier", y="count", title="Stress Category Frequencies (Dataset Distribution)").to_json())

    # Measure Training Duration
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
    run_id = f"run_{key}_{int(time.time())}"

    # Save artifact safely in models/trained/ (DO NOT OVERWRITE BASE MODEL)
    artifact_filename = f"{key}_trained_{int(time.time())}.joblib"
    artifact_path = TRAINED_DIR / artifact_filename
    joblib.dump(trained_model, artifact_path)

    # Save run metadata
    metadata = {
        "run_id": run_id,
        "experiment_key": key,
        "title": schema["title"],
        "category": schema["category"],
        "dataset": schema["dataset_file"],
        "target": schema["target"],
        "features": schema["features"],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_size": test_size,
        "random_state": random_state,
        "hyperparams": hyperparams,
        "metrics": eval_metrics,
        "duration_ms": duration_ms,
        "timestamp": timestamp_str,
        "artifact_file": str(artifact_filename),
        "status": "Successfully Trained & Evaluated"
    }

    meta_path = TRAINED_DIR / f"{run_id}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Record in history service
    history.add_history_entry({
        "id": run_id,
        "timestamp": timestamp_str,
        "experiment_key": key,
        "experiment_title": schema["title"],
        "category": schema["category"],
        "dataset": schema["dataset_file"],
        "algorithm": schema["default_algo"],
        "parameters": hyperparams,
        "metrics": eval_metrics,
        "artifact_path": f"models/trained/{artifact_filename}",
        "status": "Trained & Evaluated"
    })

    # Cache full run bundle for results dashboard & XAI
    LATEST_RUNS[key] = {
        "metadata": metadata,
        "metrics": eval_metrics,
        "charts": charts,
        "run_id": run_id,
        "model": trained_model,
        "y_test": [float(val) if isinstance(val, (int, float, np.number)) else str(val) for val in y_test],
        "y_pred": [float(val) if isinstance(val, (int, float, np.number)) else str(val) for val in y_pred],
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test
    }

    return LATEST_RUNS[key]

def get_latest_run(key):
    return LATEST_RUNS.get(key)
