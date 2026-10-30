import os
import pickle
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "datasets"

STUDENT_PROFILE = {
    "name": "Prathamesh Balaji Patil",
    "college": "D. Y. Patil College of Engineering and Technology",
    "location": "Kasaba Bawada, Kolhapur, Maharashtra",
    "college_full": "D. Y. Patil College of Engineering and Technology, Kasaba Bawada, Kolhapur, Maharashtra",
    "course": "Artificial Intelligence and Machine Learning",
    "class": "TY A (AIML)",
    "year": "TY A (AIML)",
    "roll_no": "A11",
    "subject": "Machine Learning",
    "subject_lab": "Machine Learning Laboratory",
    "email": "shreepatil0802@gmail.com",
    "phone": "9021830410",
    "project_title": "Integrated Machine Learning Laboratory Dashboard",
    "project_type": "Machine Learning Laboratory Mini Project",
}

def _load_pickle(filename):
    with open(MODEL_DIR / filename, "rb") as f:
        return pickle.load(f)

# Load original base models safely
MODELS = {
    "slr": joblib.load(MODEL_DIR / "salary_slr.pkl"),
    "mlr": joblib.load(MODEL_DIR / "package_mlr.pkl"),
    "placement": _load_pickle("placement_lr.pkl"),
    "stress": _load_pickle("stress_lr.pkl"),
    "heart": _load_pickle("heart_dt.pkl"),
    "breast": joblib.load(MODEL_DIR / "breast_knn.joblib"),
    "nb_model": joblib.load(MODEL_DIR / "message_nb.pkl"),
    "nb_vectorizer": joblib.load(MODEL_DIR / "message_vectorizer.pkl"),
    "diabetes_bundle": joblib.load(MODEL_DIR / "diabetes_classifier.joblib"),
    "glucose_bundle": joblib.load(MODEL_DIR / "glucose_regressor.joblib"),
    "linear_svm": joblib.load(MODEL_DIR / "banknote_linear_svm.pkl"),
    "rbf_svm": joblib.load(MODEL_DIR / "banknote_rbf_svm.pkl"),
}
# Direct 1-to-1 algorithm key aliases
MODELS["message"] = MODELS["nb_model"]
MODELS["diabetes"] = MODELS["diabetes_bundle"]
MODELS["glucose"] = MODELS["glucose_bundle"]

HEART_FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
BREAST_FEATURES = [x for x in MODELS["breast"].feature_names_in_] if hasattr(MODELS["breast"], "feature_names_in_") else [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]
DIABETES_FEATURES = MODELS["diabetes_bundle"]["features"]
GLUCOSE_FEATURES = MODELS["glucose_bundle"]["features"]
STRESS_FEATURES = list(getattr(MODELS["stress"], "feature_names_in_", ["Study_Hours", "Attendance"]))
BANKNOTE_FEATURES = ["variance", "skewness", "curtosis", "entropy"]

ALGORITHMS = {
    "slr": {
        "num": "01",
        "title": "Simple Linear Regression",
        "category": "Regression",
        "description": "Predict a continuous output using a single input feature.",
        "input_feature": "CGPA",
        "model_type": "Linear Regression",
        "dataset_name": "Placement.csv",
        "saved_model": "salary_slr.pkl",
        "model_class": "sklearn.linear_model.LinearRegression",
        "expected_output": "Placement Package (LPA)",
        "supports_confidence": False,
        "objective": "Predict student placement package in Lakhs Per Annum (LPA) from undergraduate CGPA using ordinary least squares regression.",
        "fields": [("cgpa", "Undergraduate CGPA", "number", "0.01", "Enter CGPA between 0 and 10 (e.g., 8.5)")],
        "sample_data": {"cgpa": "8.5"}
    },
    "mlr": {
        "num": "02",
        "title": "Multiple Linear Regression",
        "category": "Regression",
        "description": "Predict a continuous output using multiple input features.",
        "input_feature": "CGPA and IQ",
        "model_type": "Multiple Linear Regression",
        "dataset_name": "Cgpa_iq_package.csv",
        "saved_model": "package_mlr.pkl",
        "model_class": "sklearn.linear_model.LinearRegression",
        "expected_output": "Placement Package (LPA)",
        "supports_confidence": False,
        "objective": "Estimate placement compensation by modeling the joint linear relationship between CGPA, IQ score, and offered package.",
        "fields": [
            ("cgpa", "Undergraduate CGPA", "number", "0.01", "e.g., 8.2"),
            ("iq", "IQ Score", "number", "1", "e.g., 110")
        ],
        "sample_data": {"cgpa": "8.2", "iq": "110"}
    },
    "placement": {
        "num": "03",
        "title": "Logistic Regression: Binary Classification",
        "category": "Classification",
        "description": "Classify an input into one of two classes using the saved binary logistic regression model.",
        "input_feature": "CGPA and IQ, based on the existing model.",
        "model_type": "Logistic Regression",
        "dataset_name": "Placement_Y_N.csv",
        "saved_model": "placement_lr.pkl",
        "model_class": "sklearn.linear_model.LogisticRegression",
        "expected_output": "Placement Likelihood (Likely / Not Likely)",
        "supports_confidence": True,
        "objective": "Map academic and cognitive features into a sigmoid probability curve to predict binary placement outcome.",
        "fields": [
            ("cgpa", "Undergraduate CGPA", "number", "0.01", "e.g., 7.8"),
            ("iq", "IQ Score", "number", "1", "e.g., 105")
        ],
        "sample_data": {"cgpa": "7.8", "iq": "105"}
    },
    "stress": {
        "num": "04",
        "title": "Logistic Regression: Multiclass Classification",
        "category": "Classification",
        "description": "Predict a category using the saved multiclass logistic regression model.",
        "input_feature": "Study Hours and Attendance",
        "model_type": "Multiclass Logistic Regression",
        "dataset_name": "Saved multiclass experiment dataset",
        "saved_model": "stress_lr.pkl",
        "model_class": "sklearn.pipeline.Pipeline (LogisticRegression)",
        "expected_output": "Stress / Performance Category (Low, Medium, High)",
        "supports_confidence": True,
        "objective": "Classify student performance and stress levels across 3 discrete tiers using multinomial logistic regression.",
        "fields": [
            ("Study_Hours", "Daily Study Hours", "number", "0.1", "e.g., 6.5"),
            ("Attendance", "Attendance Percentage", "number", "0.1", "e.g., 85.0")
        ],
        "sample_data": {"Study_Hours": "6.5", "Attendance": "85.0"}
    },
    "heart": {
        "num": "05",
        "title": "Decision Tree Classification",
        "category": "Classification",
        "description": "Classify inputs using a decision tree trained on the heart disease dataset.",
        "input_feature": "13 Clinical Attributes",
        "model_type": "Decision Tree (Gini Index)",
        "dataset_name": "heart_disease_dataset.csv",
        "saved_model": "heart_dt.pkl",
        "model_class": "sklearn.tree.DecisionTreeClassifier",
        "expected_output": "Heart Disease Detection (Class 0 / Class 1)",
        "supports_confidence": True,
        "disclaimer": "Educational demonstration only; this experiment is not a medical diagnostic tool.",
        "objective": "Hierarchically partition clinical cardiovascular attributes via Gini impurity to classify heart disease risk.",
        "fields": [(f, f.replace("_", " ").upper(), "number", "any", f"Enter {f.replace('_', ' ')}") for f in HEART_FEATURES],
        "sample_data": {
            "age": "57", "sex": "1", "cp": "2", "trestbps": "130", "chol": "236", "fbs": "0",
            "restecg": "0", "thalach": "174", "exang": "0", "oldpeak": "0.0", "slope": "1", "ca": "1", "thal": "2"
        }
    },
    "breast": {
        "num": "06",
        "title": "K-Nearest Neighbors (KNN)",
        "category": "Classification",
        "description": "Perform classification using the saved KNN model trained on the breast cancer dataset.",
        "input_feature": "30 Numeric Measurements",
        "model_type": "K-Nearest Neighbors",
        "dataset_name": "breast-cancer.csv",
        "saved_model": "breast_knn.joblib",
        "model_class": "sklearn.neighbors.KNeighborsClassifier",
        "expected_output": "Diagnosis (Malignant / Benign)",
        "supports_confidence": False,
        "disclaimer": "Educational demonstration only; this model is not designed for clinical diagnostic advice.",
        "objective": "Classify fine needle aspirate biopsy features into benign or malignant categories based on Euclidean distance to k-nearest training instances.",
        "fields": [(f, f.replace("_", " ").title(), "number", "any", f"Value for {f.replace('_', ' ')}") for f in BREAST_FEATURES],
        "sample_data": {
            "radius_mean": "17.99", "texture_mean": "10.38", "perimeter_mean": "122.8", "area_mean": "1001.0", "smoothness_mean": "0.1184",
            "compactness_mean": "0.2776", "concavity_mean": "0.3001", "concave points_mean": "0.1471", "symmetry_mean": "0.2419", "fractal_dimension_mean": "0.07871",
            "radius_se": "1.095", "texture_se": "0.9053", "perimeter_se": "8.589", "area_se": "153.4", "smoothness_se": "0.006399",
            "compactness_se": "0.04904", "concavity_se": "0.05373", "concave points_se": "0.01587", "symmetry_se": "0.03003", "fractal_dimension_se": "0.006193",
            "radius_worst": "25.38", "texture_worst": "17.33", "perimeter_worst": "184.6", "area_worst": "2019.0", "smoothness_worst": "0.1622",
            "compactness_worst": "0.6656", "concavity_worst": "0.7119", "concave points_worst": "0.2654", "symmetry_worst": "0.4601", "fractal_dimension_worst": "0.1189"
        }
    },
    "message": {
        "num": "07",
        "title": "Naive Bayes Classification",
        "category": "Classification",
        "description": "Classify text messages using the saved Naive Bayes model and its associated vectorizer.",
        "input_feature": "Message text string",
        "model_type": "Multinomial Naive Bayes",
        "dataset_name": "college_messages_222.csv",
        "saved_model": "message_nb.pkl + message_vectorizer.pkl",
        "model_class": "MultinomialNB + CountVectorizer",
        "expected_output": "Message Classification (e.g. Important, Promotional)",
        "supports_confidence": True,
        "objective": "Apply Bayes' theorem with naive feature independence assumptions over bag-of-words token frequencies to categorize campus notifications.",
        "fields": [("message", "Message Text Content", "text", "", "Type or paste a college announcement or notification...")],
        "sample_data": {
            "message": "Notice: Final semester Machine Learning laboratory practical exam schedule has been released. Please check the departmental notice board."
        }
    },
    "diabetes": {
        "num": "08",
        "title": "Random Forest Classification",
        "category": "Classification",
        "description": "Predict a diabetes-related class using the saved Random Forest classifier.",
        "input_feature": "8 Diagnostic Measurements",
        "model_type": "Random Forest Classifier",
        "dataset_name": "diabetes.csv",
        "saved_model": "diabetes_classifier.joblib",
        "model_class": "sklearn.pipeline.Pipeline (RandomForestClassifier)",
        "expected_output": "Outcome (Diabetes-Positive / Diabetes-Negative)",
        "supports_confidence": True,
        "disclaimer": "Educational demonstration only; not a clinical medical diagnostic evaluation.",
        "objective": "Aggregate predictions from an ensemble of de-correlated decision trees to assess diabetes risk.",
        "fields": [(f, f.replace("_", " "), "number", "any", f"Enter {f}") for f in DIABETES_FEATURES],
        "sample_data": {
            "Pregnancies": "2", "Glucose": "138", "BloodPressure": "62", "SkinThickness": "35",
            "Insulin": "0", "BMI": "33.6", "DiabetesPedigreeFunction": "0.127", "Age": "47"
        }
    },
    "glucose": {
        "num": "09",
        "title": "Random Forest Regression",
        "category": "Regression",
        "description": "Predict a continuous glucose-related value using the saved regression model.",
        "input_feature": "7 Clinical & Demographic Features",
        "model_type": "Random Forest Regressor",
        "dataset_name": "diabetes.csv",
        "saved_model": "glucose_regressor.joblib",
        "model_class": "sklearn.pipeline.Pipeline (RandomForestRegressor)",
        "expected_output": "Estimated Blood Glucose Level (mg/dL)",
        "supports_confidence": False,
        "objective": "Predict continuous blood glucose levels using an ensemble random forest regression model trained on remaining patient vitals.",
        "fields": [(f, f.replace("_", " "), "number", "any", f"Enter {f}") for f in GLUCOSE_FEATURES],
        "sample_data": {
            "Pregnancies": "2", "BloodPressure": "70", "SkinThickness": "25",
            "Insulin": "80", "BMI": "28.5", "DiabetesPedigreeFunction": "0.35", "Age": "32"
        }
    },
    "linear_svm": {
        "num": "10",
        "title": "Linear Support Vector Machine (SVM)",
        "category": "Classification",
        "description": "Classify banknote records using the saved linear SVM model.",
        "input_feature": "Variance, Skewness, Curtosis, Entropy",
        "model_type": "Linear Support Vector Machine",
        "dataset_name": "data_banknote_authentication.csv",
        "saved_model": "banknote_linear_svm.pkl",
        "model_class": "sklearn.svm.SVC (kernel='linear')",
        "expected_output": "Authenticity (Genuine / Counterfeit Banknote)",
        "supports_confidence": False,
        "objective": "Find an optimal hyper-plane with maximum margin in four-dimensional continuous wavelet transform space to classify banknotes.",
        "fields": [(f, f.title(), "number", "any", f"Enter wavelet {f}") for f in BANKNOTE_FEATURES],
        "sample_data": {
            "variance": "3.6216", "skewness": "8.6661", "curtosis": "-2.8073", "entropy": "-0.44699"
        }
    },
    "rbf_svm": {
        "num": "11",
        "title": "Nonlinear Support Vector Machine (RBF SVM)",
        "category": "Classification",
        "description": "Classify banknote records using the saved nonlinear SVM model with an RBF kernel.",
        "input_feature": "Variance, Skewness, Curtosis, Entropy",
        "model_type": "Nonlinear SVM (RBF Kernel)",
        "dataset_name": "banknote_authentication_nonlinear_svm.csv",
        "saved_model": "banknote_rbf_svm.pkl",
        "model_class": "sklearn.svm.SVC (kernel='rbf')",
        "expected_output": "Authenticity (Genuine / Counterfeit Banknote)",
        "supports_confidence": False,
        "objective": "Map wavelet transformed banknote features into an infinite-dimensional reproducing kernel Hilbert space using Gaussian RBF.",
        "fields": [(f, f.title(), "number", "any", f"Enter wavelet {f}") for f in BANKNOTE_FEATURES],
        "sample_data": {
            "variance": "-1.3971", "skewness": "3.3191", "curtosis": "-1.3927", "entropy": "-1.9948"
        }
    }
}

def extract_numeric_values(form_dict, keys):
    values = []
    for key in keys:
        raw = str(form_dict.get(key, "")).strip()
        if raw == "":
            raise ValueError(f"Please provide a value for '{key.replace('_', ' ')}'.")
        try:
            val = float(raw)
        except ValueError:
            raise ValueError(f"Invalid numeric input for '{key.replace('_', ' ')}': '{raw}'")
        if not np.isfinite(val):
            raise ValueError(f"Non-finite numeric value entered for '{key}'.")
        values.append(val)
    return values

def predict_single(key, form_data):
    """
    Executes live model prediction using original saved model artifacts.
    Preserves all feature orders, schemas, and preprocessing pipelines.
    """
    if key not in ALGORITHMS:
        raise ValueError(f"Unknown experiment algorithm '{key}'.")

    if key == "message":
        message = str(form_data.get("message", "")).strip()
        if not message:
            raise ValueError("Please enter message text to classify.")
        vec = MODELS["nb_vectorizer"].transform([message])
        pred = MODELS["nb_model"].predict(vec)[0]
        probs = MODELS["nb_model"].predict_proba(vec)[0]
        return {
            "prediction": f"Message Category: {pred}",
            "confidence": round(float(np.max(probs)) * 100, 2),
            "confidence_label": "Class Probability",
            "class_probabilities": {str(c): round(float(p) * 100, 2) for c, p in zip(MODELS["nb_model"].classes_, probs)}
        }
    elif key == "slr":
        x = extract_numeric_values(form_data, ["cgpa"])
        frame = pd.DataFrame([x], columns=["cgpa"])
        pred = float(MODELS["slr"].predict(frame)[0])
        return {"prediction": f"{pred:.2f} LPA", "confidence": None}
    elif key == "mlr":
        x = extract_numeric_values(form_data, ["cgpa", "iq"])
        frame = pd.DataFrame([x], columns=["cgpa", "iq"])
        pred = float(MODELS["mlr"].predict(frame)[0])
        return {"prediction": f"{pred:.2f} LPA", "confidence": None}
    elif key == "placement":
        x = extract_numeric_values(form_data, ["cgpa", "iq"])
        frame = pd.DataFrame([x], columns=["cgpa", "iq"])
        pred = int(MODELS["placement"].predict(frame)[0])
        labels = {0: "Not Likely to be Placed", 1: "Likely to be Placed"}
        probs = MODELS["placement"].predict_proba(frame)[0] if hasattr(MODELS["placement"], "predict_proba") else []
        conf = round(float(max(probs)) * 100, 2) if len(probs) else None
        return {
            "prediction": labels.get(pred, f"Class {pred}"),
            "confidence": conf,
            "class_probabilities": {labels.get(i, f"Class {i}"): round(float(p) * 100, 2) for i, p in enumerate(probs)} if len(probs) else None
        }
    elif key == "stress":
        x = extract_numeric_values(form_data, STRESS_FEATURES)
        frame = pd.DataFrame([x], columns=STRESS_FEATURES)
        pred = int(MODELS["stress"].predict(frame)[0])
        labels = {0: "Low Stress", 1: "Medium Stress", 2: "High Stress"}
        probs = MODELS["stress"].predict_proba(frame)[0] if hasattr(MODELS["stress"], "predict_proba") else []
        conf = round(float(max(probs)) * 100, 2) if len(probs) else None
        return {
            "prediction": labels.get(pred, str(pred)),
            "confidence": conf,
            "class_probabilities": {labels.get(i, f"Class {i}"): round(float(p) * 100, 2) for i, p in enumerate(probs)} if len(probs) else None
        }
    elif key == "heart":
        x = extract_numeric_values(form_data, HEART_FEATURES)
        frame = pd.DataFrame([x], columns=HEART_FEATURES)
        pred = int(MODELS["heart"].predict(frame)[0])
        probs = MODELS["heart"].predict_proba(frame)[0] if hasattr(MODELS["heart"], "predict_proba") else []
        conf = round(float(max(probs)) * 100, 2) if len(probs) else None
        pred_label = "Heart Disease Detected (Class 1)" if pred == 1 else "No Heart Disease Detected (Class 0)"
        return {
            "prediction": pred_label,
            "confidence": conf,
            "class_probabilities": {"No Heart Disease (0)": round(float(probs[0]) * 100, 2), "Heart Disease (1)": round(float(probs[1]) * 100, 2)} if len(probs) else None
        }
    elif key == "breast":
        x = extract_numeric_values(form_data, BREAST_FEATURES)
        pred = MODELS["breast"].predict(np.array([x]))[0]
        return {
            "prediction": "Malignant (Cancerous)" if str(pred).upper() in ("M", "1") else "Benign (Non-Cancerous)",
            "confidence": None
        }
    elif key == "diabetes":
        x = extract_numeric_values(form_data, DIABETES_FEATURES)
        frame = pd.DataFrame([x], columns=DIABETES_FEATURES)
        for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
            if c in frame.columns and frame.loc[0, c] == 0:
                frame.loc[0, c] = np.nan
        bundle = MODELS["diabetes_bundle"]
        clf = bundle["pipeline"]
        pred = int(clf.predict(frame)[0])
        probs = clf.predict_proba(frame)[0]
        pred_label = "Diabetes-Positive Outcome (Class 1)" if pred == 1 else "Diabetes-Negative Outcome (Class 0)"
        return {
            "prediction": pred_label,
            "confidence": round(float(max(probs)) * 100, 2),
            "class_probabilities": {"Negative (0)": round(float(probs[0]) * 100, 2), "Positive (1)": round(float(probs[1]) * 100, 2)}
        }
    elif key == "glucose":
        x = extract_numeric_values(form_data, GLUCOSE_FEATURES)
        frame = pd.DataFrame([x], columns=GLUCOSE_FEATURES)
        pred = float(MODELS["glucose_bundle"]["pipeline"].predict(frame)[0])
        return {"prediction": f"{pred:.1f} mg/dL", "confidence": None}
    elif key in ("linear_svm", "rbf_svm"):
        x = extract_numeric_values(form_data, BANKNOTE_FEATURES)
        frame = pd.DataFrame([x], columns=BANKNOTE_FEATURES)
        pred = int(MODELS[key].predict(frame)[0])
        return {
            "prediction": "Genuine Banknote (Class 0)" if pred == 0 else "Counterfeit Banknote (Class 1)",
            "confidence": None
        }
    else:
        raise ValueError(f"Prediction handler not implemented for {key}.")
