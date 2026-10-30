# 🚀 Integrated Machine Learning Laboratory Dashboard
# 👉Live Link 🔗: https://ml-experiments-interactive-dashboard.onrender.com/

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![ReportLab](https://img.shields.io/badge/ReportLab-PDF%20Generator-E11D48?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)
![Deployment](https://img.shields.io/badge/Deployment-Ready%20(Render%20%2F%20Railway)-10B981?style=for-the-badge&logo=render&logoColor=white)

<br/>

**A Full-Stack, Production-Grade Academic Machine Learning Laboratory Studio**  
*Model Training • Scientific Visualizations • Explainable AI (XAI) • Fair Benchmarking • Automated PDF Lab Reports*

---

### 🎓 Academic Laboratory Mini-Project
**Department of Artificial Intelligence and Machine Learning**  
**D. Y. Patil College of Engineering and Technology, Kasaba Bawada, Kolhapur**

</div>

---

## 👨‍🎓 Student Project Profile

| Parameter | Specification |
| :--- | :--- |
| **Student Name** | **Prathamesh Balaji Patil** |
| **Class & Branch** | **TY A (Artificial Intelligence & Machine Learning)** |
| **Roll Number** | **A11** |
| **Course** | B.Tech - Artificial Intelligence and Machine Learning |
| **Subject** | Machine Learning (`ML`) / Machine Learning Laboratory |
| **College** | **D. Y. Patil College of Engineering and Technology (DYPCET)** |
| **Location** | Kasaba Bawada, Kolhapur, Maharashtra, India |
| **Email** | [shreepatil0802@gmail.com](mailto:shreepatil0802@gmail.com) |
| **Mobile** | +91 9021830410 |
| **GitHub Repo** | [Shree0802/ML-Experiments_Interactive-Dashboard](https://github.com/Shree0802/ML-Experiments_Interactive-Dashboard) |

---

## ⚡ Quick Start

### 💻 Local Run in 3 Steps

```bash
# 1. Clone the repository
git clone https://github.com/Shree0802/ML-Experiments_Interactive-Dashboard.git
cd ML-Experiments_Interactive-Dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the application
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser!

---

## 🧪 Comprehensive Laboratory Curriculum (12 Modules)

<details open>
<summary><b>Click to expand / collapse all 12 supported laboratory experiments</b></summary>
<br/>

| Module | Title | Category | Algorithm Architecture | Benchmark Dataset | Features & Output | Held-out Accuracy / $R^2$ |
| :---: | :--- | :---: | :--- | :--- | :--- | :---: |
| **01** | **Python Libraries & EDA** | Supporting | NumPy, Pandas, Plotly | All Lab CSVs | Feature distributions, correlation heatmaps, missing value audits | 100% Data Integrity |
| **02** | **Simple Linear Regression** | Regression | Ordinary Least Squares (OLS) | `Placement.csv` | `CGPA` &rarr; `Package (LPA)` | $R^2 = 0.7731$ |
| **03** | **Multiple Linear Regression** | Regression | OLS Hyperplane | `Cgpa_iq_package.csv` | `CGPA` + `IQ` &rarr; `Package (LPA)` | $R^2 = 0.8124$ |
| **04** | **Logistic Regression (Binary)** | Classification | Sigmoid Probabilistic Curve | `Placement_Y_N.csv` | `CGPA` + `IQ` &rarr; Placement Likelihood | $85.00\%$ Accuracy |
| **05** | **Multiclass Logistic Regression** | Classification | Multinomial Softmax | `account_.csv` | `Study_Hours` + `Attendance` &rarr; Stress Level | $70.00\%$ Accuracy |
| **06** | **Decision Tree Classification** | Classification | CART / Gini Impurity | `heart_disease_dataset.csv` | 13 Clinical Attributes &rarr; Heart Disease Risk | $78.69\%$ Accuracy |
| **07** | **K-Nearest Neighbors (KNN)** | Classification | Instance Lazy Learner ($K=5$) | `breast-cancer.csv` | 30 Biopsy Morphology Features &rarr; Benign / Malignant | $94.74\%$ Accuracy |
| **08** | **Naive Bayes Classification** | Classification | Multinomial NB + BoW | `college_messages_222.csv` | Text Token Frequencies &rarr; Notice Category | $91.11\%$ Accuracy |
| **09** | **Random Forest Classification** | Classification | Bagged Tree Ensemble ($B=100$) | `diabetes.csv` | 8 Metabolic Attributes &rarr; Diabetes Outcome | $76.62\%$ Accuracy |
| **10** | **Random Forest Regression** | Regression | Bagged Regression Trees | `diabetes_dataset.csv` | 7 Patient Vitals &rarr; Blood Glucose Level | $R^2 = 0.4812$ |
| **11** | **Linear Support Vector Machine** | Classification | Convex Quadratic Program (Max-Margin) | `data_banknote_authentication.csv` | 4 Continuous Wavelet Features &rarr; Genuine / Counterfeit | $98.55\%$ Accuracy |
| **12** | **Nonlinear Support Vector Machine** | Classification | Kernel Trick (Gaussian RBF) | `banknote_authentication_nonlinear_svm.csv` | 4 Continuous Wavelet Features &rarr; Genuine / Counterfeit | $99.64\%$ Accuracy |

</details>

---

## 🌟 Key Platform Capabilities

<details>
<summary><b>1. Interactive Model Training Studio (<code>/training</code>)</b></summary>
<br/>

* **Leakage-Free Preprocessing**: Automated `SimpleImputer` and `StandardScaler` transformations fitted exclusively on training splits.
* **Hyperparameter Tuning**: Dynamically adjust train/test ratios (10% to 40%), random seeds, decision tree depth, KNN neighbors, SVM regularization penalties ($C$), and kernels.
* **Custom Dataset Ingestion**: Safe CSV file uploader with size validation ($\le 10$ MB), column parsing, and preview inspection.
</details>

<details>
<summary><b>2. Scientific Visualization Studio (<code>/results/&lt;key&gt;</code>)</b></summary>
<br/>

* **Interactive Plotly Graphics**: Zoom, pan, hover tooltips, and SVG export.
* **Regression Charts**: Actual vs. Predicted scatter with ideal $y=x$ reference lines, residual error scatter, and error distribution histograms.
* **Classification Charts**: Confusion Matrix heatmaps with cell counts and percentages, ROC curves with computed AUC, Precision-Recall curves with Average Precision, and 2D decision boundary projections.
</details>

<details>
<summary><b>3. Explainable AI (XAI) Studio (<code>/explainability/&lt;key&gt;</code>)</b></summary>
<br/>

* **Global Feature Importance**: Native Tree Mean Decrease in Impurity (MDI), normalized linear model weights ($|\beta|$), and permutation importance.
* **Local Instance Attribution**:
  * Real-time **Plotly Waterfall / Diverging Bar Chart** displaying directional feature contributions (green $=$ positive push, red $=$ negative push).
  * **Step-by-Step Decision Tracing**: Internal node threshold comparisons for Decision Trees and Euclidean neighbor distances for KNN.
  * **Counterfactual "What-If" Analysis**: Actionable feedback indicating which feature adjustments flip or elevate the prediction.
* **3-Tier Educational Explanations**:
  1. *Tier 1 (Intuition)*: Real-world analogies for engineering students.
  2. *Tier 2 (Formulation)*: Mathematical equations and loss functions.
  3. *Tier 3 (Application)*: Practical dataset context and model limitations.
</details>

<details>
<summary><b>4. Academic Lab Report Generator (<code>/reports/&lt;key&gt;</code>)</b></summary>
<br/>

* **Formal Academic PDF Reports**: Built with ReportLab to generate publication-grade lab experiment reports containing:
  * Official DYPCET Institutional header and emblem.
  * Student credentials (Name, Roll No, Branch, Subject).
  * Mathematical definitions, governing formulas, and loss functions.
  * Held-out test evaluation metrics table.
  * Embedded high-resolution Matplotlib visualizations.
  * Student practical observations, conclusions, and signature blocks.
* **Export Options**: One-click download as **PDF** (`/api/export-pdf/<key>`), **JSON Metadata** (`/api/export-report/<key>?format=json`), or **CSV** (`/api/export-report/<key>?format=csv`).
</details>

<details>
<summary><b>5. Viva Voce & Learning Mode (<code>/learning/&lt;key&gt;</code>)</b></summary>
<br/>

* **Vector Architecture Diagrams**: Responsive inline SVG schematics for every algorithm.
* **Faculty Viva Voce Bank**: **60 in-depth viva questions** (5 per module) with rigorous model answers.
* **Practical Challenges**: Copyable Python implementation code snippets with one-click copy buttons.
</details>

---

## 🏛️ Project Directory Structure

```text
ML-Experiments_Interactive-Dashboard/
├── app.py                         # Primary Flask application controller & route endpoints
├── Procfile                       # Production process file for Cloud Deployments (Gunicorn)
├── requirements.txt               # Pinned Python package dependencies
├── test_suite.py                  # End-to-end automated test runner (127 verification tests)
├── README.md                      # Interactive project documentation
├── datasets/                      # 11 Clean laboratory CSV benchmark datasets
│   ├── Placement.csv
│   ├── Cgpa_iq_package.csv
│   ├── Placement_Y_N.csv
│   ├── heart_disease_dataset.csv
│   ├── breast-cancer.csv
│   ├── data_banknote_authentication.csv
│   └── ...
├── models/                        # Pre-trained scikit-learn models & estimators (.pkl, .joblib)
├── services/                      # Analytical and business logic services
│   ├── model_service.py           # Model loading, schemas, and live inference execution
│   ├── training_service.py        # Leakage-free training pipelines & hyperparameter grids
│   ├── evaluation_service.py      # Standard regression & classification test metrics
│   ├── visualization_service.py   # Plotly interactive chart generators
│   ├── explainability_service.py  # XAI attribution, waterfall charts & counterfactuals
│   ├── learning_service.py        # 12-module theory, formulas, workflows & viva Q&As
│   ├── report_service.py          # ReportLab PDF academic lab report generator
│   ├── dataset_service.py         # CSV inspection, statistics, and secure uploads
│   ├── comparison_service.py      # Multi-model benchmarking engine
│   └── history_service.py         # Audit logging and session tracking
├── static/                        # CSS styles, images, and college logo
│   ├── style.css                  # Custom responsive CSS design system (Dark/Light mode)
│   └── images/college-logo.png    # Official DYPCET college logo
└── templates/                     # Jinja2 HTML templates
    ├── base.html                  # Core layout with student profile & navigation sidebar
    ├── index.html                 # Workspace overview & KPI metrics
    ├── experiments.html           # Experiments catalog & search filters
    ├── algorithm.html             # Live prediction testing forms
    ├── dataset_explorer.html      # Dataset preview, summary stats & uploader
    ├── model_training.html        # Interactive training studio & hyperparameters
    ├── results.html               # Evaluation metrics & interactive charts
    ├── model_comparison.html      # Multi-model benchmark comparisons
    ├── explainability.html        # XAI studio, waterfall charts & what-if analysis
    ├── reports.html               # Lab report previewer & PDF download
    ├── learning.html              # Viva voce preparation & theory guides
    ├── history.html               # Audit trail of training runs
    └── error.html                 # Custom 404 & 500 error display
```

---

## 📡 REST API Reference

| Endpoint | Method | Description | Sample Request / Response |
| :--- | :---: | :--- | :--- |
| `/api/predict/<key>` | `POST` | Real-time live inference on pre-trained model | `{"cgpa": "8.5"}` &rarr; `{"prediction": "3.85 LPA"}` |
| `/api/train` | `POST` | Execute training pipeline with custom parameters | `{"key": "heart", "test_size": 0.2}` &rarr; `{"accuracy": 78.69}` |
| `/api/explain/<key>` | `POST` | Generate local instance attribution & waterfall chart | `{"cgpa": "8.5"}` &rarr; `{"waterfall_chart": {...}}` |
| `/api/compare` | `POST` | Fair comparison across multiple algorithms | `{"algorithms": ["logistic_regression", "linear_svm"]}` |
| `/api/export-pdf/<key>` | `GET` | Generate and download formal PDF Lab Report | Returns binary `application/pdf` stream |
| `/api/export-report/<key>` | `GET` | Export run metrics in JSON or CSV format | `?format=json` or `?format=csv` |
| `/api/dataset-preview/<key>`| `GET` | Retrieve dataset preview and column metadata | Returns rows, columns, feature names |
| `/api/upload-dataset` | `POST` | Safely upload and validate a custom CSV dataset | `multipart/form-data` file upload |
| `/api/save-profile` | `POST` | Update student profile information | `{"name": "Prathamesh Balaji Patil"}` |

---

## 🧪 Automated Testing & Verification

The project includes an automated test runner ([`test_suite.py`](test_suite.py)) that executes **127 automated checks** covering all navigation routes, inference forms, training pipelines, PDF compilation, and APIs.

To run tests:
```bash
python test_suite.py
```

```text
==================================================
ADVANCED ML EXPERIMENTS LAB - EXTENDED TEST SUITE
==================================================
--- 1. Testing Core Navigation GET Routes --- [14/14 OK]
--- 2. Testing All 11 Individual Prediction Form Routes (GET) --- [11/11 OK]
--- 3. Testing Real Model Predictions on All 11 Algorithms (POST) --- [11/11 OK]
--- 4. Testing Input Validation & Error Handling --- [22/22 OK]
--- 5. Testing Results & Visualizations Routes --- [11/11 OK]
--- 6. Testing Explainable AI (XAI) Routes --- [12/12 OK]
--- 7. Testing Academic Lab Reports Routes --- [11/11 OK]
--- 8. Testing Learning Mode Routes for All 12 Modules --- [12/12 OK]
--- 9. Testing REST API Endpoints --- [7/7 OK]
--- 10. Testing PDF Report Generation for All 11 Experiments --- [11/11 OK]
==================================================
ALL VERIFICATION SUITE CHECKS PASSED (100% SUCCESS)
==================================================
```

---

## 📜 Academic Integrity & License

This project was developed by **Prathamesh Balaji Patil** for the **Machine Learning Laboratory** practical course at **D. Y. Patil College of Engineering and Technology, Kolhapur**. 

Distributed under the **MIT License**. Feel free to use this dashboard for academic demonstrations, laboratory viva preparation, and educational exploration!

---

<div align="center">
  <b>Developed with ❤️ by Prathamesh Balaji Patil</b><br/>
  <i>TY A (AIML) · Roll No: A11 · DYPCET Kolhapur</i>
</div>
