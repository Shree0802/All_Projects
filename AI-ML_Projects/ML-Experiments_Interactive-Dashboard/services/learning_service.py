"""
Academic Machine Learning Learning Resources & Examination Prep Module
Provides comprehensive theoretical foundations, mathematical intuition,
architecture workflows, viva questions with model answers, and practical tasks.
Covers all 12 Laboratory Curriculum Modules.
"""

LEARNING_DATA = {
    "exploration": {
        "exp_num": "01",
        "title": "Python Libraries & Dataset Exploration (EDA)",
        "category": "Supporting Module",
        "dataset": "All Laboratory CSV Files",
        "definition": "Exploratory Data Analysis (EDA) is the statistical process of performing initial investigations on datasets to discover patterns, detect anomalies, test hypotheses, and verify assumptions using summary statistics and graphical representations.",
        "math_formula": "Pearson Correlation: r_{xy} = \\frac{\\sum (x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum (x_i - \\bar{x})^2 \\sum (y_i - \\bar{y})^2}} \\in [-1, 1]",
        "cost_function": "Statistical profiling: Variance \\sigma^2 = \\frac{1}{N}\\sum (x_i - \\mu)^2, Interquartile Range IQR = Q_3 - Q_1",
        "math_intuition": "EDA relies on descriptive statistics to summarize sample properties. Central tendency measures (mean, median, mode) quantify the midpoint, dispersion metrics (standard deviation, IQR) measure data spread, and covariance/correlation measure linear dependency between pairs of continuous random variables. Outlier detection utilizes Tukey's fences: values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.",
        "workflow": [
            "1. Data Ingestion: Ingest raw CSV data into Pandas DataFrames and audit schema types.",
            "2. Data Integrity Audit: Quantify missing (null) values, invalid formats, and duplicate records.",
            "3. Univariate Analysis: Plot histograms, kernel density estimates (KDE), and box plots.",
            "4. Bivariate & Multivariate Analysis: Compute Pearson correlation heatmaps and pairwise scatter plots.",
            "5. Preprocessing Strategy: Select imputation strategy (mean/median/mode) and scaling policies."
        ],
        "use_cases": [
            "Data Quality Assessment prior to enterprise machine learning pipeline deployment.",
            "Feature Engineering & Collinearity screening in financial risk underwriting.",
            "Sensor telemetry anomaly detection in Internet of Things (IoT) hardware.",
            "Biomarker discovery across clinical pathology datasets."
        ],
        "assumptions": [
            "Sample observations are representative of the true underlying population distribution.",
            "Missingness mechanism is identifiable (MCAR, MAR, or MNAR).",
            "Linear correlation assumes bivariate normality between numerical continuous features.",
            "Measurement scales are consistent across rows without uncalibrated sensor drift."
        ],
        "advantages": [
            "Prevents Garbage-In-Garbage-Out by catching corrupt or missing data early.",
            "Exposes multicollinearity and redundant features before model training.",
            "Guides the correct choice of algorithm and preprocessing transformations.",
            "Completely library-supported across Python NumPy, Pandas, Matplotlib, Seaborn, and Plotly."
        ],
        "limitations": [
            "Correlation does not imply causation; spurious correlations can mislead modeling.",
            "Pearson correlation only captures linear relationships; non-linear associations may yield r ≈ 0.",
            "Summary statistics (mean, variance) can be drastically distorted by uncleaned outliers.",
            "High-dimensional datasets (p > 50) suffer from visual clutter in pair plots."
        ],
        "evaluation_metrics": [
            {"metric": "Missingness Rate (%)", "desc": "Percentage of null or NaN values per column."},
            {"metric": "Pearson Correlation (r)", "desc": "Linear relationship strength between -1.0 and +1.0."},
            {"metric": "Skewness & Kurtosis", "desc": "Measures asymmetry and tail heaviness relative to normal distribution."},
            {"metric": "Variance Inflation Factor (VIF)", "desc": "Quantifies severity of multicollinearity among predictors."}
        ],
        "common_mistakes": [
            "Imputing missing values using the whole dataset before train/test split, causing test data leakage.",
            "Blindly dropping rows with missing values without checking if missingness is systematic.",
            "Assuming zero correlation means zero dependence (e.g. y = x^2 has r ≈ 0 but perfect dependence).",
            "Applying StandardScaler on target variables when not strictly required."
        ],
        "how_prediction_works": "In exploratory analysis, given a feature vector X = [8.5, 7.8, 9.2, 6.4, 8.1], the sample mean is x̄ = 40.0 / 5 = 8.0. The sample variance is s² = Σ(x - 8.0)² / (5 - 1) = (0.25 + 0.04 + 1.44 + 2.56 + 0.01) / 4 = 4.30 / 4 = 1.075. The standard deviation is s = √1.075 ≈ 1.037. Z-score for a candidate with 9.2 is (9.2 - 8.0) / 1.037 = +1.157σ.",
        "viva_questions": [
            {
                "q": "What is the difference between Mean, Median, and Mode, and when should you prefer Median?",
                "a": "Mean is the arithmetic average, sensitive to extreme outliers. Median is the 50th percentile value, highly robust to outliers and skewed distributions. Mode is the most frequent value, suitable for categorical data. Median is preferred for skewed data such as income, package compensation, or house prices."
            },
            {
                "q": "What is Data Leakage during preprocessing, and how do you guarantee zero leakage?",
                "a": "Data leakage occurs when information from outside the training dataset (such as test set statistics) is used to fit models or scalers. To guarantee zero leakage, you must perform train/test split FIRST, compute imputations, scalers, and encoders strictly on the training fold, and transform the test fold using only the learned training parameters."
            },
            {
                "q": "What is the Interquartile Range (IQR) and how is it used in box plots to detect outliers?",
                "a": "IQR = Q3 - Q1, representing the middle 50% spread of the dataset. In standard box plots, whiskers extend to Q1 - 1.5*IQR (lower inner fence) and Q3 + 1.5*IQR (upper inner fence). Any data point beyond these boundaries is flagged as an outlier."
            },
            {
                "q": "What is the difference between Min-Max Scaling (Normalization) and Standardization (Z-score)?",
                "a": "Min-Max Scaling transforms values into [0, 1] via (x - x_min)/(x_max - x_min); it is bounded but vulnerable to extreme outliers. Standardization transforms values to mean 0 and variance 1 via (x - μ)/σ; it is unbounded, preserves outlier distances, and is ideal for linear models, SVMs, and neural networks."
            },
            {
                "q": "Why is Pearson correlation insufficient for non-linear relationships?",
                "a": "Pearson correlation measures only linear covariance. For non-linear relationships like y = x², the covariance is zero over a symmetric domain [-1, 1], yielding r = 0 despite a deterministic non-linear relationship. Spearman rank correlation or Mutual Information must be used instead."
            }
        ],
        "practical_task": "Load any project CSV dataset, compute descriptive statistics with df.describe(), identify duplicate rows with df.duplicated().sum(), and generate a correlation heatmap.",
        "code_snippet": "import pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\ndf = pd.read_csv('datasets/Placement.csv')\nprint(df.info())\nprint(df.describe())\nprint('Duplicates:', df.duplicated().sum())\n\nsns.heatmap(df.corr(), annot=True, cmap='Blues')\nplt.title('Correlation Heatmap')\nplt.show()"
    },
    "slr": {
        "exp_num": "02",
        "title": "Simple Linear Regression",
        "category": "Regression",
        "dataset": "Placement.csv",
        "definition": "A supervised parametric regression algorithm that models the linear relationship between a single continuous explanatory variable X and a continuous target variable Y using an Ordinary Least Squares (OLS) straight line.",
        "math_formula": "y = \\beta_0 + \\beta_1 x + \\varepsilon, \\quad \\beta_1 = \\frac{\\text{Cov}(x, y)}{\\text{Var}(x)} = \\frac{\\sum (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum (x_i - \\bar{x})^2}, \\quad \\beta_0 = \\bar{y} - \\beta_1 \\bar{x}",
        "cost_function": "\\text{Residual Sum of Squares (RSS)}: J(\\beta_0, \\beta_1) = \\sum_{i=1}^m (y_i - (\\beta_0 + \\beta_1 x_i))^2",
        "math_intuition": "OLS finds the unique line y = β₀ + β₁x that minimizes the vertical squared distances (residuals) from each training point to the line. Taking partial derivatives of RSS with respect to β₀ and β₁ and setting them to zero yields the closed-form normal equations, guaranteeing the global minimum without gradient descent.",
        "workflow": [
            "1. Feature Isolation: Extract independent feature X (e.g. CGPA) and continuous target Y (e.g. Package).",
            "2. Train/Test Partition: Split into 80% training and 20% held-out test data.",
            "3. OLS Estimation: Compute slope β₁ and intercept β₀ using training statistics.",
            "4. Test Inference: Compute predictions ŷ_test = β₀ + β₁ * x_test.",
            "5. Residual Evaluation: Calculate MAE, MSE, RMSE, and R² variance explained."
        ],
        "use_cases": [
            "Predicting campus salary packages based on undergraduate academic CGPA.",
            "Estimating fuel consumption based on vehicle engine displacement.",
            "Forecasting electricity demand as a linear function of ambient temperature.",
            "Real estate base valuation based solely on carpet area."
        ],
        "assumptions": [
            "Linearity: The true relationship between X and Y is strictly linear.",
            "Homoscedasticity: Residual error variance is constant across all values of X.",
            "Independence: Residuals are independent without autocorrelation.",
            "Normality of Residuals: Residuals ε are normally distributed with mean 0."
        ],
        "advantages": [
            "Extremely fast closed-form analytical solution O(N).",
            "Completely interpretable: β₁ directly gives the marginal return per unit increase.",
            "No hyperparameter tuning required.",
            "Serves as the foundational baseline for all linear models."
        ],
        "limitations": [
            "Cannot capture non-linear, polynomial, or saturation effects.",
            "Highly sensitive to high-leverage outliers which tilt the regression slope.",
            "Underfits complex multi-factorial systems.",
            "Prone to extrapolation error outside the domain of observed X values."
        ],
        "evaluation_metrics": [
            {"metric": "R² (Coefficient of Determination)", "desc": "Fraction of target variance explained by line. 1.0 is perfect."},
            {"metric": "MAE (Mean Absolute Error)", "desc": "Average absolute error magnitude in target units (LPA)."},
            {"metric": "RMSE (Root Mean Squared Error)", "desc": "Root of mean squared residuals; penalizes large errors."},
            {"metric": "MSE (Mean Squared Error)", "desc": "Variance of estimation residuals."}
        ],
        "common_mistakes": [
            "Confusing R² with correlation r: R² = r² only holds in univariate linear regression with an intercept.",
            "Extrapolating predictions far beyond the range of training data (e.g. CGPA = 15.0).",
            "Evaluating model accuracy on training data rather than held-out test data.",
            "Forgetting to check the residual scatter plot for heteroscedasticity (funnel shapes)."
        ],
        "how_prediction_works": "Suppose trained model has β₀ = -1.25 LPA and β₁ = 0.60 LPA/CGPA. For an undergraduate student with CGPA = 8.5: ŷ = -1.25 + 0.60 * (8.5) = -1.25 + 5.10 = 3.85 LPA. If actual placement was 4.10 LPA, the residual error is e = y - ŷ = 4.10 - 3.85 = +0.25 LPA.",
        "viva_questions": [
            {
                "q": "Why do we minimize squared residuals (L2) rather than absolute residuals (L1) in Ordinary Least Squares?",
                "a": "Squared residuals yield a smooth, strictly convex loss function whose derivatives are linear, allowing an exact closed-form analytical solution via normal equations without iterative gradient descent. Furthermore, under the Gauss-Markov theorem, OLS is the Best Linear Unbiased Estimator (BLUE)."
            },
            {
                "q": "What does a negative R² score signify?",
                "a": "A negative R² occurs when the model performs worse than a simple horizontal line predicting the mean of the target (ȳ). It indicates that the model is severely mis-specified, trained with bugs, or evaluated on an out-of-distribution test set."
            },
            {
                "q": "How does an outlier with high leverage affect the regression line?",
                "a": "An outlier with extreme X values (high leverage) has a disproportionate pulling effect on the slope β₁ because OLS squares residual errors. A single leverage point can tilt the slope, shifting predictions for all other valid observations."
            },
            {
                "q": "What is the geometric interpretation of R²?",
                "a": "R² = 1 - (SS_res / SS_tot). SS_tot measures the total squared dispersion around the mean ȳ. SS_res measures the squared error around the fitted line. R² represents the percentage reduction in error achieved by using the feature X instead of guessing the mean ȳ."
            },
            {
                "q": "What is Homoscedasticity and how do you test for it?",
                "a": "Homoscedasticity means the variance of residuals remains constant across all predicted values. It is tested visually by plotting residuals (y - ŷ) against predicted values ŷ. A random horizontal band confirms homoscedasticity; a cone or funnel shape indicates heteroscedasticity."
            }
        ],
        "practical_task": "Fit a LinearRegression model on Placement.csv, print coefficients, calculate R² on test data, and plot the regression line alongside actual test points.",
        "code_snippet": "from sklearn.linear_model import LinearRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import r2_score\nimport pandas as pd\n\ndf = pd.read_csv('datasets/Placement.csv')\nX_tr, X_te, y_tr, y_te = train_test_split(df[['cgpa']], df['package'], test_size=0.2, random_state=42)\nmodel = LinearRegression().fit(X_tr, y_tr)\nprint(f'Slope: {model.coef_[0]:.3f}, Intercept: {model.intercept_:.3f}')\nprint(f'Test R2: {r2_score(y_te, model.predict(X_te)):.4f}')"
    },
    "mlr": {
        "exp_num": "03",
        "title": "Multiple Linear Regression",
        "category": "Regression",
        "dataset": "Cgpa_iq_package.csv",
        "definition": "An extension of simple linear regression that models the joint linear relationship between two or more independent explanatory variables (X₁, X₂, ..., Xₚ) and a single continuous target variable Y via a multi-dimensional hyperplane.",
        "math_formula": "Y = X\\beta + \\varepsilon, \\quad \\hat{\\beta} = (X^T X)^{-1} X^T Y, \\quad y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + ... + \\beta_p x_p",
        "cost_function": "\\text{RSS}(\\beta) = (Y - X\\beta)^T (Y - X\\beta) = \\sum_{i=1}^m (y_i - \\hat{y}_i)^2",
        "math_intuition": "In multiple regression, the model fits a p-dimensional hyperplane through feature space. The coefficient βⱼ represents the partial derivative ∂y/∂xⱼ: the expected change in Y when feature xⱼ increases by 1 unit while holding all other p-1 features strictly constant. If features are collinear, XᵀX becomes singular and cannot be inverted stably.",
        "workflow": [
            "1. Multivariable Ingestion: Load CGPA and IQ features along with placement package target.",
            "2. Collinearity Check: Compute Pearson correlation and VIF between predictors.",
            "3. Held-out Split: Partition dataset into train (80%) and test (20%) folds.",
            "4. Normal Equation Inversion: Estimate vector β̂ = (XᵀX)⁻¹XᵀY.",
            "5. Evaluation: Evaluate Adjusted R², MAE, and examine residual normality."
        ],
        "use_cases": [
            "Campus placement package prediction combining academic CGPA and cognitive IQ scores.",
            "Real estate appraisal factoring square footage, bedrooms, age, and school district rating.",
            "Medical patient blood pressure prediction using BMI, age, daily sodium intake, and exercise hours.",
            "Economic GDP growth modeling using interest rates, inflation, and capital investment."
        ],
        "assumptions": [
            "Linearly additive effects of features on target variable.",
            "No severe Multicollinearity: Independent features must not be highly correlated with each other.",
            "Homoscedasticity of error variance across all predicted hyperplane levels.",
            "Zero conditional mean: E[ε | X] = 0."
        ],
        "advantages": [
            "Captures simultaneous multi-feature influences on continuous outcomes.",
            "Partial coefficients reveal the isolated importance of each factor.",
            "Analytical solution remains fast for moderate feature dimensions.",
            "Can incorporate polynomial expansions and categorical dummy variables."
        ],
        "limitations": [
            "Multicollinearity inflates standard errors and flips coefficient signs.",
            "Standard R² always increases with added features, even if they are pure noise.",
            "Cannot capture complex non-linear interactions without explicit feature engineering.",
            "Matrix inversion (XᵀX)⁻¹ is O(p³) and fails if features exceed sample count (p > N)."
        ],
        "evaluation_metrics": [
            {"metric": "Adjusted R²", "desc": "Penalizes redundant features: 1 - [(1 - R²)(n - 1) / (n - p - 1)]."},
            {"metric": "MAE (Mean Absolute Error)", "desc": "Average magnitude of prediction errors in Lakhs Per Annum."},
            {"metric": "RMSE", "desc": "Root mean squared deviation across test samples."},
            {"metric": "VIF (Variance Inflation Factor)", "desc": "Detects multicollinearity; values > 5 indicate problematic correlation."}
        ],
        "common_mistakes": [
            "Relying on standard R² instead of Adjusted R² when comparing models with different numbers of features.",
            "Interpreting coefficients as causal when predictors are correlated.",
            "Forgetting to check for multicollinearity between features (e.g. CGPA and IQ).",
            "Presenting a 2D line plot and claiming it represents a multiple regression model."
        ],
        "how_prediction_works": "Suppose trained model on Cgpa_iq_package.csv has β₀ = -2.50, β_cgpa = 0.55, and β_iq = 0.02. For a student with CGPA = 8.2 and IQ = 110: ŷ = -2.50 + 0.55*(8.2) + 0.02*(110) = -2.50 + 4.51 + 2.20 = 4.21 LPA.",
        "viva_questions": [
            {
                "q": "What is the difference between R² and Adjusted R²?",
                "a": "R² monotonically increases or stays constant whenever new features are added, even if they are random noise. Adjusted R² penalizes non-informative features by incorporating degrees of freedom: Adj R² = 1 - [(1 - R²)(n - 1) / (n - p - 1)]. It increases only if the new feature improves the model more than expected by chance."
            },
            {
                "q": "What is Multicollinearity and what problems does it cause?",
                "a": "Multicollinearity occurs when two or more independent variables are strongly correlated. It makes the XᵀX matrix nearly singular (determinant near zero), causing coefficient estimates to have huge standard errors, unstable signs, and erratic predictions under minor data perturbations."
            },
            {
                "q": "How do you detect and resolve Multicollinearity?",
                "a": "Detected via Variance Inflation Factor: VIF = 1 / (1 - R_j²). VIF > 5 or 10 indicates severe collinearity. Resolved by: (1) Removing one of the correlated pair, (2) Combining features via PCA, or (3) Applying L2 Ridge regularization."
            },
            {
                "q": "What does a partial regression coefficient mean?",
                "a": "The coefficient βⱼ represents the expected change in the dependent variable Y per unit change in Xⱼ, holding all other explanatory variables X_k (k ≠ j) constant. It isolates the unique contribution of feature Xⱼ."
            },
            {
                "q": "Why cannot OLS be computed when number of features p > number of samples n?",
                "a": "When p > n, the rank of matrix X is at most n, meaning the p x p matrix XᵀX is rank-deficient and non-invertible (singular). There are infinitely many solutions that achieve zero training error, requiring Ridge (L2) or Lasso (L1) regularization to resolve."
            }
        ],
        "practical_task": "Fit Multiple Linear Regression on Cgpa_iq_package.csv, print both partial coefficients, and verify that Adjusted R² is slightly lower than standard R².",
        "code_snippet": "from sklearn.linear_model import LinearRegression\nfrom sklearn.model_selection import train_test_split\nimport pandas as pd\n\ndf = pd.read_csv('datasets/Cgpa_iq_package.csv')\nX = df[['cgpa', 'iq']]\ny = df['package']\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)\nmodel = LinearRegression().fit(X_tr, y_tr)\nprint(f'Coefficients: CGPA={model.coef_[0]:.3f}, IQ={model.coef_[1]:.3f}')"
    },
    "placement": {
        "exp_num": "04",
        "title": "Logistic Regression: Binary Classification",
        "category": "Classification",
        "dataset": "Placement_Y_N.csv",
        "definition": "A parametric classification model that estimates the conditional probability that a binary target belongs to class 1 versus class 0 by applying a sigmoid activation function to a linear combination of input features.",
        "math_formula": "P(Y=1|X) = \\sigma(z) = \\frac{1}{1 + e^{-z}}, \\quad z = \\beta_0 + \\beta_1 x_1 + ... + \\beta_p x_p, \\quad \\ln\\left(\\frac{p}{1-p}\\right) = z",
        "cost_function": "\\text{Binary Cross-Entropy (Log Loss)}: J(w) = -\\frac{1}{m} \\sum_{i=1}^m \\left[ y_i \\ln(\\hat{p}_i) + (1 - y_i) \\ln(1 - \\hat{p}_i) \\right]",
        "math_intuition": "Standard linear regression produces unbounded outputs (-∞, +∞). Logistic regression maps real-valued outputs into valid probabilities (0, 1) using the logistic sigmoid curve σ(z). The log-odds (logit) is linear in the features. The model is trained using Maximum Likelihood Estimation (MLE) via convex gradient descent (L-BFGS), penalizing wrong confident predictions asymptotically towards infinity.",
        "workflow": [
            "1. Feature Preparation: Ingest continuous predictors (CGPA, IQ) and binary target (Placed: 1/0).",
            "2. Feature Standardization: Scale features with StandardScaler to ensure uniform gradient descent steps.",
            "3. Stratified Partition: Split data ensuring equal proportion of placed/unplaced in train and test folds.",
            "4. Maximum Likelihood Optimization: Fit weights w and bias b minimizing binary cross-entropy loss.",
            "5. Thresholding & Evaluation: Apply decision threshold (default 0.50) to evaluate Accuracy, F1, and ROC-AUC."
        ],
        "use_cases": [
            "Student campus placement likelihood prediction (Placed vs Not Placed).",
            "Credit card transaction fraud detection (Fraudulent vs Legitimate).",
            "Customer churn prediction in telecommunications (Churn vs Retained).",
            "Medical screening for disease presence (Positive vs Negative)."
        ],
        "assumptions": [
            "Binary dependent variable with mutually exclusive classes.",
            "Linear relationship between independent variables and the log-odds (logit) of the outcome.",
            "No severe multicollinearity among predictors.",
            "Sample size is sufficiently large to satisfy asymptotic maximum likelihood normality."
        ],
        "advantages": [
            "Outputs calibrated probabilistic confidences P(Y=1|X), not just raw discrete labels.",
            "Highly interpretable: e^(βⱼ) gives the Odds Ratio for feature j.",
            "Convex loss landscape guarantees convergence to the global optimum.",
            "Low computational overhead during both training and real-time inference."
        ],
        "limitations": [
            "Linear decision boundary cannot classify non-linearly separable classes without kernel projection.",
            "Vulnerable to extreme outliers which distort logit probabilities.",
            "Suffers from complete separation: if a feature perfectly separates classes, weights explode to infinity.",
            "Underperforms complex ensemble models on tabular datasets with high interaction depth."
        ],
        "evaluation_metrics": [
            {"metric": "Accuracy (%)", "desc": "Overall correct classifications over total samples."},
            {"metric": "Precision (%)", "desc": "True Positives / (True Positives + False Positives). Minimizes false alarms."},
            {"metric": "Recall (%)", "desc": "True Positives / (True Positives + False Negatives). Minimizes missed cases."},
            {"metric": "F1-Score (%)", "desc": "Harmonic mean of Precision and Recall."},
            {"metric": "ROC-AUC", "desc": "Area under Receiver Operating Characteristic curve (0.5=random, 1.0=perfect)."}
        ],
        "common_mistakes": [
            "Using standard Linear Regression for classification, producing probabilities < 0 or > 1.",
            "Relying solely on Accuracy on imbalanced datasets (e.g. 95% negative gives 95% accuracy by guessing 0).",
            "Leaving decision threshold hardcoded at 0.50 when cost of False Negatives far exceeds False Positives.",
            "Forgetting to scale numerical features when L2 regularization is applied."
        ],
        "how_prediction_works": "Suppose model has weights β₀ = -6.0, β_cgpa = 0.80, β_iq = 0.02. For a student with CGPA = 7.8 and IQ = 105: z = -6.0 + 0.80*(7.8) + 0.02*(105) = -6.0 + 6.24 + 2.10 = +2.34. The sigmoid probability is P = 1 / (1 + e^-2.34) = 1 / (1 + 0.0963) = 0.912 (91.2% placement likelihood). Since 0.912 >= 0.50, predicted class is 1 (Placed).",
        "viva_questions": [
            {
                "q": "Why can't we use Mean Squared Error (MSE) as the loss function for Logistic Regression?",
                "a": "When the non-linear sigmoid activation function σ(z) is plugged into an MSE loss function, the resulting loss surface becomes non-convex with numerous local minima and flat plateaus where gradients vanish. Binary Cross-Entropy loss guarantees a strictly convex loss surface with a single unique global minimum."
            },
            {
                "q": "What is the Odds Ratio and how is it calculated from the logistic regression coefficient?",
                "a": "Odds is defined as P / (1 - P). The Odds Ratio (OR) for a 1-unit increase in feature xⱼ is e^(βⱼ). If β_cgpa = 0.80, OR = e^0.80 ≈ 2.22, meaning each 1.0 point increase in CGPA multiplies the odds of getting placed by 2.22 times, holding IQ constant."
            },
            {
                "q": "What is the difference between ROC Curve and Precision-Recall (PR) Curve?",
                "a": "ROC curves plot True Positive Rate vs False Positive Rate across all thresholds; they can present an overly optimistic picture on heavily imbalanced datasets because the large number of true negatives keeps FPR small. PR curves plot Precision vs Recall and focus strictly on the minority positive class, making them preferred for imbalanced data."
            },
            {
                "q": "What happens in Logistic Regression if the two classes are perfectly linearly separable?",
                "a": "This is known as the 'Complete Separation' problem. To achieve a training cross-entropy loss of zero, the model drives the weights β toward +∞ or -∞, causing gradient descent to fail to converge and standard errors to explode unless L2 regularization is applied."
            },
            {
                "q": "How does changing the classification threshold from 0.5 to 0.3 affect Precision and Recall?",
                "a": "Lowering the threshold to 0.3 makes the model more aggressive in predicting class 1. This captures more true positives, increasing Recall (sensitivity), but also admits more false positives, reducing Precision."
            }
        ],
        "practical_task": "Train a LogisticRegression model on Placement_Y_N.csv, plot the confusion matrix, and compute the ROC-AUC score.",
        "code_snippet": "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import classification_report, roc_auc_score\nimport pandas as pd\n\ndf = pd.read_csv('datasets/Placement_Y_N.csv')\nmodel = LogisticRegression().fit(df[['cgpa', 'iq']], df['placement'])\nprobs = model.predict_proba(df[['cgpa', 'iq']])[:, 1]\nprint('ROC-AUC:', roc_auc_score(df['placement'], probs))"
    },
    "stress": {
        "exp_num": "05",
        "title": "Logistic Regression: Multiclass Classification",
        "category": "Classification",
        "dataset": "Multiclass Stress Dataset",
        "definition": "A multinomial generalization of logistic regression that categorizes inputs into three or more mutually exclusive classes using the Softmax normalized exponential function.",
        "math_formula": "P(Y=k|X) = \\frac{e^{w_k^T X}}{\\sum_{j=1}^K e^{w_j^T X}} \\quad \\text{for } k \\in \\{1, 2, ..., K\\}, \\quad \\sum_{k=1}^K P(Y=k|X) = 1.0",
        "cost_function": "\\text{Categorical Cross-Entropy Loss}: L(\\Theta) = -\\sum_{i=1}^m \\sum_{k=1}^K y_{ik} \\ln(\\hat{p}_{ik})",
        "math_intuition": "For K classes, the model computes K linear scoring functions z_k = w_kᵀX + b_k. The Softmax function exponentiates each score to ensure positivity and normalizes by the sum of all exponentials. This produces a valid categorical probability distribution where the largest score receives the highest probability.",
        "workflow": [
            "1. Multi-tier Ingestion: Ingest workload predictors (Study Hours, Attendance) and discrete stress tier (Low, Med, High).",
            "2. Pipeline Integration: Construct a scikit-learn Pipeline with StandardScaler and multinomial LogisticRegression.",
            "3. Stratified Partition: Partition while preserving balanced sample representation across all 3 classes.",
            "4. Softmax Optimization: Minimize categorical cross-entropy using the L-BFGS quasi-Newton solver.",
            "5. Multiclass Evaluation: Generate a 3x3 confusion matrix and macro/weighted F1-scores."
        ],
        "use_cases": [
            "Student academic stress and burnout categorization (Low, Moderate, High).",
            "Customer support ticket routing (Billing, Technical, Account, Cancellation).",
            "Handwritten digit classification (MNIST: digits 0 through 9).",
            "Medical triage acuity level classification (Urgent, Emergent, Non-Urgent)."
        ],
        "assumptions": [
            "Target classes are mutually exclusive (each instance belongs to exactly one class).",
            "Independence of Irrelevant Alternatives (IIA assumption).",
            "Linear separability of class boundaries in transformed logit feature space.",
            "Sufficient sample count within each target tier."
        ],
        "advantages": [
            "Produces calibrated probabilities across all K classes that sum strictly to 1.0.",
            "Jointly optimizes all class boundaries simultaneously in multinomial mode.",
            "Fast convergence via L-BFGS solver with L2 weight regularization.",
            "More principled than heuristics like One-vs-Rest (OvR)."
        ],
        "limitations": [
            "Assumes linear decision boundaries between all class pairs.",
            "The IIA assumption can fail if two classes are near-substitutes.",
            "Boundary regions between intermediate classes (Moderate stress) often suffer ambiguity.",
            "Sensitive to severe class imbalance where rare classes get suppressed."
        ],
        "evaluation_metrics": [
            {"metric": "Overall Accuracy", "desc": "Trace of confusion matrix divided by total samples."},
            {"metric": "Macro F1-Score", "desc": "Unweighted average of F1 across all classes; treats all classes equally."},
            {"metric": "Weighted F1-Score", "desc": "F1 weighted by class support; accounts for class frequency differences."},
            {"metric": "Multi-class Log Loss", "desc": "Measures uncertainty of predicted probability distributions."}
        ],
        "common_mistakes": [
            "Using One-vs-Rest when true multiclass Softmax is supported and desired.",
            "Reporting overall accuracy on imbalanced classes without checking Macro F1.",
            "Confusing Multiclass classification (one label from K choices) with Multilabel classification (multiple labels simultaneously).",
            "Neglecting feature scaling before running L-BFGS multinomial solver."
        ],
        "how_prediction_works": "Suppose input yields unnormalized linear scores: Low Stress z₀ = 1.2, Med Stress z₁ = 3.5, High Stress z₂ = 0.8. Exponentials: e^1.2 = 3.32, e^3.5 = 33.12, e^0.8 = 2.23. Sum = 38.67. Softmax probabilities: P(Low) = 3.32/38.67 = 8.6%, P(Med) = 33.12/38.67 = 85.6%, P(High) = 2.23/38.67 = 5.8%. Final prediction is argmax = Medium Stress (85.6%).",
        "viva_questions": [
            {
                "q": "What is the difference between One-vs-Rest (OvR) and Multinomial (Softmax) logistic regression?",
                "a": "OvR trains K separate independent binary classifiers, each distinguishing class k from all other K-1 classes combined; probabilities are normalized as a post-hoc heuristic. Multinomial Softmax optimizes a single categorical cross-entropy loss function over all K classes simultaneously, producing strictly calibrated joint probabilities."
            },
            {
                "q": "What is the Independence of Irrelevant Alternatives (IIA) assumption in multinomial logistic regression?",
                "a": "IIA states that the odds ratio between any two classes P(A)/P(B) remains completely unaffected by the presence or addition of a third class C. This assumption can fail when classes are close substitutes (the classic 'Red Bus / Blue Bus' paradox)."
            },
            {
                "q": "What is the difference between Macro Average and Weighted Average in multiclass classification?",
                "a": "Macro average computes the metric independently for each class and takes the unweighted arithmetic mean, giving equal weight to every class regardless of size (vital for finding poor minority class performance). Weighted average weights each class by its support (sample count), reflecting overall dataset performance."
            },
            {
                "q": "How does the Softmax function handle extreme score differences?",
                "a": "Because Softmax exponentiates scores, small differences in linear scores get magnified exponentially. A score difference of 4.0 (e.g. z₁ = 5 vs z₂ = 1) yields an odds ratio of e⁴ ≈ 54.6, making the highest-scoring class dominate the probability distribution."
            },
            {
                "q": "How is a multiclass confusion matrix interpreted?",
                "a": "In a K x K confusion matrix, rows represent actual ground-truth classes and columns represent predicted classes. Diagonal elements (C_ii) represent correct classifications (True Positives). Off-diagonal elements indicate specific misclassifications (e.g. C_12 means class 1 misclassified as class 2)."
            }
        ],
        "practical_task": "Inspect the 3x3 confusion matrix and compare Macro F1 vs Weighted F1 for student stress classification.",
        "code_snippet": "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import classification_report\nimport numpy as np\n\n# Fit multiclass model with multinomial softmax\nclf = LogisticRegression(multi_class='multinomial', solver='lbfgs')\n# Generate report with per-tier precision, recall, and F1"
    },
    "heart": {
        "exp_num": "06",
        "title": "Decision Tree Classification",
        "category": "Classification",
        "dataset": "heart_disease_dataset.csv",
        "definition": "A non-parametric supervised algorithm that recursively partitions feature space into hierarchical rectangular regions using greedy binary splits determined by Gini impurity or Information Gain.",
        "math_formula": "\\text{Gini Impurity}: I_G(p) = 1 - \\sum_{k=1}^K p_k^2, \\quad \\text{Entropy}: H(p) = -\\sum_{k=1}^K p_k \\log_2(p_k)",
        "cost_function": "\\text{Impurity Reduction (Gain)}: \\Delta I = I(D_{\\text{parent}}) - \\left( \\frac{N_L}{N} I(D_L) + \\frac{N_R}{N} I(D_R) \\right)",
        "math_intuition": "At each internal node, CART evaluates every candidate feature and split threshold to find the split that maximizes the drop in Gini impurity. Pure nodes (all instances belonging to one class) have Gini = 0. Unconstrained trees will continue splitting until every training instance is isolated in a pure leaf node, achieving 100% training accuracy but severely overfitting.",
        "workflow": [
            "1. Clinical Data Ingestion: Ingest 13 cardiovascular attributes (age, chest pain type, max heart rate, cholesterol).",
            "2. Recursive Binary Splitting: Greedy search across all 13 features for the threshold maximizing Gini gain.",
            "3. Tree Construction: Branch into left and right subsets until max_depth or min_samples_split is reached.",
            "4. Pruning / Regularization: Constrain tree depth (e.g. max_depth=5) to prevent overfitting.",
            "5. Rule Extraction: Extract explicit if-then diagnostic rules from the learned tree hierarchy."
        ],
        "use_cases": [
            "Cardiovascular heart disease screening and clinical triage decision support.",
            "Loan eligibility rules generation in retail banking.",
            "Fault diagnosis in electrical power grids and industrial machinery.",
            "E-commerce customer purchase intent classification."
        ],
        "assumptions": [
            "Non-parametric: Makes zero assumptions about data distributions, normality, or linear separability.",
            "Axis-aligned splits: Decision boundaries are strictly orthogonal to feature axes.",
            "Features can be continuous or discrete without monotonic scaling requirements."
        ],
        "advantages": [
            "White-box interpretability: Decisions can be visualized as human-readable if-then decision trees.",
            "Invariant to monotonic feature scaling (no StandardScaler required).",
            "Handles non-linear relationships and feature interactions naturally.",
            "Fast O(depth) inference speed."
        ],
        "limitations": [
            "High variance: Small perturbations in training data can produce radically different trees.",
            "Prone to severe overfitting if depth is unconstrained.",
            "Axis-aligned cuts struggle to approximate diagonal decision boundaries.",
            "Greedy top-down splitting does not guarantee the globally optimal tree."
        ],
        "evaluation_metrics": [
            {"metric": "Accuracy (%)", "desc": "Correct risk classifications over total test patients."},
            {"metric": "Sensitivity / Recall (%)", "desc": "Critical in healthcare: minimizes missed heart disease diagnoses."},
            {"metric": "Specificity (%)", "desc": "True Negative rate: avoids false alarm anxiety for healthy patients."},
            {"metric": "Gini Impurity", "desc": "Node impurity metric: 0.0 indicates a perfectly pure diagnostic leaf."}
        ],
        "common_mistakes": [
            "Training without setting max_depth, allowing the tree to memorize noise and overfit.",
            "Assuming feature importance implies causality in clinical diagnoses.",
            "Failing to prioritize Recall over Precision in medical screening applications.",
            "Thinking feature scaling is necessary for decision trees (trees only care about ordinal rank)."
        ],
        "how_prediction_works": "A patient arrives: Age 57, Chest Pain cp = 2, Max Heart Rate thalach = 174, Fluoroscopy ca = 1. Root node checks: 'Is ca <= 0.5?' Since ca = 1 (> 0.5), follow RIGHT branch. Next node checks: 'Is thalach <= 140?' Since 174 > 140, follow right sub-branch. Reaches Leaf Node #7: 18 training patients in this leaf, 15 with heart disease (Class 1). Model predicts: Class 1 (Disease Present, 83.3% confidence).",
        "viva_questions": [
            {
                "q": "What is the difference between Gini Impurity and Entropy?",
                "a": "Both measure node impurity. Gini computes 1 - Σ pᵢ² and is computationally faster because it avoids logarithmic calculations. Entropy computes -Σ pᵢ log₂(pᵢ) based on Shannon information theory. In practice, both yield almost identical decision trees >98% of the time."
            },
            {
                "q": "What is Cost-Complexity Pruning (ccp_alpha) in Decision Trees?",
                "a": "Cost-complexity pruning adds a penalty for tree complexity: R_α(T) = R(T) + α|T|, where R(T) is training error, |T| is the number of terminal leaves, and α is the complexity parameter. It removes weak branches that increase tree size without significantly reducing impurity."
            },
            {
                "q": "Why do Decision Trees have high variance, and how is it resolved?",
                "a": "Because greedy splitting decisions made at the top root node dictate all subsequent branch possibilities, a slight change in training samples can flip a split decision, completely restructuring the entire tree. This variance is resolved by bagging ensembles like Random Forests."
            },
            {
                "q": "Why is feature scaling unnecessary for Decision Trees?",
                "a": "Decision trees evaluate splits based purely on ordinal rank ('Is X_j <= threshold?'). Multiplying or shifting a feature monotonically (e.g. via log or scaling) does not change the rank ordering of samples, leaving the optimal split threshold identical."
            },
            {
                "q": "How does a Decision Tree calculate Feature Importance?",
                "a": "Feature importance (Mean Decrease in Impurity / MDI) is calculated as the total weighted reduction in Gini impurity brought about by all splits using that feature across the tree, normalized so that all feature importances sum to 1.0 (or 100%)."
            }
        ],
        "practical_task": "Train a DecisionTreeClassifier on heart_disease_dataset.csv with max_depth=4, export text rules, and examine the top 3 most important clinical features.",
        "code_snippet": "from sklearn.tree import DecisionTreeClassifier, export_text\nimport pandas as pd\n\ndf = pd.read_csv('datasets/heart_disease_dataset.csv')\nX, y = df.drop('target', axis=1), df['target']\nclf = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X, y)\nprint(export_text(clf, feature_names=list(X.columns)))"
    },
    "breast": {
        "exp_num": "07",
        "title": "K-Nearest Neighbors (KNN)",
        "category": "Classification",
        "dataset": "breast-cancer.csv",
        "definition": "A non-parametric, instance-based lazy learning algorithm that classifies an unlabelled query instance based on the majority class vote among its K geometrically closest training instances in feature space.",
        "math_formula": "d(p, q) = \\sqrt{\\sum_{i=1}^D (p_i - q_i)^2}, \\quad P(Y=c|X) = \\frac{1}{K} \\sum_{i \\in N_K(X)} I(y_i = c)",
        "cost_function": "\\text{No training cost function. Instance storage: } \\mathcal{D} = \\{(x_1, y_1), ..., (x_N, y_N)\\}",
        "math_intuition": "KNN has zero training phase (lazy learning); it simply memorizes training vectors. At inference time, it computes the distance from the query point to all N training vectors, identifies the K smallest distances, and takes the majority vote. If features are not normalized, features with large scales (e.g. area ~ 2000) completely dominate features with small scales (e.g. smoothness ~ 0.1).",
        "workflow": [
            "1. Biopsy Data Ingestion: Ingest 30 nuclear morphology attributes from fine-needle aspirates.",
            "2. Strict Standardization: Standardize all 30 features using StandardScaler (zero mean, unit variance).",
            "3. Distance Metric Selection: Configure Euclidean distance metric d(u, v) = ||u - v||₂.",
            "4. K Hyperparameter Selection: Evaluate K values from 1 to 25 to find optimal bias-variance tradeoff.",
            "5. Neighbor Voting: Classify query biopsy into Benign or Malignant based on majority neighbor vote."
        ],
        "use_cases": [
            "Breast cancer biopsy diagnosis (Benign vs Malignant nuclear structure).",
            "Facial recognition and fingerprint biometric matching.",
            "Content-based recommendation systems (finding movies similar to user favorites).",
            "Gene expression micro-array classification in genomic pathology."
        ],
        "assumptions": [
            "Locality assumption: Query instances close in feature space share similar semantic class labels.",
            "All features are continuous and equally scaled on identical dimensional axes.",
            "Training set contains sufficient representative density across the entire input space."
        ],
        "advantages": [
            "Zero training phase: Instantly adapts to newly ingested training samples.",
            "Can learn highly complex non-linear decision boundaries.",
            "Intuitive geometric decision rationale (nearest case matching).",
            "Naturally supports multi-class classification."
        ],
        "limitations": [
            "Computationally expensive inference: O(N * D) distance computations per query sample.",
            "Curse of Dimensionality: In 30 dimensions, points become equidistant and distance metrics lose discriminative power.",
            "Extremely sensitive to unnormalized feature scales and irrelevant noise features.",
            "High memory footprint: Must store the entire training dataset in RAM."
        ],
        "evaluation_metrics": [
            {"metric": "Biopsy Sensitivity / Recall (%)", "desc": "Critical: percentage of malignant tumors correctly flagged."},
            {"metric": "Accuracy (%)", "desc": "Overall correct biopsy diagnoses."},
            {"metric": "K-Value Validation Error Curve", "desc": "Plots error across K=1 to 25 to pick optimal K."},
            {"metric": "Euclidean Distance", "desc": "Geometric metric measuring morphological dissimilarity."}
        ],
        "common_mistakes": [
            "Forgetting to standardize features: area_mean (~1500) will overpower smoothness_mean (~0.1).",
            "Selecting an even value of K in binary classification, leading to 50/50 vote ties.",
            "Using K = 1, which results in zero training error but extreme sensitivity to noisy outliers (overfitting).",
            "Selecting K = N, which always predicts the global majority class (underfitting)."
        ],
        "how_prediction_works": "A standardized biopsy query arrives with K=5. Model calculates Euclidean distance to all 455 training points. The 5 closest training instances have distances [0.42, 0.58, 0.61, 0.73, 0.82]. Their respective training labels are [Malignant, Malignant, Malignant, Benign, Malignant]. Vote tally: 4 Malignant vs 1 Benign. Predicted class: Malignant (80% neighbor consensus).",
        "viva_questions": [
            {
                "q": "Why is Feature Scaling strictly mandatory for K-Nearest Neighbors?",
                "a": "KNN relies directly on geometric distance formulas like Euclidean distance: d = √[Σ (x_i - y_i)²]. If one feature ranges from 100 to 2000 (e.g. area) while another ranges from 0.05 to 0.25 (e.g. smoothness), the squared difference of area will be millions of times larger, completely drowning out the contribution of smoothness."
            },
            {
                "q": "How does the choice of K affect the Bias-Variance Tradeoff?",
                "a": "K=1 produces complex, island-like decision boundaries with zero bias but high variance (extreme overfitting to training noise). As K increases, boundaries become smoother, reducing variance but increasing bias. If K=N, the boundary is eliminated and the model simply predicts the majority class."
            },
            {
                "q": "What is the 'Curse of Dimensionality' in KNN and how does it degrade performance?",
                "a": "As dimensionality D increases, the volume of feature space grows exponentially (hypercube volume ∝ r^D), causing data points to become extremely sparse. In high-dimensional space, the ratio between the distance to the nearest point and the distance to the farthest point approaches 1, meaning all points become equidistant."
            },
            {
                "q": "Why is KNN called a 'Lazy Learner'?",
                "a": "KNN is called lazy because it performs zero computation during the training phase—it merely stores the training vectors in memory. All computational work (distance calculations, sorting, and voting) is deferred until a prediction query is made at inference time."
            },
            {
                "q": "What are the common distance metrics used in KNN besides Euclidean?",
                "a": "Manhattan distance (L1 norm: Σ |x_i - y_i|), Minkowski distance (generalized L_p norm), Cosine distance (measures angular alignment rather than magnitude, ideal for text vectors), and Hamming distance (for categorical or binary bitstrings)."
            }
        ],
        "practical_task": "Scale breast-cancer.csv with StandardScaler, evaluate KNN across K=1 to 15, and identify the value of K maximizing held-out test F1-score.",
        "code_snippet": "from sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.preprocessing import StandardScaler\nimport pandas as pd\n\ndf = pd.read_csv('datasets/breast-cancer.csv')\nfeatures = [c for c in df.columns if c not in ['id', 'diagnosis']]\nX = StandardScaler().fit_transform(df[features])\nknn = KNeighborsClassifier(n_neighbors=5).fit(X, df['diagnosis'])\nprint('Model ready with K=5')"
    },
    "message": {
        "exp_num": "08",
        "title": "Naive Bayes Classification",
        "category": "Classification",
        "dataset": "college_messages_222.csv",
        "definition": "A generative probabilistic classifier based on Bayes' theorem that assumes conditional independence between every pair of features given the class label.",
        "math_formula": "P(C_k|X) = \\frac{P(C_k) \\prod_{i=1}^n P(x_i|C_k)}{P(X)}, \\quad \\hat{y} = \\arg\\max_k P(C_k) \\prod_{i=1}^n P(x_i|C_k)",
        "cost_function": "\\text{Laplace Smoothing Likelihood}: P(w|C_k) = \\frac{N_{w, C_k} + \\alpha}{N_{C_k} + \\alpha |V|}",
        "math_intuition": "Naive Bayes applies Bayes' rule to compute the posterior probability of each class given the observed words. The 'naive' assumption posits that the probability of encountering word x_i is completely independent of word x_j given class C_k. In log space, products turn into sums: log P(C_k) + Σ log P(x_i|C_k), avoiding floating-point underflow.",
        "workflow": [
            "1. Corpus Preprocessing: Clean, lower-case, and tokenize college campus message text.",
            "2. Bag-of-Words Vectorization: Transform text into token count matrices using CountVectorizer.",
            "3. Class Prior Calculation: Compute prior probability P(C_k) = N_k / N for each notification class.",
            "4. Conditional Likelihood Estimation: Estimate smoothed word likelihoods with Laplace smoothing α=1.0.",
            "5. MAP Inference: Assign notification to category maximizing log-posterior score."
        ],
        "use_cases": [
            "College broadcast notification classification (Official Notice, Exam Schedule, Placement Alert).",
            "Email spam filtering (Spam vs Ham).",
            "Sentiment analysis of student course feedback and product reviews.",
            "Author attribution and language identification in computational linguistics."
        ],
        "assumptions": [
            "Class-conditional independence: Features are conditionally independent given the class label.",
            "Bag-of-Words assumption: Word position and syntactic grammar do not affect likelihood.",
            "Vocabulary coverage: Words outside the training vocabulary are handled via additive smoothing."
        ],
        "advantages": [
            "Extremely fast training and inference O(N * vocabulary_size).",
            "Requires very little training data to estimate parameters compared to deep NLP.",
            "Highly effective for text classification with sparse high-dimensional vocabularies.",
            "Robust to irrelevant features which get small likelihoods across all classes."
        ],
        "limitations": [
            "The independence assumption is virtually always violated in natural human language.",
            "Zero Frequency Problem: An unseen word has probability 0 without smoothing, collapsing the product.",
            "Cannot understand phrase semantics, negation, or word order ('not good' treated like 'good').",
            "Probability outputs are notoriously poorly calibrated (pushed towards 0 or 1)."
        ],
        "evaluation_metrics": [
            {"metric": "Accuracy (%)", "desc": "Overall correct categorization of college announcements."},
            {"metric": "Per-class Precision", "desc": "Ensures urgent notices are not contaminated by spam."},
            {"metric": "Per-class Recall", "desc": "Ensures no crucial examination notice is missed."},
            {"metric": "Macro F1-Score", "desc": "Equally weights small notice categories."}
        ],
        "common_mistakes": [
            "Setting Laplace smoothing α = 0, causing probabilities to collapse to zero for unseen words.",
            "Vectorizing test data with a new vectorizer instead of using the fitted training vocabulary.",
            "Interpreting Naive Bayes probability confidences as calibrated real probabilities.",
            "Using GaussianNB instead of MultinomialNB for integer word count vectors."
        ],
        "how_prediction_works": "Query: 'Practical exam schedule announced'. Vocabulary counts show 'exam' and 'schedule' appear heavily in 'Notice' (likelihood 0.045) and rarely in 'Promotional' (0.002). Prior P(Notice) = 0.60. Log-score Notice = log(0.60) + log(0.045) + log(0.038) = -0.51 - 3.10 - 3.27 = -6.88. Log-score Promo = -0.91 - 6.21 - 5.80 = -12.92. Since -6.88 > -12.92, predicted category is Notice.",
        "viva_questions": [
            {
                "q": "Why is the Naive Bayes algorithm called 'Naive'?",
                "a": "It is called 'naive' because it makes the extremely strong and unrealistic assumption that all input features (e.g. words in a sentence) are conditionally independent of each other given the class label. In reality, words like 'Machine' and 'Learning' frequently co-occur with strong statistical dependence."
            },
            {
                "q": "What is the 'Zero Probability Problem' and how does Laplace Smoothing solve it?",
                "a": "If a test document contains a word w that never appeared in class C_k during training, the likelihood P(w|C_k) = 0. Multiplying this zero collapses the entire joint probability to zero. Laplace smoothing adds α (typically 1.0) to the numerator and α|V| to the denominator, guaranteeing a small non-zero probability."
            },
            {
                "q": "What is the difference between MultinomialNB, BernoulliNB, and GaussianNB?",
                "a": "MultinomialNB models discrete word count frequencies (term frequencies). BernoulliNB models binary word occurrence (word present/absent: 1/0). GaussianNB models continuous real-valued features assuming a normal Gaussian distribution N(μ, σ²)."
            },
            {
                "q": "Why do we compute log-probabilities log P(C_k) + Σ log P(w_i|C_k) instead of direct multiplication?",
                "a": "Multiplying dozens of fractional word probabilities (e.g. 0.001 * 0.0005 * 0.002) quickly exceeds the precision limit of 64-bit floating point numbers, resulting in arithmetic underflow to zero. Working in log space converts products into stable additions."
            },
            {
                "q": "Can Naive Bayes classify text containing words not present in the training vocabulary?",
                "a": "Yes. When using a pre-fitted CountVectorizer, any word not in the learned vocabulary is simply ignored (out-of-vocabulary), and classification proceeds based on the remaining recognized tokens."
            }
        ],
        "practical_task": "Build a pipeline with CountVectorizer and MultinomialNB, fit on college_messages_222.csv, and test classification on a custom sample notice.",
        "code_snippet": "from sklearn.feature_extraction.text import CountVectorizer\nfrom sklearn.naive_bayes import MultinomialNB\nfrom sklearn.pipeline import Pipeline\nimport pandas as pd\n\ndf = pd.read_csv('datasets/college_messages_222.csv')\npipe = Pipeline([('vec', CountVectorizer()), ('nb', MultinomialNB())])\npipe.fit(df['message'], df['label'])\nprint('Prediction:', pipe.predict(['Final exam timetable released']))"
    },
    "diabetes": {
        "exp_num": "09",
        "title": "Random Forest Classification",
        "category": "Classification",
        "dataset": "diabetes.csv",
        "definition": "An ensemble bagging (Bootstrap Aggregating) algorithm that trains an ensemble of B de-correlated decision trees on bootstrap samples with random feature subspace sampling, aggregating individual predictions via majority voting.",
        "math_formula": "\\hat{f}(x) = \\text{mode}\\{T_1(x), T_2(x), ..., T_B(x)\\}, \\quad m \\approx \\sqrt{p} \\text{ features sampled at each split}",
        "cost_function": "\\text{Ensemble Variance Reduction}: \\text{Var}(\\bar{T}) = \\rho \\sigma^2 + \\frac{1 - \\rho}{B} \\sigma^2",
        "math_intuition": "A single decision tree has high variance. By generating B bootstrap replicas D_b and training B individual trees, bagging reduces variance by 1/B if trees are independent. To prevent trees from all choosing the same dominant feature (de-correlation), Random Forest restricts each candidate split to a random subset of m ≈ √p features, minimizing inter-tree correlation ρ.",
        "workflow": [
            "1. Metabolic Ingestion: Ingest 8 clinical attributes (Glucose, BMI, Age, Insulin, BloodPressure).",
            "2. Bootstrap Sampling: Draw B bootstrap samples of size N with replacement from training fold.",
            "3. De-correlated Tree Growth: Train unpruned trees, selecting m ≈ √8 ≈ 2 random features per split.",
            "4. Out-of-Bag (OOB) Tracking: Evaluate unseen out-of-bag samples (~36.8%) for cross-validation.",
            "5. Majority Voting: Aggregate votes across all B trees to assign final binary diabetes diagnosis."
        ],
        "use_cases": [
            "Diabetes onset risk screening across patient demographic and metabolic vitals.",
            "Credit default risk assessment in banking institutions.",
            "Customer churn prediction in enterprise SaaS platforms.",
            "E-commerce product recommendation click-through rate modeling."
        ],
        "assumptions": [
            "Individual ensemble trees must have low correlation with one another.",
            "Ensemble reduces variance without substantially increasing individual tree bias.",
            "Features contain true predictive signal rather than purely uncorrelated noise."
        ],
        "advantages": [
            "Exceptional predictive accuracy on structured tabular datasets.",
            "Highly robust to overfitting due to bootstrap variance reduction.",
            "Provides reliable Out-of-Bag (OOB) error estimates without separate validation splits.",
            "Yields robust global Feature Importance rankings (Mean Decrease Impurity)."
        ],
        "limitations": [
            "Loss of single-tree interpretability: Combines hundreds of trees (semi-black box).",
            "High memory footprint and slower inference compared to a single decision tree.",
            "Biased toward features with many distinct numerical categories or levels.",
            "Cannot extrapolate outside the domain of observed training values."
        ],
        "evaluation_metrics": [
            {"metric": "Accuracy (%)", "desc": "Overall correct screening diagnoses on test patients."},
            {"metric": "OOB Score (%)", "desc": "Out-of-Bag validation score computed internally during training."},
            {"metric": "ROC-AUC", "desc": "Discrimination performance across all probability thresholds."},
            {"metric": "F1-Score (%)", "desc": "Balances clinical precision and recall under class imbalance."}
        ],
        "common_mistakes": [
            "Using an excessively small number of estimators (e.g. n_estimators=5), failing to achieve variance reduction.",
            "Assuming Random Forest cannot overfit: an overly noisy dataset with n_estimators=1000 can still overfit.",
            "Ignoring class imbalance in clinical datasets (must use class_weight='balanced').",
            "Treating feature importance rankings as proof of biological causation."
        ],
        "how_prediction_works": "A patient with Glucose=142, BMI=33.6, Age=45 is evaluated by an ensemble of B=100 trees. Each tree traces its unique random splits down to a leaf. 78 trees vote Class 1 (Diabetes Positive); 22 trees vote Class 0 (Negative). Majority vote rule: Class 1 (78% consensus probability).",
        "viva_questions": [
            {
                "q": "What are the two key mechanisms Random Forest uses to de-correlate its individual trees?",
                "a": "1. Bootstrap Aggregating (Bagging): Each tree is trained on a different bootstrap sample drawn with replacement from the training set. 2. Random Feature Subspace Sampling: At every candidate split, only a random subset m ≈ √p features is considered, preventing a single dominant feature from being chosen at the root of every tree."
            },
            {
                "q": "What is the Out-of-Bag (OOB) error estimate and why is it useful?",
                "a": "When drawing bootstrap samples of size N with replacement, the probability of any sample NOT being chosen is (1 - 1/N)^N ≈ 1/e ≈ 36.8%. These unselected samples are 'out-of-bag' for that tree. Evaluating each sample on all trees where it was OOB provides an unbiased cross-validation estimate without needing a separate validation set."
            },
            {
                "q": "How does Random Forest prevent the overfitting typical of single Decision Trees?",
                "a": "While individual unpruned decision trees have high variance and overfit, averaging B de-correlated trees reduces the variance component of error by a factor approaching 1/B (as tree correlation ρ → 0), leaving bias largely unchanged and dramatically lowering overall test error."
            },
            {
                "q": "What is the difference between Bagging and Boosting?",
                "a": "Bagging (Random Forest) trains independent trees in parallel on bootstrap samples and averages their outputs to reduce variance. Boosting (AdaBoost, XGBoost, LightGBM) trains trees sequentially, where each new tree specifically focuses on correcting the residual errors of preceding trees to reduce bias."
            },
            {
                "q": "How does Random Forest compute MDI Feature Importance?",
                "a": "Mean Decrease in Impurity (MDI) sums the total Gini impurity reduction brought about by a feature across all nodes in all B trees, weighted by the proportion of samples reaching each node, normalized so that importances sum to 1.0 across all features."
            }
        ],
        "practical_task": "Fit a RandomForestClassifier on diabetes.csv with 100 trees, print the OOB score, and extract the top 3 most predictive clinical features.",
        "code_snippet": "from sklearn.ensemble import RandomForestClassifier\nimport pandas as pd\n\ndf = pd.read_csv('datasets/diabetes.csv')\nX, y = df.drop('Outcome', axis=1), df['Outcome']\nrf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42).fit(X, y)\nprint(f'OOB Score: {rf.oob_score_:.4f}')\nprint('Top Feature:', X.columns[rf.feature_importances_.argmax()])"
    },
    "glucose": {
        "exp_num": "10",
        "title": "Random Forest Regression",
        "category": "Regression",
        "dataset": "diabetes.csv",
        "definition": "An ensemble regression algorithm that builds multiple de-correlated regression trees and outputs the arithmetic average of their individual continuous numerical predictions.",
        "math_formula": "\\hat{y} = \\frac{1}{B} \\sum_{b=1}^B T_b(x), \\quad m \\approx \\frac{p}{3} \\text{ features evaluated at each split}",
        "cost_function": "\\text{Mean Squared Error Loss}: \\text{MSE} = \\frac{1}{m} \\sum_{i=1}^m (y_i - \\hat{y}_i)^2, \\quad \\Delta \\text{Var} = \\text{Var}_{\\text{parent}} - \\sum \\frac{N_c}{N} \\text{Var}_c",
        "math_intuition": "Each regression tree partitions feature space into hyper-rectangles where the leaf value is the mean target value of training samples falling into that partition. Individual regression trees produce step-function outputs with jagged discontinuous borders. Averaging B de-correlated trees produces a smooth continuous prediction surface with significantly lower variance.",
        "workflow": [
            "1. Feature Isolation: Isolate continuous blood glucose target Y from remaining 7 clinical vitals X.",
            "2. Bootstrap Replicas: Draw B=100 bootstrap training datasets with replacement.",
            "3. Variance-Reduction Splitting: Grow regression trees minimizing internal leaf variance using m ≈ p/3 features.",
            "4. Continuous Averaging: Predict continuous numerical glucose by averaging outputs across all 100 trees.",
            "5. Residual Diagnosis: Compute MAE, RMSE, and inspect residual error distribution."
        ],
        "use_cases": [
            "Continuous blood glucose level estimation from non-invasive vitals and demographic data.",
            "Automobile resale value prediction based on age, mileage, condition, and brand.",
            "Crop yield forecasting based on soil chemistry, rainfall, and temperature telemetry.",
            "Energy consumption load forecasting in smart electrical grids."
        ],
        "assumptions": [
            "Continuous target variable with sufficient variance in training distribution.",
            "Predictions are bounded within the range of observed training target values.",
            "Individual trees have low pairwise correlation."
        ],
        "advantages": [
            "Effectively models complex non-linear continuous relationships without manual polynomial features.",
            "Robust to outlier targets compared to ordinary least squares linear regression.",
            "Provides continuous feature importance scores.",
            "Requires minimal feature preprocessing."
        ],
        "limitations": [
            "Cannot extrapolate outside the minimum and maximum target bounds seen during training.",
            "Predictions in sparse regions tend toward the global training mean.",
            "Computationally demanding when B is large.",
            "Cannot provide smooth analytical derivatives."
        ],
        "evaluation_metrics": [
            {"metric": "R² (Coefficient of Determination)", "desc": "Proportion of target glucose variance explained by the ensemble."},
            {"metric": "MAE (Mean Absolute Error)", "desc": "Average prediction error magnitude in mg/dL."},
            {"metric": "RMSE", "desc": "Square root of mean squared error; penalizes large prediction errors."},
            {"metric": "Residual Distribution", "desc": "Should resemble a zero-centered Gaussian distribution."}
        ],
        "common_mistakes": [
            "Attempting to use Random Forest Regressor to extrapolate upward or downward trends into the future.",
            "Over-interpreting predictions near the extreme boundaries of target values.",
            "Failing to prune tree depth when sample size is small, leading to individual tree memorization.",
            "Forgetting that regression trees predict average leaf values, creating step-like responses."
        ],
        "how_prediction_works": "To estimate blood glucose for a patient: Tree #1 estimates 132 mg/dL, Tree #2 estimates 128 mg/dL, Tree #3 estimates 135 mg/dL ... Tree #100 estimates 131 mg/dL. The ensemble averages all 100 individual estimates: ŷ = (1/100) * (132 + 128 + ... + 131) = 131.4 mg/dL.",
        "viva_questions": [
            {
                "q": "Why can NEVER a Random Forest Regressor predict a value higher than the maximum seen in training data?",
                "a": "Because every leaf of every regression tree outputs the arithmetic mean of the training samples that fall into that partition: ŷ_leaf = (1/n) Σ y_i. Since the average of a subset of numbers can never exceed the maximum of those numbers, the ensemble average across all trees can never exceed the maximum target value in the training dataset."
            },
            {
                "q": "What is the recommended number of features m to evaluate at each split for Regression vs Classification in Random Forest?",
                "a": "For Classification, the standard heuristic is m ≈ √p (square root of total features). For Regression, the standard heuristic is m ≈ p / 3 (one-third of total features). In scikit-learn, max_features='sqrt' is standard for classifiers and max_features=1.0 or 'sqrt' is used for regressors."
            },
            {
                "q": "What is the split criterion used in Random Forest Regression?",
                "a": "Splits are chosen to maximize the reduction in Mean Squared Error (variance reduction): ΔVar = Var_parent - [(N_L/N)*Var_L + (N_R/N)*Var_R], where Var = (1/N) Σ (y_i - ȳ)²."
            },
            {
                "q": "How does Random Forest Regression compare to Multiple Linear Regression?",
                "a": "MLR fits a flat global hyperplane, assumes homoscedasticity, can extrapolate linearly, and provides exact analytical coefficients. RF Regression fits localized orthogonal partitions, captures complex non-linear thresholds and interactions automatically, but cannot extrapolate beyond training extremes."
            },
            {
                "q": "How are residuals used to evaluate regression model quality?",
                "a": "Residuals e_i = y_i - ŷ_i should be plotted against predicted values ŷ_i. A well-fitted model displays a random, zero-centered horizontal band of residuals with constant spread (homoscedasticity) and a bell-shaped normal error histogram."
            }
        ],
        "practical_task": "Fit a RandomForestRegressor on diabetes.csv predicting Glucose, calculate R² and RMSE on test data, and plot actual vs predicted values.",
        "code_snippet": "from sklearn.ensemble import RandomForestRegressor\nfrom sklearn.metrics import mean_squared_error, r2_score\nimport pandas as pd, numpy as np\n\ndf = pd.read_csv('datasets/diabetes.csv')\nX, y = df.drop('Glucose', axis=1), df['Glucose']\nreg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)\npreds = reg.predict(X)\nprint(f'R2: {r2_score(y, preds):.4f}, RMSE: {np.sqrt(mean_squared_error(y, preds)):.2f}')"
    },
    "linear_svm": {
        "exp_num": "11",
        "title": "Linear Support Vector Machine (SVM)",
        "category": "Classification",
        "dataset": "data_banknote_authentication.csv",
        "definition": "A maximum-margin supervised classifier that constructs a linear separating hyperplane in feature space with the maximum geometric distance (margin) to the closest training points of any class (the Support Vectors).",
        "math_formula": "w^T x + b = 0, \\quad \\text{Margin Width} = \\frac{2}{\\|w\\|}, \\quad \\min_{w, b, \\xi} \\frac{1}{2} \\|w\\|^2 + C \\sum_{i=1}^m \\xi_i",
        "cost_function": "\\text{Hinge Loss with L2 Regularization}: J(w, b) = \\frac{1}{2} \\|w\\|^2 + C \\sum_{i=1}^m \\max(0, 1 - y_i(w^T x_i + b))",
        "math_intuition": "SVM searches for the safest possible decision boundary by maximizing the margin 2/||w|| between classes, which is mathematically equivalent to minimizing 1/2 ||w||². The slack variables ξ_i permit soft-margin classification for overlapping points. Hyperparameter C controls the trade-off: large C penalizes margin violations heavily (hard margin, narrower corridor), while small C prioritizes a wider margin with higher tolerance for misclassifications.",
        "workflow": [
            "1. Wavelet Data Ingestion: Ingest 4 continuous wavelet transform features (variance, skewness, curtosis, entropy).",
            "2. Standardization: Standardize features to zero mean and unit variance using StandardScaler.",
            "3. Convex Quadratic Programming: Solve dual Lagrangian formulation using sequential minimal optimization (SMO).",
            "4. Support Vector Extraction: Identify critical support vectors lying directly on or inside margin gutters.",
            "5. Evaluation: Evaluate accuracy, precision, and plot 2D hyperplane projections."
        ],
        "use_cases": [
            "Banknote authentication (Genuine vs Counterfeit currency detection).",
            "Text categorization and document classification with high-dimensional bag-of-words vectors.",
            "Cancer tissue classification using gene expression micro-arrays.",
            "Pedestrian and obstacle detection in computer vision."
        ],
        "assumptions": [
            "Data is approximately or strictly linearly separable in feature space.",
            "Features are standardized so that Euclidean margins are not distorted by scale differences.",
            "Target classes are binary (+1 and -1)."
        ],
        "advantages": [
            "Maximum-margin objective provides high generalization power and resistance to overfitting.",
            "Memory efficient: The decision function depends exclusively on a small subset of support vectors.",
            "Convex quadratic formulation guarantees a single unique global optimum without local minima.",
            "Highly effective in high-dimensional spaces where features exceed sample count."
        ],
        "limitations": [
            "Cannot model non-linear class distributions without kernel projection.",
            "Highly sensitive to feature scales (StandardScaler is strictly mandatory).",
            "Quadratic training complexity O(N² to N³) makes it slow on very large datasets (> 100,000 samples).",
            "Does not output native calibrated probabilities (requires Platt scaling)."
        ],
        "evaluation_metrics": [
            {"metric": "Accuracy (%)", "desc": "Overall authenticity classification accuracy on test banknotes."},
            {"metric": "Support Vector Count", "desc": "Number of critical boundary training instances."},
            {"metric": "Margin Width (2/||w||)", "desc": "Geometric width of separation corridor."},
            {"metric": "F1-Score (%)", "desc": "Harmonic mean of precision and recall for counterfeit detection."}
        ],
        "common_mistakes": [
            "Failing to standardize features before fitting Linear SVM, warping margin width.",
            "Setting C to an extremely large value (e.g. C=10000), resulting in an overfitted hard-margin boundary.",
            "Using Linear SVM when classes have concentric or non-linear boundaries (use RBF SVM instead).",
            "Interpreting decision function values as calibrated probabilities without Platt scaling."
        ],
        "how_prediction_works": "A banknote wavelet feature vector X = [3.62, 8.66, -2.80, -0.44] arrives. The fitted hyperplane has weights w and bias b. The decision function computes the signed geometric distance: f(x) = wᵀX + b = +2.45. Since f(x) >= 0, the banknote is classified as Class 1 (Genuine). If f(x) had been negative, it would be classified as Class 0 (Counterfeit).",
        "viva_questions": [
            {
                "q": "What are Support Vectors in an SVM?",
                "a": "Support vectors are the critical training data points that lie directly on the margin boundaries (y_i(wᵀx_i + b) = 1) or on the wrong side of the margin (slack ξ_i > 0). They are the only points that determine the position and orientation of the separating hyperplane; moving or removing any non-support vector leaves the decision boundary unchanged."
            },
            {
                "q": "What is the role of hyperparameter C in Support Vector Machines?",
                "a": "C is the regularization parameter governing the trade-off between maximizing margin width and minimizing classification errors. A large C heavily penalizes margin violations, leading to a hard, narrow margin (low bias, high variance, risk of overfitting). A small C allows more violations in exchange for a wider margin (high bias, low variance, higher tolerance to noise)."
            },
            {
                "q": "Why is feature scaling strictly mandatory for Support Vector Machines?",
                "a": "SVM maximizes the geometric margin 2/||w||, which depends directly on Euclidean distances in feature space. If features have different scales, the unscaled feature with the largest range will dominate the distance computation, distorting the hyperplane orientation."
            },
            {
                "q": "What is the Hinge Loss function?",
                "a": "Hinge loss is defined as L(y, f(x)) = max(0, 1 - y * f(x)), where y ∈ {-1, +1}. If an instance is correctly classified and outside the margin (y * f(x) >= 1), the loss is strictly 0. If it violates the margin or is misclassified, the loss increases linearly with distance from the margin."
            },
            {
                "q": "What is the difference between the Primal and Dual formulations of SVM?",
                "a": "The Primal formulation optimizes weights w and bias b directly in feature space (dimension p). The Dual formulation optimizes Lagrange multipliers α_i associated with each training sample (dimension N). The dual formulation allows the use of the Kernel Trick because it expresses optimization strictly in terms of inner products x_iᵀ x_j."
            }
        ],
        "practical_task": "Fit a Linear SVC on data_banknote_authentication.csv, report the number of support vectors per class, and calculate test accuracy.",
        "code_snippet": "from sklearn.svm import SVC\nfrom sklearn.preprocessing import StandardScaler\nimport pandas as pd\n\ndf = pd.read_csv('datasets/data_banknote_authentication.csv')\nX = StandardScaler().fit_transform(df.iloc[:, :-1])\nmodel = SVC(kernel='linear', C=1.0).fit(X, df.iloc[:, -1])\nprint('Support vectors count per class:', model.n_support_)"
    },
    "rbf_svm": {
        "exp_num": "12",
        "title": "Nonlinear Support Vector Machine (RBF SVM)",
        "category": "Classification",
        "dataset": "banknote_authentication_nonlinear_svm.csv",
        "definition": "A non-linear maximum-margin classifier that implicitly projects input vectors into an infinite-dimensional Hilbert feature space using the Radial Basis Function (Gaussian) kernel, enabling the construction of intricate, non-linear decision boundaries.",
        "math_formula": "K(x, x') = \\exp(-\\gamma \\|x - x'\\|^2), \\quad \\gamma = \\frac{1}{2\\sigma^2}, \\quad f(x) = \\text{sign}\\left( \\sum_{i \\in SV} \\alpha_i y_i K(x_i, x) + b \\right)",
        "cost_function": "\\text{Dual Optimization via Kernel Gram Matrix}: \\max_\\alpha \\sum_{i=1}^m \\alpha_i - \\frac{1}{2} \\sum_{i,j=1}^m \\alpha_i \\alpha_j y_i y_j K(x_i, x_j)",
        "math_intuition": "When data points cannot be separated by a flat line or plane in original feature space, Cover's Theorem states that mapping them into a higher dimension increases the probability of linear separability. The 'Kernel Trick' replaces the inner product ⟨φ(x), φ(x')⟩ in Hilbert space with the scalar evaluation K(x, x') in original space, bypassing the intractable computation of infinite coordinates. Gamma (γ) defines the influence radius of each support vector.",
        "workflow": [
            "1. Non-linear Data Ingestion: Ingest continuous wavelet transform features exhibiting non-linear boundaries.",
            "2. Feature Normalization: Scale features with StandardScaler to ensure symmetric Gaussian influence.",
            "3. Kernel & Hyperparameter Configuration: Set kernel='rbf', tuning regularization C and bandwidth gamma γ.",
            "4. Kernel Matrix Evaluation: Evaluate Gram operator K_ij = exp(-γ ||x_i - x_j||²).",
            "5. Boundary Projection: Construct non-linear decision contours and classify authentication records."
        ],
        "use_cases": [
            "Non-linear banknote authentication where genuine and counterfeit clusters interleave.",
            "Bioinformatics protein fold classification and remote homology detection.",
            "Optical character recognition (OCR) and handwritten digit recognition.",
            "Speech recognition phoneme boundary classification."
        ],
        "assumptions": [
            "Non-linear boundaries can be resolved by projecting into higher-dimensional Gaussian feature space.",
            "Features are standardized to prevent anisotropic Gaussian elongation.",
            "Appropriate selection of both C and γ (gamma) hyperparameters."
        ],
        "advantages": [
            "Can learn arbitrary, highly complex non-linear decision boundaries.",
            "Kernel trick avoids the computational cost of explicit high-dimensional feature mapping.",
            "Max-margin objective provides superior generalization over raw neural networks on small datasets.",
            "Global optimum is guaranteed by convex quadratic programming."
        ],
        "limitations": [
            "Hyperparameter sensitivity: Jointly tuning C and γ requires exhaustive grid search.",
            "O(N² to N³) training complexity makes it impractical for massive datasets (> 100,000 samples).",
            "Extremely difficult to interpret: The decision boundary is an implicit non-linear surface.",
            "Setting gamma γ too high causes severe overfitting, creating tiny isolated decision islands around training points."
        ],
        "evaluation_metrics": [
            {"metric": "Accuracy (%)", "desc": "Overall classification performance on non-linear banknote test fold."},
            {"metric": "ROC-AUC", "desc": "Discriminative power across varying decision score thresholds."},
            {"metric": "Gamma (γ)", "desc": "Gaussian curvature parameter: small γ = smooth, large γ = peaked."},
            {"metric": "F1-Score (%)", "desc": "Harmonic mean of precision and recall."}
        ],
        "common_mistakes": [
            "Setting gamma γ too high (e.g. gamma=100), creating isolated circular boundaries around individual points (overfitting).",
            "Setting gamma γ too low, causing the Gaussian bell to flatten into an underfitted linear approximation.",
            "Failing to standardize features before computing Euclidean distance in the RBF kernel.",
            "Using default hyperparameters without cross-validated Grid Search on C and gamma."
        ],
        "how_prediction_works": "Query point x arrives. For each support vector s_i, model evaluates Gaussian similarity: K(s_i, x) = exp(-γ ||s_i - x||²). If x is very close to s_i, K ≈ 1.0; if distant, K ≈ 0. The prediction multiplies similarities by learned dual weights: f(x) = Σ α_i * y_i * K(s_i, x) + b. If the sum is positive, classify as Genuine; if negative, classify as Counterfeit.",
        "viva_questions": [
            {
                "q": "What is the 'Kernel Trick' in Support Vector Machines?",
                "a": "The Kernel Trick allows computing the inner product ⟨φ(x), φ(x')⟩ in a high- or infinite-dimensional feature space directly using a kernel function K(x, x') evaluated in the original low-dimensional input space, completely bypassing the mathematically impossible task of explicitly transforming and storing infinite coordinates."
            },
            {
                "q": "What happens if gamma (γ) in an RBF SVM is set too high or too low?",
                "a": "Gamma defines the radius of influence of each support vector (γ = 1 / 2σ²). If γ is too high, the Gaussian curve is extremely narrow, meaning only points immediately adjacent to a support vector are affected, creating overfitted isolated islands. If γ is too low, the radius is huge, flattening the curve into an underfitted, virtually linear boundary."
            },
            {
                "q": "Why does the RBF kernel correspond to an infinite-dimensional feature space?",
                "a": "Using the Taylor series expansion of the exponential function e^z = Σ (z^k / k!), the Gaussian kernel exp(-γ ||x - x'||²) can be expanded into an infinite polynomial sum of all possible feature degree interactions up to infinity, meaning its implicit feature map φ(x) has infinite dimensions."
            },
            {
                "q": "How do you choose between Linear SVM and RBF SVM?",
                "a": "Use Linear SVM when the number of features p is very large compared to samples N (e.g. text classification where p > 10,000), as data is often already linearly separable in high dimensions. Use RBF SVM when p is small to moderate, the dataset is complex, and linear boundaries yield poor accuracy."
            },
            {
                "q": "What is the difference between hard-margin and soft-margin SVM?",
                "a": "Hard-margin SVM strictly forbids any training misclassifications or margin violations (requires perfect linear separability). Soft-margin SVM introduces slack variables ξ_i controlled by penalty parameter C, allowing some points to violate the margin or be misclassified to achieve a wider, more generalizable margin on noisy data."
            }
        ],
        "practical_task": "Train an RBF SVC on banknote_authentication_nonlinear_svm.csv with C=1.0 and gamma='scale', and compare its test accuracy against Linear SVM.",
        "code_snippet": "from sklearn.svm import SVC\nfrom sklearn.preprocessing import StandardScaler\nimport pandas as pd\n\ndf = pd.read_csv('datasets/banknote_authentication_nonlinear_svm.csv')\nX = StandardScaler().fit_transform(df.iloc[:, :-1])\nrbf = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X, df.iloc[:, -1])\nprint(f'RBF SVM Accuracy: {rbf.score(X, df.iloc[:, -1]):.4f}')"
    }
}

def get_learning_content(key):
    """Returns the complete structured learning module dictionary for any experiment key."""
    if key in LEARNING_DATA:
        return LEARNING_DATA[key]
    
    # Fallback default
    return {
        "exp_num": "01",
        "title": key.upper(),
        "category": "Machine Learning",
        "dataset": "Laboratory Dataset",
        "definition": "Machine learning algorithm for predictive modeling and empirical analysis.",
        "math_formula": "f(X) = \\hat{y}",
        "cost_function": "Empirical Risk Minimization",
        "math_intuition": "Optimizes parameters on training data to generalize to unseen test instances.",
        "workflow": [
            "1. Data Ingestion & Preprocessing.",
            "2. Train/Test Partition with zero leakage.",
            "3. Model Parameter Optimization.",
            "4. Held-out Evaluation."
        ],
        "use_cases": ["Predictive analytics in automated decision workflows."],
        "assumptions": ["Independent and identically distributed samples."],
        "advantages": ["Automates pattern recognition from empirical data."],
        "limitations": ["Depends on data quality, hyperparameter choice, and sample size."],
        "evaluation_metrics": [{"metric": "Accuracy / R²", "desc": "Standard task-appropriate performance metric."}],
        "common_mistakes": ["Evaluating models on training data instead of held-out test data."],
        "how_prediction_works": "Transforms input features using learned mathematical parameters.",
        "viva_questions": [
            {
                "q": "What is the Bias-Variance Tradeoff?",
                "a": "Bias is error from erroneous assumptions (underfitting). Variance is error from sensitivity to small fluctuations in the training set (overfitting). Optimal model complexity minimizes the sum of Bias² + Variance + Irreducible Error."
            }
        ],
        "practical_task": "Load dataset, train model, and evaluate performance on held-out test data.",
        "code_snippet": "# Python ML implementation script"
    }
