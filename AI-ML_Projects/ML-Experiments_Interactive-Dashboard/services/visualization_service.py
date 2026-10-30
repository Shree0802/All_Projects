import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_text

# Common chart layout styling matching our navy/teal laboratory theme
THEME_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'Inter', sans-serif", color="#1e293b", size=11),
    margin=dict(l=45, r=30, t=50, b=45),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=10)
    )
)

def _fig_to_dict(fig):
    return json.loads(fig.to_json())

def make_confusion_matrix_chart(cm, classes, title="Confusion Matrix"):
    classes_str = [str(c) for c in classes]
    z = np.array(cm)
    total = np.sum(z)
    text_annotations = []
    for i in range(len(classes)):
        row_text = []
        for j in range(len(classes)):
            val = z[i][j]
            pct = round((val / total) * 100, 1) if total > 0 else 0
            row_text.append(f"<b>{val}</b><br>({pct}%)")
        text_annotations.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"Pred: {c}" for c in classes_str],
        y=[f"Actual: {c}" for c in classes_str],
        colorscale=[[0, "#f0fdfa"], [0.5, "#14b8a6"], [1, "#0f766e"]],
        text=text_annotations,
        texttemplate="%{text}",
        textfont={"size": 12, "color": "#0f172a"},
        showscale=True
    ))
    fig.update_layout(
        title=title,
        height=360,
        **THEME_LAYOUT
    )
    return _fig_to_dict(fig)

def make_classification_report_chart(report, classes, title="Per-Class Performance (Precision, Recall, F1)"):
    classes_str = [str(c) for c in classes]
    precisions = [round(report.get(c, {}).get("precision", 0) * 100, 1) for c in classes_str]
    recalls = [round(report.get(c, {}).get("recall", 0) * 100, 1) for c in classes_str]
    f1s = [round(report.get(c, {}).get("f1-score", 0) * 100, 1) for c in classes_str]

    fig = go.Figure(data=[
        go.Bar(name="Precision", x=classes_str, y=precisions, marker_color="#0d9488"),
        go.Bar(name="Recall", x=classes_str, y=recalls, marker_color="#0284c7"),
        go.Bar(name="F1-Score", x=classes_str, y=f1s, marker_color="#6366f1")
    ])
    fig.update_layout(
        barmode="group",
        title=title,
        yaxis=dict(title="Score (%)", range=[0, 105]),
        xaxis=dict(title="Class Label"),
        height=340,
        **THEME_LAYOUT
    )
    return _fig_to_dict(fig)

def make_roc_curve_chart(roc_data, title="Receiver Operating Characteristic (ROC) Curve"):
    if not roc_data:
        return None
    fig = go.Figure()
    # Baseline diagonal (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="#94a3b8", width=1.5),
        name="Random Classifier (AUC = 0.5)"
    ))
    # Model ROC
    fig.add_trace(go.Scatter(
        x=roc_data["fpr"], y=roc_data["tpr"],
        mode="lines",
        line=dict(color="#0d9488", width=2.5),
        name=f"Trained Model (AUC = {roc_data['auc']})",
        fill="tozeroy",
        fillcolor="rgba(13, 148, 136, 0.1)"
    ))
    fig.update_layout(
        title=f"{title} · AUC: {roc_data['auc']}",
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[0, 1]),
        yaxis=dict(title="True Positive Rate (Sensitivity / Recall)", range=[0, 1.05]),
        height=360,
        **THEME_LAYOUT
    )
    return _fig_to_dict(fig)

def make_pr_curve_chart(pr_data, title="Precision-Recall Curve"):
    if not pr_data:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pr_data["recall"], y=pr_data["precision"],
        mode="lines",
        line=dict(color="#0284c7", width=2.5),
        name=f"Model (AP = {pr_data['ap']})",
        fill="tozeroy",
        fillcolor="rgba(2, 132, 199, 0.1)"
    ))
    fig.update_layout(
        title=f"{title} · Average Precision: {pr_data['ap']}",
        xaxis=dict(title="Recall", range=[0, 1]),
        yaxis=dict(title="Precision", range=[0, 1.05]),
        height=360,
        **THEME_LAYOUT
    )
    return _fig_to_dict(fig)

def make_actual_vs_predicted_chart(y_true, y_pred, target_name="Target"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    padding = (max_val - min_val) * 0.05
    ref_line = [min_val - padding, max_val + padding]

    fig = go.Figure()
    # Ideal 45-degree line
    fig.add_trace(go.Scatter(
        x=ref_line, y=ref_line,
        mode="lines",
        line=dict(dash="dash", color="#94a3b8", width=1.5),
        name="Ideal Fit (y = x)"
    ))
    # Predictions scatter
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        marker=dict(color="#0d9488", size=7, opacity=0.8, line=dict(width=1, color="#0f766e")),
        name="Test Observations"
    ))
    fig.update_layout(
        title=f"Actual vs. Predicted Plot ({target_name})",
        xaxis=dict(title=f"Actual {target_name}"),
        yaxis=dict(title=f"Predicted {target_name}"),
        height=360,
        **THEME_LAYOUT
    )
    return _fig_to_dict(fig)

def make_residual_plot(y_pred, residuals, title="Residual Plot (Errors vs. Predicted)"):
    fig = go.Figure()
    # Zero error horizontal line
    fig.add_hline(y=0, line_dash="dash", line_color="#ef4444", line_width=1.5, annotation_text="Zero Residual Line", annotation_position="top right")
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode="markers",
        marker=dict(color="#0284c7", size=7, opacity=0.75, line=dict(width=1, color="#0369a1")),
        name="Residuals (y - ŷ)"
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Predicted Value (ŷ)"),
        yaxis=dict(title="Residual Error (y - ŷ)"),
        height=360,
        **THEME_LAYOUT
    )
    return _fig_to_dict(fig)

def make_error_distribution_chart(residuals, title="Prediction Error (Residuals) Distribution"):
    fig = px.histogram(
        x=residuals, nbins=25, marginal="box",
        title=title,
        color_discrete_sequence=["#0d9488"]
    )
    fig.update_layout(
        xaxis=dict(title="Residual Error"),
        yaxis=dict(title="Frequency"),
        height=340,
        **THEME_LAYOUT
    )
    return _fig_to_dict(fig)

def make_feature_importance_chart(feature_names, importances, title="Feature Importance"):
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=True)

    fig = px.bar(
        df_imp, x="Importance", y="Feature", orientation="h",
        title=title,
        color="Importance",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(
        xaxis=dict(title="Model-Derived Importance"),
        yaxis=dict(title="Feature"),
        height=max(320, len(feature_names) * 26),
        **THEME_LAYOUT
    )
    return _fig_to_dict(fig)

# Experiment 01: Simple Linear Regression Plot
def make_slr_regression_plot(X_test, y_test, model):
    x_vals = X_test.iloc[:, 0] if hasattr(X_test, "iloc") else X_test[:, 0]
    y_vals = y_test.values if hasattr(y_test, "values") else y_test

    # Line points
    x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
    line_x = np.linspace(x_min, x_max, 100)
    line_frame = pd.DataFrame(line_x, columns=["cgpa"])
    line_y = model.predict(line_frame)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="markers",
        name="Actual Observations",
        marker=dict(color="#0284c7", size=8, opacity=0.8)
    ))
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y,
        mode="lines",
        name="Fitted OLS Regression Line",
        line=dict(color="#0d9488", width=3)
    ))
    fig.update_layout(
        title="Simple Linear Regression: CGPA vs. Placement Package",
        xaxis=dict(title="Undergraduate CGPA (Independent Feature)"),
        yaxis=dict(title="Offered Package in LPA (Target Variable)"),
        height=380,
        **THEME_LAYOUT
    )
    return _fig_to_dict(fig)

# Experiment 03: 2D Logistic Regression Decision Boundary
def make_logistic_2d_decision_boundary(X_test, y_test, model, feature_names=("cgpa", "iq")):
    x0 = X_test[feature_names[0]].values
    x1 = X_test[feature_names[1]].values

    x0_min, x0_max = x0.min() - 0.5, x0.max() + 0.5
    x1_min, x1_max = x1.min() - 5, x1.max() + 5
    xx0, xx1 = np.meshgrid(np.linspace(x0_min, x0_max, 80), np.linspace(x1_min, x1_max, 80))

    grid_df = pd.DataFrame({
        feature_names[0]: xx0.ravel(),
        feature_names[1]: xx1.ravel()
    })

    if hasattr(model, "predict_proba"):
        zz = model.predict_proba(grid_df)[:, 1].reshape(xx0.shape)
    else:
        zz = model.predict(grid_df).reshape(xx0.shape)

    fig = go.Figure()
    # Decision boundary contour
    fig.add_trace(go.Contour(
        x=np.linspace(x0_min, x0_max, 80),
        y=np.linspace(x1_min, x1_max, 80),
        z=zz,
        colorscale=[[0, "#e0f2fe"], [0.5, "#ffffff"], [1, "#ccfbf1"]],
        opacity=0.6,
        showscale=False,
        contours=dict(start=0.5, end=0.5, size=0.5, coloring="heatmap")
    ))
    # Scatter observations
    for target_val, label, color in [(0, "Not Likely to be Placed (0)", "#e11d48"), (1, "Likely to be Placed (1)", "#0d9488")]:
        mask = (y_test == target_val)
        fig.add_trace(go.Scatter(
            x=x0[mask], y=x1[mask],
            mode="markers",
            name=label,
            marker=dict(color=color, size=8, line=dict(width=1, color="#ffffff"))
        ))

    fig.update_layout(
        title="2D Logistic Regression Decision Boundary (CGPA vs. IQ)",
        xaxis=dict(title="CGPA"),
        yaxis=dict(title="IQ Score"),
        height=380,
        **THEME_LAYOUT
    )
    return _fig_to_dict(fig)

# Experiment 06: KNN K-Value Cross Validation Curve
def make_knn_k_curve(X_train, y_train):
    k_range = list(range(1, 16, 2))
    cv_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
        cv_scores.append(round(float(scores.mean()) * 100, 2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_range, y=cv_scores,
        mode="lines+markers",
        name="5-Fold CV Accuracy",
        line=dict(color="#0d9488", width=2.5),
        marker=dict(size=8, color="#0f766e")
    ))
    best_idx = int(np.argmax(cv_scores))
    best_k = k_range[best_idx]
    best_score = cv_scores[best_idx]

    fig.add_annotation(
        x=best_k, y=best_score,
        text=f"Optimal K={best_k} ({best_score}%)",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#0f766e",
        font=dict(size=10, color="#0f766e")
    )

    fig.update_layout(
        title="KNN Hyperparameter Optimization: K-Value vs. Cross-Validation Accuracy",
        xaxis=dict(title="Number of Neighbors (K)", tickmode="linear", dtick=2),
        yaxis=dict(title="Cross-Validation Accuracy (%)"),
        height=350,
        **THEME_LAYOUT
    )
    return _fig_to_dict(fig)

# Experiment 05: Decision Tree Visual Text/Hierarchy Representation
def make_tree_structure_summary(tree_model, feature_names, max_depth=4):
    try:
        txt = export_text(tree_model, feature_names=list(feature_names), max_depth=max_depth)
        return {
            "text_representation": txt,
            "max_depth": int(tree_model.get_depth()),
            "n_leaves": int(tree_model.get_n_leaves()),
            "n_features": int(tree_model.n_features_in_)
        }
    except Exception:
        return None

# Experiment 10 & 11: 2D SVM Decision Boundary
def make_svm_2d_boundary(X_test, y_test, model, feature_names=("variance", "skewness"), title="SVM Decision Boundary"):
    x0 = X_test[feature_names[0]].values
    x1 = X_test[feature_names[1]].values

    x0_min, x0_max = x0.min() - 0.5, x0.max() + 0.5
    x1_min, x1_max = x1.min() - 0.5, x1.max() + 0.5
    xx0, xx1 = np.meshgrid(np.linspace(x0_min, x0_max, 80), np.linspace(x1_min, x1_max, 80))

    # Fix remaining 2 features to median if model expects 4 features
    all_features = ["variance", "skewness", "curtosis", "entropy"]
    grid_df = pd.DataFrame({
        "variance": xx0.ravel(),
        "skewness": xx1.ravel(),
        "curtosis": np.zeros(xx0.size),
        "entropy": np.zeros(xx0.size)
    })

    try:
        zz = model.predict(grid_df).reshape(xx0.shape)
    except Exception:
        zz = np.zeros(xx0.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=np.linspace(x0_min, x0_max, 80),
        y=np.linspace(x1_min, x1_max, 80),
        z=zz,
        colorscale=[[0, "#e0f2fe"], [1, "#ccfbf1"]],
        opacity=0.55,
        showscale=False
    ))
    for target_val, label, color in [(0, "Genuine Banknote (0)", "#0d9488"), (1, "Counterfeit Banknote (1)", "#e11d48")]:
        mask = (y_test == target_val)
        fig.add_trace(go.Scatter(
            x=x0[mask], y=x1[mask],
            mode="markers",
            name=label,
            marker=dict(color=color, size=7, line=dict(width=1, color="#ffffff"))
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Wavelet Variance (Projection)"),
        yaxis=dict(title="Wavelet Skewness (Projection)"),
        height=380,
        **THEME_LAYOUT
    )
    return _fig_to_dict(fig)
