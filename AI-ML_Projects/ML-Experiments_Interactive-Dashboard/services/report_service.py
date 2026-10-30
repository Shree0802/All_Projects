import io
import os
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from services import model_service as ms
from services import training_service as ts
from services import explainability_service as xs

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
LOGO_PATH = STATIC_DIR / "images" / "college-logo.png"

def generate_pdf_report(key, custom_profile=None):
    """
    Generates a formal Academic Laboratory Experiment Report in PDF format.
    Includes:
    - Institutional header & logo
    - Student credentials
    - Experiment metadata & theoretical foundations
    - Dataset properties & preprocessing pipeline
    - Hyperparameters & held-out test evaluation metrics
    - High-res embedded visualizations (Confusion matrix / Actual vs Predicted)
    - Explainable AI summary & student observations
    - Practical conclusions & academic references
    """
    # 1. Retrieve or run experiment evaluation
    run_bundle = ts.get_latest_run(key)
    if not run_bundle:
        run_bundle = ts.run_experiment_pipeline(key)

    metadata = run_bundle["metadata"]
    metrics = run_bundle["metrics"]
    schema = ts.EXPERIMENT_SCHEMAS.get(key, {})
    algo_spec = ms.ALGORITHMS.get(key, {})
    xai_info = xs.get_algo_explanation(key)

    # 2. Student profile resolution
    profile = dict(ms.STUDENT_PROFILE)
    if custom_profile:
        for k, v in custom_profile.items():
            if v and v.strip():
                profile[k] = v.strip()

    # 3. Setup ReportLab PDF document
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    # 4. Color Palette
    PRIMARY = colors.HexColor("#0f172a")     # Deep Navy
    SECONDARY = colors.HexColor("#0d9488")   # Teal
    ACCENT = colors.HexColor("#0284c7")      # Blue
    TEXT_DARK = colors.HexColor("#1e293b")   # Slate
    BG_LIGHT = colors.HexColor("#f8fafc")    # Off-white
    BORDER_COLOR = colors.HexColor("#cbd5e1") # Subtle grey

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=PRIMARY,
        alignment=1 # Center
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=SECONDARY,
        alignment=1
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=PRIMARY,
        spaceBefore=10,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=TEXT_DARK
    )
    bold_body = ParagraphStyle(
        "BoldBody",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    story = []

    # --- HEADER: College Logo & Details ---
    logo_img = None
    if LOGO_PATH.exists():
        try:
            # 65x65 px logo with contain aspect ratio
            logo_img = Image(str(LOGO_PATH), width=0.85*inch, height=0.85*inch)
        except Exception:
            logo_img = None

    header_text = [
        Paragraph(f"<b>{profile['college'].upper()}</b>", title_style),
        Paragraph(f"{profile['location']} | Department of {profile['course']}", subtitle_style),
        Paragraph(f"<b>{profile['subject_lab'].upper()} - EXPERIMENT REPORT</b>", ParagraphStyle(
            "LabTitle", parent=subtitle_style, fontSize=9, textColor=ACCENT, spaceBefore=2
        ))
    ]

    if logo_img:
        header_table = Table([[logo_img, header_text]], colWidths=[1.1*inch, 6.2*inch])
        header_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (0, 0), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4)
        ]))
        story.append(header_table)
    else:
        for p in header_text:
            story.append(p)

    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=8))

    # --- STUDENT & EXPERIMENT CREDENTIALS TABLE ---
    exp_num = algo_spec.get("num", "01")
    exec_date = datetime.now().strftime("%B %d, %Y, %I:%M %p")

    cred_data = [
        [
            Paragraph("<b>Student Name:</b>", body_style), Paragraph(profile["name"], bold_body),
            Paragraph("<b>Experiment No:</b>", body_style), Paragraph(f"Experiment {exp_num}", bold_body)
        ],
        [
            Paragraph("<b>Class / Roll No:</b>", body_style), Paragraph(f"{profile['class']} / {profile['roll_no']}", body_style),
            Paragraph("<b>Execution Date:</b>", body_style), Paragraph(exec_date, body_style)
        ],
        [
            Paragraph("<b>Academic Subject:</b>", body_style), Paragraph(profile["subject"], body_style),
            Paragraph("<b>Algorithm / Task:</b>", body_style), Paragraph(f"{algo_spec.get('title', 'ML Algorithm')} ({schema.get('category')})", body_style)
        ]
    ]
    cred_table = Table(cred_data, colWidths=[1.3*inch, 2.3*inch, 1.4*inch, 2.3*inch])
    cred_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.75, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(cred_table)
    story.append(Spacer(1, 8))

    # --- SECTION 1: AIM & OBJECTIVES ---
    story.append(Paragraph("1. AIM & LEARNING OBJECTIVES", h1_style))
    aim_text = f"<b>Aim:</b> To implement, train, evaluate, and interpret <b>{algo_spec.get('title')}</b> for {schema.get('category').lower()} on the <i>{schema.get('dataset_file')}</i> benchmark dataset, assessing held-out generalization metrics and decision characteristics."
    story.append(Paragraph(aim_text, body_style))
    story.append(Spacer(1, 3))
    obj_text = "<b>Objectives:</b> (1) Preprocess raw data without data leakage. (2) Perform train/test partition. (3) Fit mathematical model parameters. (4) Quantify empirical test performance. (5) Extract explainable model insights."
    story.append(Paragraph(obj_text, body_style))
    story.append(Spacer(1, 8))

    # --- SECTION 2: ALGORITHM THEORY ---
    story.append(Paragraph("2. ALGORITHMIC & MATHEMATICAL FOUNDATIONS", h1_style))
    story.append(Paragraph(f"<b>Intuitive Concept:</b> {xai_info.get('beginner')}", body_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph(f"<b>Mathematical Formulation:</b> {xai_info.get('technical')}", body_style))
    story.append(Spacer(1, 8))

    # --- SECTION 3: DATASET SPECIFICATIONS & PREPROCESSING ---
    story.append(Paragraph("3. DATASET & PREPROCESSING METHODOLOGY", h1_style))
    ds_rows = [
        [Paragraph("<b>Benchmark Dataset:</b>", body_style), Paragraph(str(metadata.get("dataset")), body_style),
         Paragraph("<b>Target Variable:</b>", body_style), Paragraph(f"{metadata.get('target')} ({schema.get('target_label')})", body_style)],
        [Paragraph("<b>Sample Count:</b>", body_style), Paragraph(f"{metadata.get('train_samples', 0) + metadata.get('test_samples', 0)} records", body_style),
         Paragraph("<b>Train/Test Split:</b>", body_style), Paragraph(f"{metadata.get('train_samples')} train / {metadata.get('test_samples')} test ({int((1-metadata.get('test_size', 0.2))*100)}:{int(metadata.get('test_size', 0.2)*100)})", body_style)],
        [Paragraph("<b>Input Features:</b>", body_style), Paragraph(", ".join([f.replace("_", " ") for f in schema.get("features", [])[:6]]) + ("..." if len(schema.get("features", [])) > 6 else ""), body_style),
         Paragraph("<b>Leakage Policy:</b>", body_style), Paragraph("Fitted exclusively on training fold; zero test data leakage", body_style)]
    ]
    ds_table = Table(ds_rows, colWidths=[1.4*inch, 2.2*inch, 1.4*inch, 2.3*inch])
    ds_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(ds_table)
    story.append(Spacer(1, 8))

    # --- SECTION 4: EVALUATION METRICS TABLE ---
    story.append(Paragraph("4. HELD-OUT TEST EVALUATION METRICS", h1_style))
    
    is_regression = schema.get("category") == "Regression"
    if is_regression:
        metric_headers = [Paragraph("<b>Metric Name</b>", bold_body), Paragraph("<b>Empirical Value</b>", bold_body), Paragraph("<b>Evaluation Criterion</b>", bold_body)]
        metric_rows = [
            metric_headers,
            [Paragraph("R² Coefficient of Determination", body_style), Paragraph(f"{metrics.get('r2_score', 'N/A')}", bold_body), Paragraph("Proportion of variance explained (Higher is better, Max 1.0)", body_style)],
            [Paragraph("Mean Absolute Error (MAE)", body_style), Paragraph(f"{metrics.get('mae', 'N/A')}", bold_body), Paragraph("Average magnitude of errors in target units (Lower is better)", body_style)],
            [Paragraph("Root Mean Squared Error (RMSE)", body_style), Paragraph(f"{metrics.get('rmse', 'N/A')}", bold_body), Paragraph("Penalizes large outlier residuals quadratically (Lower is better)", body_style)],
            [Paragraph("Mean Squared Error (MSE)", body_style), Paragraph(f"{metrics.get('mse', 'N/A')}", bold_body), Paragraph("Variance of estimation errors (Lower is better)", body_style)]
        ]
    else:
        metric_headers = [Paragraph("<b>Metric Name</b>", bold_body), Paragraph("<b>Score (%)</b>", bold_body), Paragraph("<b>Evaluation Criterion</b>", bold_body)]
        metric_rows = [
            metric_headers,
            [Paragraph("Overall Accuracy", body_style), Paragraph(f"{metrics.get('accuracy', 'N/A')}%", bold_body), Paragraph("Overall correct classifications over total test samples", body_style)],
            [Paragraph("Precision (Weighted)", body_style), Paragraph(f"{metrics.get('precision', 'N/A')}%", bold_body), Paragraph("Ratio of true positives over total predicted positives", body_style)],
            [Paragraph("Recall / Sensitivity (Weighted)", body_style), Paragraph(f"{metrics.get('recall', 'N/A')}%", bold_body), Paragraph("Ratio of true positives identified over actual positives", body_style)],
            [Paragraph("F1-Score (Harmonic Mean)", body_style), Paragraph(f"{metrics.get('f1_score', 'N/A')}%", bold_body), Paragraph("Harmonic mean of precision and recall (robust to imbalance)", body_style)]
        ]
        if "roc_auc" in metrics:
            metric_rows.append([
                Paragraph("Area Under ROC Curve (AUC)", body_style),
                Paragraph(f"{metrics.get('roc_auc')}", bold_body),
                Paragraph("Discriminative capability across all classification thresholds", body_style)
            ])

    eval_table = Table(metric_rows, colWidths=[2.2*inch, 1.3*inch, 3.8*inch])
    eval_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("TOPPADDING", (0, 0), (-1, 0), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ("BOX", (0, 0), (-1, -1), 0.75, PRIMARY),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(eval_table)
    story.append(Spacer(1, 10))

    # --- SECTION 5: EMBEDDED VISUALIZATION CHART ---
    story.append(Paragraph("5. SCIENTIFIC VISUALIZATION", h1_style))
    try:
        # Create a dedicated matplotlib figure to embed into PDF
        fig, ax = plt.subplots(figsize=(6.8, 2.7), dpi=150)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f8fafc")

        if is_regression:
            # Actual vs Predicted plot
            y_pred = run_bundle.get("y_pred", [])
            y_test = run_bundle.get("y_test", [])
            if len(y_test) > 0 and len(y_pred) > 0:
                ax.scatter(y_test, y_pred, color="#0d9488", alpha=0.8, edgecolors="#0f172a", s=35, label="Test Observations")
                min_v = min(min(y_test), min(y_pred))
                max_v = max(max(y_test), max(y_pred))
                ax.plot([min_v, max_v], [min_v, max_v], color="#ef4444", linestyle="--", linewidth=1.5, label="Ideal y = x")
                ax.set_title(f"Actual vs. Predicted ({schema.get('target_label')})", fontsize=10, fontweight="bold", color="#0f172a")
                ax.set_xlabel("Ground Truth (Actual Test)", fontsize=8.5)
                ax.set_ylabel("Model Prediction", fontsize=8.5)
                ax.legend(fontsize=8, frameon=True)
                ax.grid(True, linestyle=":", alpha=0.6)
            else:
                ax.text(0.5, 0.5, "Regression Evaluation Data", ha="center", va="center")
        else:
            # Confusion Matrix heatmap
            cm = metrics.get("confusion_matrix")
            if cm:
                cm_arr = np.array(cm)
                sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 9, "weight": "bold"})
                ax.set_title(f"Confusion Matrix ({algo_spec.get('title')})", fontsize=10, fontweight="bold", color="#0f172a")
                ax.set_xlabel("Predicted Class", fontsize=8.5)
                ax.set_ylabel("True Class", fontsize=8.5)
            else:
                ax.text(0.5, 0.5, "Classification Matrix", ha="center", va="center")

        plt.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        img_buffer.seek(0)

        chart_flowable = Image(img_buffer, width=6.8*inch, height=2.7*inch)
        story.append(chart_flowable)
    except Exception as exc:
        story.append(Paragraph(f"<i>Visualization generation omitted: {str(exc)}</i>", body_style))

    story.append(Spacer(1, 8))

    # --- SECTION 6: EXPLAINABLE AI & CONCLUSION ---
    story.append(Paragraph("6. EXPLAINABILITY & PRACTICAL OBSERVATIONS", h1_style))
    story.append(Paragraph(f"<b>Practical Impact:</b> {xai_info.get('practical')}", body_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph(f"<b>Model Limitations:</b> {xai_info.get('limitations')}", body_style))
    story.append(Spacer(1, 6))

    # Conclusion & Signatures
    story.append(Paragraph("7. CONCLUSION & SIGN-OFF", h1_style))
    conclusion_text = f"The <b>{algo_spec.get('title')}</b> model was successfully trained on held-out test partitions without data leakage. The model achieved reliable empirical convergence with an execution latency of {metadata.get('duration_ms', 0)} ms. All evaluation metrics have been recorded for academic submission."
    story.append(Paragraph(conclusion_text, body_style))
    story.append(Spacer(1, 14))

    # Signature blocks
    sig_data = [
        [Paragraph("____________________________<br/><b>Student Signature</b><br/>" + profile["name"], body_style),
         Paragraph("____________________________<br/><b>Laboratory Faculty In-Charge</b><br/>Department of AIML", body_style)]
    ]
    sig_table = Table(sig_data, colWidths=[3.6*inch, 3.7*inch])
    sig_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (0, 0), "LEFT"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    story.append(sig_table)

    # Build PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
