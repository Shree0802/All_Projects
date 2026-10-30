import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit

def list_datasets():
    """Returns a list of all CSV datasets available in the datasets directory."""
    files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith(".csv")])
    result = []
    for f in files:
        file_path = DATASET_DIR / f
        try:
            df = pd.read_csv(file_path)
            clean_cols = [c for c in df.columns if not str(c).startswith("Unnamed")]
            result.append({
                "filename": f,
                "rows": len(df),
                "columns": len(clean_cols),
                "size_kb": round(file_path.stat().st_size / 1024, 1),
                "preview_cols": clean_cols[:5]
            })
        except Exception:
            continue
    return result

def save_uploaded_dataset(file_storage):
    """
    Safely validates and stores an uploaded CSV dataset:
    - Verifies .csv extension
    - Checks file size limits (<= 10MB)
    - Validates CSV structure (non-empty, at least 2 rows, >= 2 columns)
    - Sanitizes filename to prevent directory traversal
    """
    if not file_storage or not file_storage.filename:
        raise ValueError("No file was uploaded.")

    filename = secure_filename(file_storage.filename)
    if not filename.lower().endswith(".csv"):
        raise ValueError("Invalid file format. Only .csv files are supported.")

    # Read content to check size
    file_bytes = file_storage.read()
    if len(file_bytes) == 0:
        raise ValueError("The uploaded CSV file is empty.")
    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError(f"File size exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_BYTES / (1024*1024)} MB.")

    # Validate parseability
    try:
        import io
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file: {str(exc)}")

    if len(df) < 5:
        raise ValueError("Dataset must contain at least 5 rows for training and evaluation.")
    if len(df.columns) < 2:
        raise ValueError("Dataset must contain at least 2 columns (features and target).")

    # Save to datasets dir
    clean_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)
    target_path = DATASET_DIR / clean_name
    with open(target_path, "wb") as f:
        f.write(file_bytes)

    return clean_name

def get_dataset_schema(filename):
    """
    Returns column types, numeric vs categorical lists, and potential target columns.
    """
    file_path = DATASET_DIR / filename
    if not file_path.exists() or not filename.endswith(".csv"):
        raise FileNotFoundError(f"Dataset '{filename}' does not exist.")

    df = pd.read_csv(file_path)
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    return {
        "filename": filename,
        "rows": len(df),
        "columns": list(df.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "recommended_targets": [c for c in df.columns if c.lower() in ("target", "class", "diagnosis", "outcome", "placement", "package", "label", "glucose")] or list(df.columns)[-1:]
    }

def get_dataset_details(filename, page=1, per_page=15, search=""):
    """
    Computes summary metadata, data types, missing values, duplicates,
    descriptive statistics, and paginated rows for the requested dataset.
    """
    file_path = DATASET_DIR / filename
    if not file_path.exists() or not filename.endswith(".csv"):
        raise FileNotFoundError(f"Dataset '{filename}' does not exist.")

    df = pd.read_csv(file_path)
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]

    total_rows = len(df)
    total_cols = len(df.columns)
    duplicate_count = int(df.duplicated().sum())

    # Column info (types & nulls)
    columns_info = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0
        columns_info.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "missing": null_count,
            "missing_pct": null_pct,
            "unique": int(df[col].nunique())
        })

    # Descriptive statistics
    numeric_df = df.select_dtypes(include=[np.number])
    stats_dict = {}
    if not numeric_df.empty:
        desc = numeric_df.describe().T
        for col_name, row in desc.iterrows():
            stats_dict[col_name] = {
                "count": int(row["count"]),
                "mean": round(float(row["mean"]), 3),
                "std": round(float(row["std"]), 3) if not np.isnan(row["std"]) else 0.0,
                "min": round(float(row["min"]), 3),
                "25%": round(float(row["25%"]), 3),
                "50%": round(float(row["50%"]), 3),
                "75%": round(float(row["75%"]), 3),
                "max": round(float(row["max"]), 3)
            }

    # Filtering / Search
    filtered_df = df
    if search:
        search_lower = str(search).lower()
        mask = filtered_df.astype(str).apply(lambda row: row.str.lower().str.contains(search_lower, regex=False).any(), axis=1)
        filtered_df = filtered_df[mask]

    filtered_count = len(filtered_df)
    total_pages = max(1, (filtered_count + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_rows = filtered_df.iloc[start_idx:end_idx].to_dict(orient="records")

    return {
        "filename": filename,
        "rows": total_rows,
        "columns": total_cols,
        "column_names": list(df.columns),
        "columns_info": columns_info,
        "duplicates": duplicate_count,
        "descriptive_stats": stats_dict,
        "rows_data": page_rows,
        "page": page,
        "total_pages": total_pages,
        "filtered_count": filtered_count,
        "per_page": per_page
    }

def get_dataset_charts(filename):
    """
    Generates interactive Plotly JSON specifications for:
    1. Feature Distributions / Histograms
    2. Correlation Heatmap
    3. Categorical / Target Distribution
    4. Box Plots for Outliers
    """
    file_path = DATASET_DIR / filename
    if not file_path.exists():
        return {}

    df = pd.read_csv(file_path)
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
    charts = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # 1. Correlation Heatmap (if >= 2 numeric columns)
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(3)
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="Viridis",
            zmin=-1, zmax=1,
            text=corr.values,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        fig_corr.update_layout(
            title="Feature Correlation Matrix (Pearson)",
            margin=dict(l=40, r=40, t=50, b=40),
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        charts["correlation_heatmap"] = json.loads(fig_corr.to_json())

    # 2. Target / Discrete Distribution Chart
    candidate_targets = [c for c in df.columns if c.lower() in ("class", "target", "diagnosis", "outcome", "placement", "label", "card_assigned")]
    target_col = candidate_targets[0] if candidate_targets else (categorical_cols[0] if categorical_cols else None)

    if not target_col and len(numeric_cols) > 0:
        for c in numeric_cols:
            if df[c].nunique() <= 5:
                target_col = c
                break

    if target_col:
        counts = df[target_col].value_counts().reset_index()
        counts.columns = [target_col, "count"]
        fig_dist = px.bar(
            counts, x=target_col, y="count",
            title=f"Class / Value Distribution for '{target_col}'",
            color=target_col,
            text="count"
        )
        fig_dist.update_layout(
            margin=dict(l=30, r=30, t=50, b=30),
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        charts["target_distribution"] = json.loads(fig_dist.to_json())

    # 3. Numeric Feature Histograms (Top 2-3 key features)
    hist_cols = [c for c in numeric_cols if c != target_col][:3]
    if hist_cols:
        fig_hist = go.Figure()
        palette = ["#0d9488", "#0284c7", "#f59e0b"]
        for i, col in enumerate(hist_cols):
            fig_hist.add_trace(go.Histogram(
                x=df[col].dropna(),
                name=col,
                opacity=0.75,
                marker_color=palette[i % len(palette)]
            ))
        fig_hist.update_layout(
            barmode="overlay",
            title="Distribution Histograms of Key Numerical Features",
            margin=dict(l=30, r=30, t=50, b=30),
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        charts["numerical_distributions"] = json.loads(fig_hist.to_json())

    # 4. Box Plots for Outlier Inspection
    box_cols = [c for c in numeric_cols if c != target_col][:4]
    if box_cols:
        fig_box = go.Figure()
        for col in box_cols:
            fig_box.add_trace(go.Box(
                y=df[col].dropna(),
                name=col,
                boxpoints="outliers"
            ))
        fig_box.update_layout(
            title="Box Plots: Feature Spread & Outlier Analysis",
            margin=dict(l=30, r=30, t=50, b=30),
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        charts["box_plots"] = json.loads(fig_box.to_json())

    return charts
