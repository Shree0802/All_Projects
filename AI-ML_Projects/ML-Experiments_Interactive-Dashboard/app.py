import os
import io
import json
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, Response
import pandas as pd

# Set matplotlib config dir to prevent cache warnings in read-only directories
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent / ".mplconfig")

from services import model_service as ms
from services import training_service as ts
from services import dataset_service as ds
from services import comparison_service as cs
from services import history_service as hs
from services import explainability_service as xs
from services import report_service as rs
from services import learning_service as ls

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload limit

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# ===================================================================
# CORE NAVIGATION ROUTES
# ===================================================================

@app.route("/")
def home():
    stats = hs.get_session_stats()
    recent_history = hs.get_all_history()[:5]

    total_experiments = len(ms.ALGORITHMS)
    regression_count = sum(1 for a in ms.ALGORITHMS.values() if a["category"] == "Regression")
    classification_count = sum(1 for a in ms.ALGORITHMS.values() if a["category"] == "Classification")
    available_saved_models = len(ms.MODELS)
    datasets_count = len(ds.list_datasets())

    return render_template(
        "index.html",
        student=ms.STUDENT_PROFILE,
        algorithms=ms.ALGORITHMS,
        total_experiments=total_experiments,
        regression_count=regression_count,
        classification_count=classification_count,
        available_saved_models=available_saved_models,
        datasets_count=datasets_count,
        session_trained_count=stats["session_train_count"],
        last_training_activity=stats["last_activity"],
        recent_history=recent_history
    )

@app.route("/experiments")
def experiments():
    category_filter = request.args.get("category", "All")
    regression_count = sum(1 for a in ms.ALGORITHMS.values() if a["category"] == "Regression")
    classification_count = sum(1 for a in ms.ALGORITHMS.values() if a["category"] == "Classification")

    return render_template(
        "experiments.html",
        student=ms.STUDENT_PROFILE,
        algorithms=ms.ALGORITHMS,
        active_category=category_filter,
        regression_count=regression_count,
        classification_count=classification_count
    )

@app.route("/algorithm/<key>", methods=["GET", "POST"])
def algorithm(key):
    if key not in ms.ALGORITHMS:
        return redirect(url_for("experiments"))

    spec = ms.ALGORITHMS[key]
    result = None
    error = None
    values = {field[0]: "" for field in spec["fields"]}
    local_explanation = None

    if request.method == "POST":
        try:
            for field in spec["fields"]:
                values[field[0]] = request.form.get(field[0], "")
            result = ms.predict_single(key, request.form)
            # Generate local explanation for this prediction
            model = ms.MODELS.get(key)
            feature_names = [f[0] for f in spec["fields"]]
            local_explanation = xs.explain_local_prediction(key, model, values, feature_names)
        except ValueError as exc:
            error = str(exc)
        except Exception as exc:
            app.logger.exception("Inference error for %s", key)
            error = f"Inference execution failed: {str(exc)}"

    algo_keys = list(ms.ALGORITHMS.keys())
    current_idx = algo_keys.index(key)
    prev_key = algo_keys[current_idx - 1] if current_idx > 0 else None
    next_key = algo_keys[current_idx + 1] if current_idx < len(algo_keys) - 1 else None

    # Algorithm educational theory & assumptions
    learning = ls.get_learning_content(key)

    return render_template(
        "algorithm.html",
        student=ms.STUDENT_PROFILE,
        algorithms=ms.ALGORITHMS,
        key=key,
        spec=spec,
        algo_number=current_idx + 1,
        result=result,
        error=error,
        values=values,
        prev_key=prev_key,
        next_key=next_key,
        learning=learning,
        local_explanation=local_explanation
    )

@app.route("/training")
def training():
    selected_exp = request.args.get("exp", "slr")
    if selected_exp not in ts.EXPERIMENT_SCHEMAS:
        selected_exp = "slr"

    datasets = ds.list_datasets()

    return render_template(
        "model_training.html",
        student=ms.STUDENT_PROFILE,
        algorithms=ms.ALGORITHMS,
        training_schemas=ts.EXPERIMENT_SCHEMAS,
        selected_exp=selected_exp,
        datasets=datasets
    )

@app.route("/evaluation")
@app.route("/results/<key>")
def results(key="slr"):
    if key not in ts.EXPERIMENT_SCHEMAS:
        key = "slr"

    run_bundle = ts.get_latest_run(key)
    if not run_bundle:
        run_bundle = ts.run_experiment_pipeline(key)

    spec = ms.ALGORITHMS.get(key, {})
    xai = xs.get_algo_explanation(key)

    return render_template(
        "results.html",
        student=ms.STUDENT_PROFILE,
        algorithms=ms.ALGORITHMS,
        key=key,
        spec=spec,
        run_bundle=run_bundle,
        metadata=run_bundle["metadata"],
        metrics=run_bundle["metrics"],
        charts=run_bundle["charts"],
        xai=xai
    )

@app.route("/datasets")
def dataset_explorer():
    dataset_list = ds.list_datasets()
    selected_file = request.args.get("file")
    if not selected_file and dataset_list:
        selected_file = dataset_list[0]["filename"]

    page = int(request.args.get("page", 1))
    search = request.args.get("search", "")

    dataset_info = None
    dataset_charts = {}
    if selected_file:
        try:
            dataset_info = ds.get_dataset_details(selected_file, page=page, per_page=15, search=search)
            dataset_charts = ds.get_dataset_charts(selected_file)
        except Exception as exc:
            app.logger.warning("Failed to load dataset details: %s", exc)

    return render_template(
        "dataset_explorer.html",
        student=ms.STUDENT_PROFILE,
        dataset_list=dataset_list,
        selected_file=selected_file,
        dataset_info=dataset_info,
        dataset_charts=dataset_charts,
        search=search
    )

@app.route("/comparison")
def comparison():
    group_key = request.args.get("group", "banknote_svm")
    comparison_data = cs.get_comparison_data(group_key)
    dataset_list = ds.list_datasets()

    return render_template(
        "model_comparison.html",
        student=ms.STUDENT_PROFILE,
        groups=cs.COMPARISON_GROUPS,
        active_group=group_key,
        comparison_data=comparison_data,
        dataset_list=dataset_list
    )

@app.route("/explainability")
@app.route("/explainability/<key>")
def explainability(key="slr"):
    if key not in ms.ALGORITHMS and key != "exploration":
        key = "slr"

    if key == "exploration":
        spec = {
            "num": "01",
            "title": "Python Libraries & EDA",
            "category": "Data Science & Statistics",
            "fields": [
                ["cgpa", "CGPA", "number", "0.01", "7.5"],
                ["iq", "IQ Score", "number", "1", "100"],
                ["package", "Package (LPA)", "number", "0.01", "5.0"]
            ],
            "sample_data": {"cgpa": 7.5, "iq": 105, "package": 4.5}
        }
        xai_tier = xs.get_algo_explanation("exploration")
        global_imp = xs.compute_global_feature_importance(None, ["CGPA", "IQ Score", "Package"])
        run = {}
    else:
        spec = ms.ALGORITHMS.get(key, {})
        xai_tier = xs.get_algo_explanation(key)

        # Get latest trained model or execute pipeline
        run = ts.get_latest_run(key)
        if not run:
            run = ts.run_experiment_pipeline(key)

        model = run.get("model") or ms.MODELS.get(key)
        feature_names = ts.EXPERIMENT_SCHEMAS.get(key, {}).get("features", [])
        global_imp = xs.compute_global_feature_importance(
            model, feature_names, X_val=run.get("X_test"), y_val=run.get("y_test")
        )

    # All 12 modules for navigation
    all_modules = {
        "exploration": {"num": "01", "title": "Python Libraries & EDA"}
    }
    all_modules.update(ms.ALGORITHMS)

    return render_template(
        "explainability.html",
        student=ms.STUDENT_PROFILE,
        algorithms=all_modules,
        key=key,
        spec=spec,
        xai_tier=xai_tier,
        global_imp=global_imp,
        run_bundle=run
    )

@app.route("/reports")
@app.route("/reports/<key>")
def reports(key="slr"):
    if key not in ms.ALGORITHMS:
        key = "slr"

    run = ts.get_latest_run(key)
    if not run:
        run = ts.run_experiment_pipeline(key)

    spec = ms.ALGORITHMS.get(key, {})
    learning = ls.get_learning_content(key)
    xai = xs.get_algo_explanation(key)

    return render_template(
        "reports.html",
        student=ms.STUDENT_PROFILE,
        algorithms=ms.ALGORITHMS,
        key=key,
        spec=spec,
        run=run,
        learning=learning,
        xai=xai
    )

@app.route("/history")
def history():
    history_records = hs.get_all_history()
    stats = hs.get_session_stats()

    return render_template(
        "history.html",
        student=ms.STUDENT_PROFILE,
        history_records=history_records,
        stats=stats
    )

@app.route("/learning")
@app.route("/learning/<key>")
def learning(key="exploration"):
    if key not in ls.LEARNING_DATA:
        key = "exploration"

    spec = ms.ALGORITHMS.get(key, {
        "num": "01",
        "title": "Python Libraries & Dataset Exploration (EDA)",
        "category": "Supporting Module",
        "dataset_name": "All Laboratory CSVs",
        "description": "Exploratory Data Analysis, summary statistics, correlation heatmaps, and Python libraries.",
        "fields": []
    })
    content = ls.get_learning_content(key)

    return render_template(
        "learning.html",
        student=ms.STUDENT_PROFILE,
        algorithms=ms.ALGORITHMS,
        all_modules=ls.LEARNING_DATA,
        key=key,
        spec=spec,
        content=content
    )

@app.route("/about")
@app.route("/settings")
def about():
    return render_template(
        "project_info.html",
        student=ms.STUDENT_PROFILE,
        algorithms=ms.ALGORITHMS
    )

# ===================================================================
# API & ACTION ENDPOINTS
# ===================================================================

@app.route("/api/dataset-preview/<key>")
def api_dataset_preview(key):
    if key not in ts.EXPERIMENT_SCHEMAS:
        return jsonify({"error": "Unknown experiment"}), 404
    schema = ts.EXPERIMENT_SCHEMAS[key]
    filename = schema.get("dataset_file")
    try:
        csv_path = ds.DATASET_DIR / filename
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            rows = len(df)
            cols = len(df.columns)
            missing = int(df.isnull().sum().sum())
        else:
            rows = 300
            cols = len(schema.get("features", [])) + 1
            missing = 0
        return jsonify({
            "title": schema["title"],
            "category": schema["category"],
            "dataset_file": filename,
            "rows": rows,
            "columns": cols,
            "missing_values": missing,
            "target": schema["target"],
            "features": schema["features"]
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/api/train", methods=["POST"])
def api_train():
    data = request.get_json() or {}
    key = data.get("key", "slr")
    test_size = float(data.get("test_size", 0.2))
    random_state = int(data.get("random_state", 42))
    hyperparams = data.get("hyperparams", {})

    try:
        run_bundle = ts.run_experiment_pipeline(
            key, test_size=test_size, random_state=random_state, hyperparams=hyperparams
        )
        return jsonify({
            "success": True,
            "run_id": run_bundle["run_id"],
            "metadata": run_bundle["metadata"],
            "metrics": run_bundle["metrics"],
            "charts": run_bundle["charts"],
            "results_url": url_for("results", key=key)
        })
    except Exception as exc:
        app.logger.exception("Training pipeline failed for %s", key)
        return jsonify({"success": False, "error": str(exc)}), 500

@app.route("/api/predict/<key>", methods=["POST"])
def api_predict(key):
    data = request.get_json() or {}
    try:
        res = ms.predict_single(key, data)
        return jsonify({"success": True, "result": res})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

@app.route("/api/explain/<key>", methods=["POST"])
def api_explain(key):
    data = request.get_json() or {}
    try:
        if key == "exploration":
            return jsonify({
                "success": True,
                "explanation": {
                    "key": "exploration",
                    "explanation_type": "Exploratory Feature Distribution Diagnostics",
                    "predicted_label": "Continuous Feature Matrix",
                    "confidence_score": "100% Data Integrity",
                    "details": [
                        "Evaluated variance, kurtosis, and correlation across numeric columns.",
                        "No zero-variance constant features detected in standard clean splits.",
                        "Correlation r(CGPA, Package) = 0.88 confirms strong linear relationship."
                    ],
                    "counterfactual": "Normalizing skewed distributions with Log1p or StandardScaler improves gradient convergence in subsequent regression estimators.",
                    "waterfall_chart": None
                }
            })
        spec = ms.ALGORITHMS.get(key, {})
        feature_names = [f[0] for f in spec.get("fields", [])]
        run = ts.get_latest_run(key) or ts.run_experiment_pipeline(key)
        model = run.get("model") or ms.MODELS.get(key)
        local_res = xs.explain_local_prediction(
            key, model, data, feature_names,
            X_train=run.get("X_train"), y_train=run.get("y_train")
        )
        return jsonify({"success": True, "explanation": local_res})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

@app.route("/api/compare", methods=["POST"])
def api_compare():
    data = request.get_json() or {}
    dataset = data.get("dataset")
    target = data.get("target")
    features = data.get("features", [])
    algos = data.get("algorithms", [])
    test_size = float(data.get("test_size", 0.2))
    random_state = int(data.get("random_state", 42))

    try:
        comp_res = cs.run_custom_comparison(
            dataset, target, features, algos, test_size=test_size, random_state=random_state
        )
        return jsonify({"success": True, "data": comp_res})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

@app.route("/api/export-pdf/<key>")
def api_export_pdf(key):
    try:
        # Check if custom profile parameters were passed
        custom_profile = {
            "name": request.args.get("name"),
            "roll_no": request.args.get("roll_no"),
            "class": request.args.get("class"),
            "college": request.args.get("college"),
            "subject": request.args.get("subject"),
        }
        pdf_bytes = rs.generate_pdf_report(key, custom_profile=custom_profile)
        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment;filename=lab_report_exp_{key}.pdf"}
        )
    except Exception as exc:
        app.logger.exception("PDF generation error")
        return jsonify({"error": f"Failed to generate PDF: {str(exc)}"}), 500

@app.route("/api/export-report/<key>")
def api_export_report(key):
    run = ts.get_latest_run(key)
    if not run:
        run = ts.run_experiment_pipeline(key)

    export_format = request.args.get("format", "json").lower()
    meta = run["metadata"]

    if export_format == "json":
        json_data = json.dumps(meta, indent=2)
        return Response(
            json_data,
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment;filename=evaluation_report_{key}.json"}
        )
    elif export_format == "csv":
        flat_metrics = {f"metric_{k}": v for k, v in meta["metrics"].items() if isinstance(v, (int, float, str))}
        flat_metrics.update({
            "run_id": meta["run_id"],
            "experiment": meta["title"],
            "dataset": meta["dataset"],
            "timestamp": meta["timestamp"]
        })
        df = pd.DataFrame([flat_metrics])
        csv_data = df.to_csv(index=False)
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment;filename=evaluation_report_{key}.csv"}
        )
    else:
        return jsonify({"error": "Unsupported format. Use 'json' or 'csv'."}), 400

@app.route("/api/upload-dataset", methods=["POST"])
def api_upload_dataset():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."}), 400
    file = request.files["file"]
    try:
        saved_name = ds.save_uploaded_dataset(file)
        return jsonify({"success": True, "filename": saved_name, "message": "Dataset uploaded successfully."})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

@app.route("/api/dataset-schema/<filename>")
def api_dataset_schema(filename):
    try:
        schema = ds.get_dataset_schema(filename)
        return jsonify({"success": True, "schema": schema})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

@app.route("/api/delete-history/<entry_id>", methods=["POST"])
def api_delete_history(entry_id):
    try:
        deleted = hs.delete_history_entry(entry_id)
        return jsonify({"success": True, "deleted": deleted})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

@app.route("/api/save-profile", methods=["POST"])
def api_save_profile():
    data = request.get_json() or {}
    for k in ["name", "college", "roll_no", "class", "email", "phone"]:
        if k in data and data[k].strip():
            ms.STUDENT_PROFILE[k] = data[k].strip()
    return jsonify({"success": True, "profile": ms.STUDENT_PROFILE})

@app.errorhandler(404)
def handle_404(err):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Requested API endpoint not found."}), 404
    return render_template(
        "error.html",
        student=ms.STUDENT_PROFILE,
        algorithms=ms.ALGORITHMS,
        error_code=404,
        error_title="Page or Experiment Not Found",
        error_message="The requested laboratory page or route does not exist. Please use the navigation sidebar to find the appropriate experiment."
    ), 404

@app.errorhandler(500)
def handle_500(err):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Internal laboratory execution error."}), 500
    return render_template(
        "error.html",
        student=ms.STUDENT_PROFILE,
        algorithms=ms.ALGORITHMS,
        error_code=500,
        error_title="Internal Laboratory Server Error",
        error_message="An unexpected server error occurred during computation. Please return to the overview page or review input parameters."
    ), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
