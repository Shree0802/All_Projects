import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
)

def evaluate_regression(y_true, y_pred):
    """
    Computes standard regression metrics strictly on held-out test data.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    exp_var = float(explained_variance_score(y_true, y_pred))

    residuals = y_true - y_pred

    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "explained_variance": round(exp_var, 4),
        "residuals_mean": round(float(np.mean(residuals)), 4),
        "residuals_std": round(float(np.std(residuals)), 4),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "residuals": residuals.tolist()
    }

def evaluate_classification(y_true, y_pred, y_prob=None, classes=None):
    """
    Computes classification metrics, confusion matrix, and curve data on test data.
    """
    if classes is None:
        classes = sorted(list(set(y_true) | set(y_pred)))

    classes_str = [str(c) for c in classes]
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))

    is_binary = len(classes) == 2
    pos_label = classes[1] if is_binary else None

    # Classification report dict
    report = classification_report(
        y_true_arr, y_pred_arr, labels=classes, output_dict=True, zero_division=0
    )

    # Precision, Recall, F1
    if is_binary:
        prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=pos_label, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred_arr, pos_label=pos_label, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=pos_label, zero_division=0))
    else:
        prec = float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    macro_prec = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=classes).tolist()

    roc_data = None
    pr_data = None

    # Compute ROC and PR curves if valid probabilities are provided
    if y_prob is not None and is_binary:
        try:
            # y_prob should be 1D array of positive class probabilities
            if hasattr(y_prob, "ndim") and y_prob.ndim == 2:
                prob_scores = y_prob[:, 1]
            else:
                prob_scores = y_prob

            # Convert y_true to binary 0/1
            y_binary = (y_true_arr == pos_label).astype(int)
            fpr, tpr, thresholds = roc_curve(y_binary, prob_scores)
            roc_auc_val = float(auc(fpr, tpr))

            precision_curve, recall_curve, _ = precision_recall_curve(y_binary, prob_scores)
            ap_val = float(average_precision_score(y_binary, prob_scores))

            roc_data = {
                "fpr": [round(float(v), 4) for v in fpr],
                "tpr": [round(float(v), 4) for v in tpr],
                "auc": round(roc_auc_val, 4)
            }
            pr_data = {
                "precision": [round(float(v), 4) for v in precision_curve],
                "recall": [round(float(v), 4) for v in recall_curve],
                "ap": round(ap_val, 4)
            }
        except Exception:
            roc_data = None
            pr_data = None

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "macro_precision": round(macro_prec * 100, 2),
        "macro_recall": round(macro_rec * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "classes": classes_str,
        "confusion_matrix": cm,
        "classification_report": report,
        "roc": roc_data,
        "pr": pr_data,
        "is_binary": is_binary
    }
