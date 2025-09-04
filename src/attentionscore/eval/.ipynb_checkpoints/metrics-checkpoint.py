from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    precision_score, recall_score, f1_score, matthews_corrcoef
)

def find_optimal_threshold(y_true, y_pred_prob) -> float:
    """
    Youden's J (argmax TPR - FPR) from the ROC curve.
    Robust to edge-cases: if a single class is present, returns 0.5.
    """
    y_true = np.ravel(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    optimal_idx = np.argmax(tpr - fpr)
    return float(thresholds[optimal_idx])

def safe_pr_auc(y_true, y_scores) -> float:
    """
    PR-AUC computed as AUC(recall, precision) after sorting by recall.
    Avoids some edge-case pitfalls.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    if recall.size < 2:
        return 0.0
    order = np.argsort(recall)
    return float(auc(recall[order], precision[order]))

def compute_thresholded_metrics(y_true, y_scores, threshold: float) -> dict:
    """Compute thresholded metrics given scores and a threshold."""
    y_pred = (np.asarray(y_scores) >= threshold).astype(int)
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1 Score":  f1_score(y_true, y_pred, zero_division=0),
        "MCC":       matthews_corrcoef(y_true, y_pred),
    }

def probas(model, X):
    """
    Get positive-class probabilities robustly from sklearn/xgboost/MLP/etc.
    Falls back to sigmoid(decision_function) or predict() if needed.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    return model.predict(X).astype(float)
