from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score

from .metrics import (
    find_optimal_threshold, safe_pr_auc,
    compute_thresholded_metrics, probas,
)

def evaluate_models_cv_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    X_hard: Optional[np.ndarray] = None,
    y_hard: Optional[np.ndarray] = None,
    models: Optional[Dict[str, object]] = None,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    use_test_optimal_threshold: bool = False,   # if True, picks tau on test (optimistic)
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Cross-validate a dict of models, then refit on all training and evaluate on test (+ optional hard test).

    Returns
    -------
    {
      "cv_mean": DataFrame,  # rows=models, cols=metrics
      "cv_std":  DataFrame,
      "test":    DataFrame,
      "hard":    DataFrame or None,
      "fold_details": Dict[str, List[dict]]  # per-model per-fold metrics
    }
    """
    # Flatten labels in case they're (N,1)
    y_train = np.ravel(y_train)
    y_test  = np.ravel(y_test)
    if y_hard is not None:
        y_hard = np.ravel(y_hard)

    if models is None:
        from .models import default_models
        models = default_models(random_state=random_state)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    all_cv_summaries = {}
    all_cv_stds      = {}
    fold_details: Dict[str, list] = {}
    test_results = {}
    hard_results = {}

    for name, model in models.items():
        if verbose:
            print(f"\nEvaluating {name}...")

        per_fold = {
            "Optimal Threshold": [],
            "Average Precision": [],
            "ROC-AUC": [],
            "PR-AUC": [],
            "Precision": [],
            "Recall": [],
            "MCC": [],
            "F1 Score": [],
        }
        fold_details[name] = []

        # ---- Cross-validation
        for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            model.fit(X_tr, y_tr)
            y_va_prob = probas(model, X_va)

            tau = find_optimal_threshold(y_va, y_va_prob)
            thr = compute_thresholded_metrics(y_va, y_va_prob, tau)

            ap     = average_precision_score(y_va, y_va_prob)
            roc    = roc_auc_score(y_va, y_va_prob)
            pr_auc = safe_pr_auc(y_va, y_va_prob)

            per_fold["Optimal Threshold"].append(tau)
            per_fold["Average Precision"].append(ap)
            per_fold["ROC-AUC"].append(roc)
            per_fold["PR-AUC"].append(pr_auc)
            per_fold["Precision"].append(thr["Precision"])
            per_fold["Recall"].append(thr["Recall"])
            per_fold["MCC"].append(thr["MCC"])
            per_fold["F1 Score"].append(thr["F1 Score"])

            fold_details[name].append({
                "fold": fold_id, "tau": tau, "AP": ap, "ROC-AUC": roc, "PR-AUC": pr_auc, **thr
            })

            if verbose:
                print(f"  Fold {fold_id}: Thr={tau:.3f} | AP={ap:.3f} ROC-AUC={roc:.3f} "
                      f"PR-AUC={pr_auc:.3f} P={thr['Precision']:.3f} R={thr['Recall']:.3f} "
                      f"F1={thr['F1 Score']:.3f} MCC={thr['MCC']:.3f}")

        # Aggregate CV metrics
        all_cv_summaries[name] = {m: float(np.mean(v)) for m, v in per_fold.items()}
        all_cv_stds[name]      = {m: float(np.std(v))  for m, v in per_fold.items()}

        # ---- External test
        model.fit(X_train, y_train)
        y_te_prob = probas(model, X_test)

        if use_test_optimal_threshold:
            tau_test = find_optimal_threshold(y_test, y_te_prob)  # NOTE: optimistic
        else:
            tau_test = 0.5

        te_ap     = average_precision_score(y_test, y_te_prob)
        te_roc    = roc_auc_score(y_test, y_te_prob)
        te_pr_auc = safe_pr_auc(y_test, y_te_prob)
        te_thr    = compute_thresholded_metrics(y_test, y_te_prob, tau_test)

        test_results[name] = {
            "Optimal Threshold": tau_test,
            "Average Precision": te_ap,
            "ROC-AUC": te_roc,
            "PR-AUC": te_pr_auc,
            "Precision": te_thr["Precision"],
            "Recall": te_thr["Recall"],
            "MCC": te_thr["MCC"],
            "F1 Score": te_thr["F1 Score"],
        }

        # ---- Hard test (optional), using SAME tau_test
        if X_hard is not None and y_hard is not None:
            y_ht_prob = probas(model, X_hard)
            ht_ap     = average_precision_score(y_hard, y_ht_prob)
            ht_roc    = roc_auc_score(y_hard, y_ht_prob)
            ht_pr_auc = safe_pr_auc(y_hard, y_ht_prob)
            ht_thr    = compute_thresholded_metrics(y_hard, y_ht_prob, tau_test)

            hard_results[name] = {
                "Optimal Threshold": tau_test,
                "Average Precision": ht_ap,
                "ROC-AUC": ht_roc,
                "PR-AUC": ht_pr_auc,
                "Precision": ht_thr["Precision"],
                "Recall": ht_thr["Recall"],
                "MCC": ht_thr["MCC"],
                "F1 Score": ht_thr["F1 Score"],
            }

    # Build DataFrames
    cv_mean_df = pd.DataFrame(all_cv_summaries).T
    cv_std_df  = pd.DataFrame(all_cv_stds).T
    test_df    = pd.DataFrame(test_results).T
    hard_df    = pd.DataFrame(hard_results).T if hard_results else None

    return {
        "cv_mean": cv_mean_df,
        "cv_std":  cv_std_df,
        "test":    test_df,
        "hard":    hard_df,
        "fold_details": fold_details,
    }
