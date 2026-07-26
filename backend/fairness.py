"""
ExplainX AI — Fairness & Bias Detection
Computes standard group-fairness metrics locally: demographic parity,
equal opportunity, disparate impact, and predictive parity.
"""
import numpy as np
import pandas as pd


def _clean(v):
    """Convert NaN/inf to None so the value is JSON-serializable."""
    if v is None:
        return None
    try:
        if isinstance(v, (int, float)) and (np.isnan(v) or np.isinf(v)):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _confusion_counts(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return tp, fp, fn, tn


def evaluate_fairness(y_true: np.ndarray, y_pred: np.ndarray, protected_attr: np.ndarray,
                       privileged_value, unprivileged_value):
    """
    Returns a dict of fairness metrics comparing privileged vs. unprivileged groups.
    Thresholds follow common industry rules of thumb (e.g. the 80% / four-fifths rule).
    Any NaN/inf values (e.g. from an empty group) are converted to None.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    protected_attr = np.asarray(protected_attr)

    priv_mask = protected_attr == privileged_value
    unpriv_mask = protected_attr == unprivileged_value

    results = {}

    priv_pos_rate = y_pred[priv_mask].mean() if priv_mask.sum() > 0 else np.nan
    unpriv_pos_rate = y_pred[unpriv_mask].mean() if unpriv_mask.sum() > 0 else np.nan
    dpd = unpriv_pos_rate - priv_pos_rate
    dpd_status = "PASS" if not np.isnan(dpd) and abs(dpd) < 0.1 else ("WARN" if not np.isnan(dpd) and abs(dpd) < 0.2 else "FAIL")
    results["demographic_parity_difference"] = {
        "value": _clean(float(dpd) if not np.isnan(dpd) else np.nan),
        "privileged_rate": _clean(float(priv_pos_rate) if not np.isnan(priv_pos_rate) else np.nan),
        "unprivileged_rate": _clean(float(unpriv_pos_rate) if not np.isnan(unpriv_pos_rate) else np.nan),
        "status": dpd_status if not np.isnan(dpd) else "WARN",
    }

    disparate_impact = unpriv_pos_rate / priv_pos_rate if priv_pos_rate and priv_pos_rate > 0 else np.nan
    results["disparate_impact_ratio"] = {
        "value": _clean(float(disparate_impact) if not np.isnan(disparate_impact) else np.nan),
        "status": "PASS" if (not np.isnan(disparate_impact) and disparate_impact >= 0.8) else "FAIL",
    }

    tp_p, fp_p, fn_p, tn_p = _confusion_counts(y_true[priv_mask], y_pred[priv_mask])
    tp_u, fp_u, fn_u, tn_u = _confusion_counts(y_true[unpriv_mask], y_pred[unpriv_mask])
    tpr_priv = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else np.nan
    tpr_unpriv = tp_u / (tp_u + fn_u) if (tp_u + fn_u) > 0 else np.nan
    eod = tpr_unpriv - tpr_priv
    eod_status = "PASS" if not np.isnan(eod) and abs(eod) < 0.1 else ("WARN" if not np.isnan(eod) and abs(eod) < 0.2 else "FAIL")
    results["equal_opportunity_difference"] = {
        "value": _clean(float(eod) if not np.isnan(eod) else np.nan),
        "tpr_privileged": _clean(float(tpr_priv) if not np.isnan(tpr_priv) else np.nan),
        "tpr_unprivileged": _clean(float(tpr_unpriv) if not np.isnan(tpr_unpriv) else np.nan),
        "status": eod_status if not np.isnan(eod) else "WARN",
    }

    prec_priv = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else np.nan
    prec_unpriv = tp_u / (tp_u + fp_u) if (tp_u + fp_u) > 0 else np.nan
    ppd = prec_unpriv - prec_priv if not (np.isnan(prec_priv) or np.isnan(prec_unpriv)) else np.nan
    ppd_status = "PASS" if not np.isnan(ppd) and abs(ppd) < 0.1 else ("WARN" if not np.isnan(ppd) and abs(ppd) < 0.2 else "FAIL")
    results["predictive_parity_difference"] = {
        "value": _clean(float(ppd) if not np.isnan(ppd) else np.nan),
        "precision_privileged": _clean(float(prec_priv) if not np.isnan(prec_priv) else np.nan),
        "precision_unprivileged": _clean(float(prec_unpriv) if not np.isnan(prec_unpriv) else np.nan),
        "status": ppd_status if not np.isnan(ppd) else "WARN",
    }

    acc_priv = np.mean(y_true[priv_mask] == y_pred[priv_mask]) if priv_mask.sum() > 0 else np.nan
    acc_unpriv = np.mean(y_true[unpriv_mask] == y_pred[unpriv_mask]) if unpriv_mask.sum() > 0 else np.nan
    results["accuracy_by_group"] = {
        "privileged": _clean(float(acc_priv) if not np.isnan(acc_priv) else np.nan),
        "unprivileged": _clean(float(acc_unpriv) if not np.isnan(acc_unpriv) else np.nan),
    }

    results["group_sizes"] = {
        "privileged_n": int(priv_mask.sum()),
        "unprivileged_n": int(unpriv_mask.sum()),
    }

    return results


def overall_fairness_verdict(results: dict) -> str:
    statuses = [v["status"] for k, v in results.items() if isinstance(v, dict) and "status" in v]
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"
