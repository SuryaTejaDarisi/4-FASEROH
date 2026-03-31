"""
evaluation/metrics.py

Evaluation metrics for the symbolic regression task.

We measure how well the predicted function reconstructs the input
histogram rather than whether the symbolic form is textually identical,
since two mathematically equivalent expressions would score 0 on a
token-level comparison even though they are the same function.

Metrics
-------
histogram_chi2   : Pearson chi-squared distance (lower = better)
r2_score_hist    : R^2 coefficient of determination (higher = better, max 1)
token_accuracy   : fraction of correctly predicted tokens (position-wise)
"""

import numpy as np
from data.expression_generator import _make_numpy_fn


def _predicted_counts(expr, n_bins=50, total_count=500):
    """
    Evaluate a sympy expression at bin midpoints, normalise, and
    scale to total_count expected counts.
    Returns None if evaluation fails or the expression is non-positive.
    """
    bin_width = 1.0 / n_bins
    midpoints = np.array([(i + 0.5) * bin_width for i in range(n_bins)])

    fn = _make_numpy_fn(expr)
    if fn is None:
        return None

    try:
        vals = np.array([float(fn(xv)) for xv in midpoints], dtype=np.float64)
    except Exception:
        return None

    vals = np.clip(vals, 0.0, None)
    area = vals.sum() * bin_width
    if area <= 0:
        return None
    return vals / area * total_count


def histogram_chi2(observed, predicted_expr, n_bins=50, total_count=500):
    """
    Pearson chi-squared statistic between the observed histogram and the
    one produced by evaluating predicted_expr.

    Lower values indicate better agreement.

    Parameters
    ----------
    observed       : np.ndarray, shape (n_bins,)  (integer or float counts)
    predicted_expr : sympy expression
    n_bins         : int
    total_count    : int

    Returns
    -------
    chi2 : float  (float('inf') if the expression cannot be evaluated)
    """
    pred = _predicted_counts(predicted_expr, n_bins, total_count)
    if pred is None:
        return float("inf")
    obs  = observed.astype(np.float64) + 1e-8
    pred = pred + 1e-8
    return float(np.sum((obs - pred) ** 2 / pred))


def r2_score_hist(observed, predicted_expr, n_bins=50, total_count=500):
    """
    R^2 coefficient of determination between the observed histogram and
    the histogram produced by the predicted expression.

    R^2 = 1 is a perfect fit.  R^2 < 0 means the prediction is worse
    than predicting the mean value in every bin.

    Parameters
    ----------
    observed       : np.ndarray, shape (n_bins,)
    predicted_expr : sympy expression

    Returns
    -------
    r2 : float in (-inf, 1]
    """
    pred = _predicted_counts(predicted_expr, n_bins, total_count)
    if pred is None:
        return -float("inf")
    obs    = observed.astype(np.float64)
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else -float("inf")
    return float(1.0 - ss_res / ss_tot)


def token_accuracy(pred_ids, true_ids, pad_id=0):
    """
    Position-wise accuracy between two token ID sequences.
    Positions where true_ids[i] == pad_id are excluded from the count.

    Parameters
    ----------
    pred_ids : list of int
    true_ids : list of int
    pad_id   : int  index of the padding token

    Returns
    -------
    accuracy : float in [0, 1]
    """
    min_len = min(len(pred_ids), len(true_ids))
    if min_len == 0:
        return 0.0
    correct = total = 0
    for p, t in zip(pred_ids[:min_len], true_ids[:min_len]):
        if t == pad_id:
            continue
        total += 1
        if p == t:
            correct += 1
    return correct / total if total > 0 else 0.0