"""
utils/plot_utils.py

All matplotlib visualisation helpers used across the project.

Functions
---------
plot_prediction          -- histogram vs predicted (and optionally true) function
plot_training_curves     -- train/val loss from history.json
plot_evaluation_summary  -- chi2 and R2 distribution histograms across test set
_eval_expr               -- internal helper: evaluate sympy expr at bin midpoints
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt

from data.expression_generator import _make_numpy_fn


# -----------------------------------------------------------------------
# Internal helper used by both this module and demo.py
# -----------------------------------------------------------------------

def _eval_expr(expr, n_bins=50):
    """
    Evaluate a sympy expression at bin midpoints in [0, 1] and return
    the normalised density values (integrate to 1 over [0,1]).

    Returns None if evaluation fails or the result is non-positive.
    This function is importable from demo.py and notebook_demo.ipynb.
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
    return vals / area     # normalised density


# -----------------------------------------------------------------------
# Prediction plot
# -----------------------------------------------------------------------

def plot_prediction(histogram_float, predicted_expr,
                    true_expr_str=None, n_bins=50, save_path=None):
    """
    Plot the input histogram alongside the predicted (and optionally true)
    symbolic function overlaid as smooth curves.

    Parameters
    ----------
    histogram_float : np.ndarray, shape (n_bins,)
        Normalised bin fractions (sum to 1).
    predicted_expr  : sympy expression or None
    true_expr_str   : str or None
        String representation of the true generating expression.
        If given, plotted as a dashed green curve for reference.
    n_bins          : int
    save_path       : str or None
        Path to save the figure.  If None, the figure is displayed.
    """
    bin_width = 1.0 / n_bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_mids  = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(bin_mids, histogram_float / bin_width, width=bin_width * 0.93,
           alpha=0.50, color="steelblue", label="Input histogram")

    if predicted_expr is not None:
        pred_density = _eval_expr(predicted_expr, n_bins)
        if pred_density is not None:
            label = f"Predicted: {str(predicted_expr)[:60]}"
            ax.plot(bin_mids, pred_density, color="crimson",
                    linewidth=2.0, label=label)

    if true_expr_str is not None:
        from sympy import sympify
        try:
            true_expr    = sympify(true_expr_str)
            true_density = _eval_expr(true_expr, n_bins)
            if true_density is not None:
                ax.plot(bin_mids, true_density, color="green",
                        linewidth=1.5, linestyle="--",
                        label=f"True: {true_expr_str[:60]}")
        except Exception:
            pass

    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title("HistoDecoder: Histogram to Symbolic Expression")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    _save_or_show(fig, save_path)


# -----------------------------------------------------------------------
# Training curves
# -----------------------------------------------------------------------

def plot_training_curves(history_path, save_path=None):
    """
    Read history.json produced by train.py and plot train/val loss curves.

    Parameters
    ----------
    history_path : str   path to history.json
    save_path    : str or None
    """
    with open(history_path) as f:
        history = json.load(f)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"],
                 label="Train", marker="o", markersize=3, linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"],
                 label="Val", marker="o", markersize=3, linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss (CE + lambda * MSE)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    if "train_ce" in history and "val_ce" in history:
        axes[1].plot(epochs, history["train_ce"],
                     label="Train CE", marker="o", markersize=3, linewidth=1.5)
        axes[1].plot(epochs, history["val_ce"],
                     label="Val CE", marker="o", markersize=3, linewidth=1.5)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Cross-Entropy Loss")
        axes[1].set_title("Symbol Prediction Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# -----------------------------------------------------------------------
# Evaluation summary
# -----------------------------------------------------------------------

def plot_evaluation_summary(all_results, save_path=None):
    """
    Plot distributions of chi2 and R2 scores across a test set.

    Overlays beam-only vs beam+BFGS to visualise the improvement from
    constant refinement.

    Parameters
    ----------
    all_results : list of dicts, each with keys:
        chi2_beam, r2_beam, chi2_bfgs, r2_bfgs
    save_path   : str or None
    """
    chi2_beam = [r["chi2_beam"] for r in all_results if r["chi2_beam"] < 1e5]
    r2_beam   = [r["r2_beam"]   for r in all_results if r["r2_beam"]   > -1e3]
    chi2_bfgs = [r["chi2_bfgs"] for r in all_results if r["chi2_bfgs"] < 1e5]
    r2_bfgs   = [r["r2_bfgs"]   for r in all_results if r["r2_bfgs"]   > -1e3]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    if chi2_beam:
        axes[0].hist(chi2_beam, bins=30, alpha=0.6,
                     color="steelblue", label="Beam only")
    if chi2_bfgs:
        axes[0].hist(chi2_bfgs, bins=30, alpha=0.6,
                     color="crimson", label="Beam + BFGS")
    axes[0].set_xlabel("Chi-squared (lower = better)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Chi-Squared Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    if r2_beam:
        axes[1].hist(r2_beam, bins=30, alpha=0.6,
                     color="steelblue", label="Beam only")
    if r2_bfgs:
        axes[1].hist(r2_bfgs, bins=30, alpha=0.6,
                     color="crimson", label="Beam + BFGS")
    axes[1].set_xlabel("R^2 Score (higher = better)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("R^2 Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# -----------------------------------------------------------------------
# Internal helper
# -----------------------------------------------------------------------

def _save_or_show(fig, save_path):
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)