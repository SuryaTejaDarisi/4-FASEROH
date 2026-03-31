"""
histogram_generator.py

Converts a normalised symbolic expression into a histogram by:
  1. Evaluating the function at the midpoint of each bin
  2. Scaling to the desired total count N
  3. Adding Poisson noise to simulate real measurement uncertainty

The output is a numpy array of integer counts (one per bin) that
mimics how experimental data looks in a particle physics detector
or any counting experiment.
"""

import numpy as np
from sympy import symbols, N as sympyN


X = symbols("x")


def expression_to_histogram(expr, n_bins=50, total_count=500, rng=None):
    """
    Turn a sympy probability density function into a noisy histogram.

    Steps:
      - Evaluate expr at bin midpoints to get mean counts per bin
      - Draw each bin count from Poisson(mean_count)

    Parameters
    ----------
    expr : sympy expression
        A non-negative function normalised to integrate to 1 on [0, 1].
    n_bins : int
        Number of histogram bins (resolution).
    total_count : int
        Total expected number of counts (controls signal-to-noise).
        Higher = cleaner histogram.
    rng : numpy.random.Generator or None
        Optional seeded generator for reproducibility.

    Returns
    -------
    histogram : np.ndarray, shape (n_bins,)
        Integer counts per bin, with Poisson noise applied.
    bin_edges : np.ndarray, shape (n_bins + 1,)
        Left edges of each bin plus the right edge of the last bin.
    mean_counts : np.ndarray, shape (n_bins,)
        Expected (noiseless) counts per bin, for reference.
    """
    if rng is None:
        rng = np.random.default_rng()

    bin_width = 1.0 / n_bins
    # Bin midpoints in (0, 1)
    midpoints = np.array([(i + 0.5) * bin_width for i in range(n_bins)])
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Convert sympy expression to a fast numpy-callable function.
    # Suppress RuntimeWarnings: the expression has already passed
    # positivity validation; any remaining warnings here are harmless
    # floating-point edge cases at bin midpoints.
    from data.expression_generator import _make_numpy_fn
    import warnings
    try:
        f_numeric = _make_numpy_fn(expr)
        if f_numeric is None:
            raise ValueError("lambdify failed")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                vals_raw = [f_numeric(xv) for xv in midpoints]
        # Handle complex values by taking the real part if imaginary is tiny
        cleaned = []
        for v in vals_raw:
            cv = complex(v)
            cleaned.append(cv.real if abs(cv.imag) < 1e-8 else 0.0)
        raw_vals = np.array(cleaned, dtype=np.float64)
    except Exception:
        # Fallback: evaluate symbolically (slower but safer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_vals = np.array([float(sympyN(expr.subs(X, float(xv))))
                                 for xv in midpoints], dtype=np.float64)

    # Guard: clip negatives that arise from floating-point noise
    raw_vals = np.clip(raw_vals, 0.0, None)

    # Scale so values integrate to total_count
    # (expr is already normalised to 1, so integral * total_count = total_count)
    mean_counts = raw_vals * bin_width * total_count

    # Draw integer counts from Poisson distribution
    histogram = rng.poisson(mean_counts).astype(np.int64)

    return histogram, bin_edges, mean_counts


def histogram_to_float(histogram):
    """
    Normalise a raw count histogram to a unit-sum float array.

    The transformer encoder receives floating-point fractions, not raw
    counts, so the network learns shapes rather than absolute scales.

    Parameters
    ----------
    histogram : np.ndarray of ints or floats

    Returns
    -------
    np.ndarray of floats summing to 1 (or zeros if the histogram is empty)
    """
    total = histogram.sum()
    if total == 0:
        return np.zeros_like(histogram, dtype=np.float32)
    return (histogram / total).astype(np.float32)