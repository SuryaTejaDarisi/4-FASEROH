"""
inference/decode.py

Three inference strategies for the HistoDecoder model:

  1. Greedy decoding   -- always pick the highest-probability next token.
  2. Beam search       -- keep the B best candidate sequences at each step.
  3. BFGS refinement   -- after decoding, numerically optimise the predicted
                          constant values to minimise histogram chi-squared.

All functions operate on a single histogram (no batch dimension).
"""

import numpy as np
import torch
from scipy.optimize import minimize

from data.tokenizer import Tokenizer, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from data.expression_generator import _prefix_to_sympy, _make_numpy_fn


# -----------------------------------------------------------------------
# Helpers: token list <-> sympy expression
# -----------------------------------------------------------------------

def tokens_to_sympy(token_list):
    """
    Convert a decoded token list (strings) into a sympy expression.

    Strips BOS / EOS / PAD first, then parses the prefix sequence.
    Returns the sympy expression, or None if parsing fails.
    """
    skip = {BOS_TOKEN, EOS_TOKEN, PAD_TOKEN}
    clean = [t for t in token_list if t not in skip]
    if not clean:
        return None
    try:
        expr, _ = _prefix_to_sympy(clean)
        return expr
    except Exception:
        return None


def substitute_constants(token_list, const_values):
    """
    Replace each '<CONST>' token in token_list with the next value
    from const_values, formatted as a decimal string.

    Extra '<CONST>' tokens beyond len(const_values) are replaced with '1'
    as a safe fallback.

    Parameters
    ----------
    token_list   : list of str  (may contain '<CONST>')
    const_values : list of float

    Returns
    -------
    list of str with no '<CONST>' tokens remaining
    """
    result = []
    idx = 0
    for tok in token_list:
        if tok == "<CONST>":
            val = const_values[idx] if idx < len(const_values) else 1.0
            result.append(f"{val:.4f}")
            idx += 1
        else:
            result.append(tok)
    return result


# -----------------------------------------------------------------------
# Greedy decoding
# -----------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(model, histogram, tokenizer, device, max_len=30):
    """
    Greedily decode a symbolic expression from one histogram.

    At each step the token with the highest softmax probability is
    selected.  Stops when <EOS> is produced or max_len is reached.

    Parameters
    ----------
    model      : HistoDecoder in eval mode
    histogram  : np.ndarray or torch.Tensor, shape (n_bins,)
    tokenizer  : Tokenizer
    device     : torch.device
    max_len    : int  maximum output length including special tokens

    Returns
    -------
    token_ids  : list of int
    const_preds: list of float  (values predicted at <CONST> positions, in order)
    """
    model.eval()
    if isinstance(histogram, np.ndarray):
        histogram = torch.tensor(histogram, dtype=torch.float32)
    histogram = histogram.unsqueeze(0).to(device)          # (1, n_bins)

    memory = model.encoder(histogram)                       # (1, n_bins, d_model)
    dec_in = torch.tensor([[tokenizer.bos_id]],
                          dtype=torch.long, device=device)  # (1, 1)
    dummy_consts = torch.zeros(1, 8, device=device)

    token_ids   = []
    const_preds = []

    for _ in range(max_len):
        logits, c_preds = model.decoder(dec_in, memory, dummy_consts)
        next_id    = logits[0, -1, :].argmax().item()
        next_const = c_preds[0, -1, 0].item()

        token_ids.append(next_id)
        if next_id == tokenizer.const_id:
            const_preds.append(next_const)

        if next_id == tokenizer.eos_id:
            break

        dec_in = torch.cat([
            dec_in,
            torch.tensor([[next_id]], dtype=torch.long, device=device)
        ], dim=1)

    return token_ids, const_preds


# -----------------------------------------------------------------------
# Beam search decoding
# -----------------------------------------------------------------------

@torch.no_grad()
def beam_search_decode(model, histogram, tokenizer, device,
                       beam_width=5, max_len=30):
    """
    Beam search decoding over the symbolic expression vocabulary.

    Maintains beam_width candidate sequences in parallel.  At each step
    each candidate is extended by beam_width possible next tokens and
    only the top beam_width (by cumulative log-probability) survive.

    Finished beams (ended with <EOS>) are collected separately and the
    highest-scoring finished beam is returned.

    Parameters
    ----------
    model      : HistoDecoder in eval mode
    histogram  : np.ndarray or torch.Tensor, shape (n_bins,)
    tokenizer  : Tokenizer
    device     : torch.device
    beam_width : int
    max_len    : int

    Returns
    -------
    token_ids  : list of int   (best finished beam, or best active beam)
    const_preds: list of float
    """
    model.eval()
    if isinstance(histogram, np.ndarray):
        histogram = torch.tensor(histogram, dtype=torch.float32)
    histogram = histogram.unsqueeze(0).to(device)

    memory = model.encoder(histogram)       # (1, n_bins, d_model)
    dummy_consts = torch.zeros(1, 8, device=device)

    # Each beam: [cumulative_log_prob, token_ids, const_preds, dec_input_tensor]
    init_inp = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
    beams    = [[0.0, [], [], init_inp]]
    finished = []

    for _ in range(max_len):
        if not beams:
            break
        candidates = []

        for log_prob, ids, consts, dec_inp in beams:
            logits, c_preds = model.decoder(dec_inp, memory, dummy_consts)
            step_logits = logits[0, -1, :]
            step_const  = c_preds[0, -1, 0].item()

            log_probs        = torch.log_softmax(step_logits, dim=-1)
            topk_lp, topk_id = log_probs.topk(beam_width)

            for lp, tid in zip(topk_lp.tolist(), topk_id.tolist()):
                new_lp     = log_prob + lp
                new_ids    = ids + [tid]
                new_consts = consts + ([step_const] if tid == tokenizer.const_id else [])
                new_inp    = torch.cat([
                    dec_inp,
                    torch.tensor([[tid]], dtype=torch.long, device=device)
                ], dim=1)

                if tid == tokenizer.eos_id:
                    finished.append((new_lp, new_ids, new_consts))
                else:
                    candidates.append([new_lp, new_ids, new_consts, new_inp])

        candidates.sort(key=lambda c: c[0], reverse=True)
        beams = candidates[:beam_width]

    # Return best finished beam; fall back to best active beam
    if finished:
        finished.sort(key=lambda b: b[0], reverse=True)
        best = finished[0]
    elif beams:
        beams.sort(key=lambda b: b[0], reverse=True)
        best = beams[0]
    else:
        return [], []

    return best[1], best[2]


# -----------------------------------------------------------------------
# BFGS constant refinement
# -----------------------------------------------------------------------

def _histogram_from_sympy(expr, n_bins=50, total_count=500):
    """
    Evaluate a sympy expression at bin midpoints in [0, 1] and return
    expected histogram counts (noiseless, normalised).
    Returns None if evaluation fails.
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


def bfgs_refine_constants(token_list, initial_consts, observed_histogram,
                          n_bins=50, total_count=500, max_iter=200):
    """
    Refine the constant values in a predicted expression using L-BFGS-B
    to minimise the chi-squared distance between the observed histogram
    and the one produced by the expression.

    This is a post-decoding numerical step that improves accuracy without
    changing the structural form of the predicted expression.

    Parameters
    ----------
    token_list          : list of str  (may contain '<CONST>')
    initial_consts      : list of float  (from the regression head)
    observed_histogram  : np.ndarray, shape (n_bins,)
    n_bins              : int
    total_count         : int
    max_iter            : int

    Returns
    -------
    refined_consts : list of float
    success        : bool  (True if L-BFGS-B converged)
    """
    if not initial_consts:
        return initial_consts, True

    def objective(const_vals):
        filled    = substitute_constants(token_list, list(const_vals))
        expr      = tokens_to_sympy(filled)
        if expr is None:
            return 1e9
        pred_hist = _histogram_from_sympy(expr, n_bins, total_count)
        if pred_hist is None:
            return 1e9
        obs  = observed_histogram.astype(np.float64) + 1e-8
        pred = pred_hist + 1e-8
        return float(np.sum((obs - pred) ** 2 / pred))

    x0     = np.array(initial_consts, dtype=np.float64)
    bounds = [(-15.0, 15.0)] * len(initial_consts)

    result = minimize(objective, x0, method="L-BFGS-B",
                      bounds=bounds,
                      options={"maxiter": max_iter, "ftol": 1e-10})

    if result.success or result.fun < objective(x0):
        return list(result.x), result.success
    return initial_consts, False
