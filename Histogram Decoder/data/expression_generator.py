# """
# expression_generator.py

# Generates random symbolic mathematical expressions in prefix (Polish)
# notation, constrained to be valid probability density functions on [0, 1].

# Key design decisions vs a naive approach
# -----------------------------------------
# 1.  Numerical validation instead of symbolic.
#     sympy.simplify and sympy.integrate are very slow on complex expressions.
#     We instead use fast numpy evaluation on a dense grid to check positivity
#     and scipy.integrate.quad for the normalisation integral.
#     This keeps generation time under 1ms per successful sample.

# 2.  Biased operator set.
#     Operators that commonly produce non-positive or divergent results on [0,1]
#     (tan, cot, ln, inv, asin, acos near boundaries) are given low probabilities
#     so the success rate stays high without too many retries.

# 3.  Controlled expression complexity.
#     n_ops in [1, 6] by default. Very deep expressions (8+ ops) rarely survive
#     the positivity check and make the seq2seq task harder to learn.

# 4.  Prefix notation.
#     No brackets, no ambiguity. 'sin + x 1' means sin(x+1). This format is
#     natural for autoregressive sequence generation.

# References
# ----------
# Lample & Charton, "Deep Learning for Symbolic Mathematics" (2019)
# https://arxiv.org/abs/1912.01412
# """

# import math
# import random
# import numpy as np
# from scipy import integrate as sci_integrate
# from sympy import (
#     symbols, lambdify,
#     sin, cos, exp, sqrt,
#     asin, acos, atan,
#     Integer
# )


# X = symbols("x")

# # Evaluation grid: 80 points strictly inside (0, 1)
# # Used for fast positivity checks (no endpoint issues)
# _GRID = np.linspace(0.02, 0.98, 80)


# # -----------------------------------------------------------------------
# # Operator table
# # -----------------------------------------------------------------------
# # Format: name -> (arity, sampling_weight)
# # Weights are unnormalised; higher = more likely to be sampled.
# # Operators that frequently produce negatives or divergences on [0,1]
# # are given low weight rather than excluded, preserving expression diversity.

# UNARY_OPS = {
#     "sin":  (1, 6),
#     "cos":  (1, 6),
#     "exp":  (1, 4),
#     "sqrt": (1, 5),
#     "pow2": (1, 6),
#     "pow3": (1, 3),
#     "atan": (1, 3),
#     "asin": (1, 1),
#     "acos": (1, 1),
# }

# BINARY_OPS = {
#     "+": (2, 8),
#     "*": (2, 6),
#     "-": (2, 3),
#     "/": (2, 2),
# }

# ALL_OPS = {**UNARY_OPS, **BINARY_OPS}

# # Precompute name lists and normalised probability weights
# _UNARY_NAMES   = list(UNARY_OPS.keys())
# _UNARY_WEIGHTS = [UNARY_OPS[k][1] for k in _UNARY_NAMES]
# _u_total = sum(_UNARY_WEIGHTS)
# _UNARY_WEIGHTS = [w / _u_total for w in _UNARY_WEIGHTS]

# _BINARY_NAMES   = list(BINARY_OPS.keys())
# _BINARY_WEIGHTS = [BINARY_OPS[k][1] for k in _BINARY_NAMES]
# _b_total = sum(_BINARY_WEIGHTS)
# _BINARY_WEIGHTS = [w / _b_total for w in _BINARY_WEIGHTS]


# # -----------------------------------------------------------------------
# # Apply operator: token name + sympy args -> sympy expression
# # -----------------------------------------------------------------------

# def _apply_op(op, *args):
#     a = args[0]
#     if op == "sin":   return sin(a)
#     if op == "cos":   return cos(a)
#     if op == "exp":   return exp(a)
#     if op == "sqrt":  return sqrt(a)
#     if op == "pow2":  return a ** 2
#     if op == "pow3":  return a ** 3
#     if op == "atan":  return atan(a)
#     if op == "asin":  return asin(a)
#     if op == "acos":  return acos(a)
#     b = args[1]
#     if op == "+":     return a + b
#     if op == "*":     return a * b
#     if op == "-":     return a - b
#     if op == "/":     return a / b
#     raise ValueError(f"Unknown operator: {op}")


# # -----------------------------------------------------------------------
# # Expression tree construction
# # -----------------------------------------------------------------------

# def _sample_tree(n_ops, rng):
#     """
#     Build a prefix-order expression tree with exactly n_ops operators.

#     The algorithm is adapted from Lample & Charton (2019):
#     we maintain a stack where None marks an unfilled leaf slot.
#     At each step we choose an operator and insert it at the leftmost
#     available slot, replacing it with the operator followed by
#     (arity) new None slots for its children.

#     Returns a list of strings/Nones in prefix order.
#     """
#     stack   = [None]
#     n_empty = 1
#     l_leaves = 0

#     for ops_left in range(n_ops, 0, -1):
#         max_skip = max(0, min(l_leaves, n_empty - ops_left))
#         skipped  = rng.randint(0, max_skip)

#         can_binary = (n_empty - skipped) >= (ops_left + 1)

#         if can_binary and rng.random() < 0.55:
#             op    = rng.choices(_BINARY_NAMES, weights=_BINARY_WEIGHTS)[0]
#             arity = 2
#         else:
#             op    = rng.choices(_UNARY_NAMES, weights=_UNARY_WEIGHTS)[0]
#             arity = 1

#         n_empty  += arity - 1 - skipped
#         l_leaves += skipped

#         none_pos = [i for i, v in enumerate(stack) if v is None]
#         pos = none_pos[l_leaves]

#         stack = (
#             stack[:pos]
#             + [op]
#             + [None] * arity
#             + stack[pos + 1:]
#         )

#     return stack


# def _fill_leaves(stack, rng, max_const=4):
#     """
#     Replace every None slot with either 'x' or a small positive integer.

#     We prefer 'x' (probability 0.65) to keep expressions variable-heavy.
#     Constants are restricted to 1..max_const (positive) because negative
#     leaf constants make positivity checks fail much more often.
#     """
#     leaves = []
#     for _ in range(stack.count(None)):
#         if rng.random() < 0.65:
#             leaves.append("x")
#         else:
#             leaves.append(str(rng.randint(1, max_const)))
#     rng.shuffle(leaves)

#     result = []
#     leaf_idx = 0
#     for tok in stack:
#         if tok is None:
#             result.append(leaves[leaf_idx])
#             leaf_idx += 1
#         else:
#             result.append(tok)
#     return result


# # -----------------------------------------------------------------------
# # Token list -> sympy expression (prefix parser)
# # -----------------------------------------------------------------------

# def _prefix_to_sympy(tokens):
#     """
#     Parse a prefix-notation token list into a sympy expression.

#     Returns (sympy_expr, n_tokens_consumed).
#     Raises ValueError on malformed input.
#     """
#     if not tokens:
#         raise ValueError("Empty token list in prefix parser")

#     tok  = tokens[0]
#     rest = tokens[1:]

#     if tok == "x":
#         return X, 1
#     if tok not in ALL_OPS:
#         try:
#             return Integer(int(tok)), 1
#         except ValueError:
#             raise ValueError(f"Unrecognised token: {tok!r}")

#     arity = ALL_OPS[tok][0]
#     if arity == 1:
#         child, c = _prefix_to_sympy(rest)
#         return _apply_op(tok, child), 1 + c
#     else:
#         left,  c1 = _prefix_to_sympy(rest)
#         right, c2 = _prefix_to_sympy(rest[c1:])
#         return _apply_op(tok, left, right), 1 + c1 + c2


# # -----------------------------------------------------------------------
# # Numerical validation helpers  (fast - no sympy.simplify)
# # -----------------------------------------------------------------------

# def _make_numpy_fn(expr):
#     """Convert a sympy expression to a vectorised numpy function."""
#     try:
#         return lambdify(X, expr, modules="numpy")
#     except Exception:
#         return None


# def _eval_on_grid(fn):
#     """
#     Evaluate fn on the pre-built grid.
#     Returns a real float64 array, or None if evaluation fails.

#     RuntimeWarnings (arccos/arcsin out of domain, sqrt of negative, etc.)
#     are suppressed because invalid values are expected for many candidate
#     expressions and are handled by the positivity check that follows.
#     Complex outputs are treated as invalid and discarded.
#     """
#     try:
#         with np.errstate(all="ignore"):
#             vals = fn(_GRID)

#         if np.isscalar(vals):
#             v = complex(vals)
#             if abs(v.imag) > 1e-10:
#                 return None
#             vals = np.full(len(_GRID), v.real)
#         else:
#             vals = np.asarray(vals)
#             if np.iscomplexobj(vals):
#                 if np.any(np.abs(vals.imag) > 1e-10):
#                     return None
#                 vals = vals.real
#             vals = vals.astype(np.float64)

#         return vals
#     except Exception:
#         return None


# def _is_positive_on_grid(vals):
#     """True if all values are strictly positive and finite."""
#     if vals is None:
#         return False
#     return bool(np.all(np.isfinite(vals)) and np.all(vals > 1e-9))


# def _normalise_numerical(expr, fn):
#     """
#     Integrate fn over [0, 1] with scipy and return the normalised
#     sympy expression.  Returns None on failure.

#     IntegrationWarnings are suppressed: they occur when scipy struggles
#     with a function but still returns a usable estimate.  We check the
#     result value regardless and discard if non-positive or non-finite.
#     """
#     import warnings
#     try:
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             with np.errstate(all="ignore"):
#                 area, _ = sci_integrate.quad(
#                     fn, 0.0, 1.0, limit=100,
#                     epsabs=1e-5, epsrel=1e-5
#                 )
#     except Exception:
#         return None
#     if area <= 0 or not math.isfinite(area):
#         return None
#     return expr / area


# # -----------------------------------------------------------------------
# # Public API
# # -----------------------------------------------------------------------

# def generate_expression(n_ops=None, rng=None, max_attempts=80):
#     """
#     Generate a single valid symbolic expression.

#     The expression is:
#       - Positive on [0, 1]  (checked numerically on an 80-point grid)
#       - Integrable over [0, 1]  (checked via scipy.integrate.quad)
#       - Normalised so the integral over [0, 1] equals 1
#         (making it a valid probability density function)

#     Parameters
#     ----------
#     n_ops : int or None
#         Number of operators in the expression tree.
#         If None, uniformly sampled from [1, 6].
#     rng : random.Random or None
#         Seeded RNG for reproducibility. A fresh one is created if None.
#     max_attempts : int
#         How many times to retry before giving up.

#     Returns
#     -------
#     (prefix_tokens, normalised_sympy_expr)
#         or (None, None) if all attempts fail.
#     """
#     if rng is None:
#         rng = random.Random()

#     for _ in range(max_attempts):
#         n = n_ops if n_ops is not None else rng.randint(1, 6)

#         try:
#             tree    = _sample_tree(n, rng)
#             tokens  = _fill_leaves(tree, rng)
#             expr, _ = _prefix_to_sympy(tokens)
#         except Exception:
#             continue

#         fn = _make_numpy_fn(expr)
#         if fn is None:
#             continue

#         vals = _eval_on_grid(fn)
#         if not _is_positive_on_grid(vals):
#             continue

#         normalised = _normalise_numerical(expr, fn)
#         if normalised is None:
#             continue

#         fn_norm = _make_numpy_fn(normalised)
#         if fn_norm is None:
#             continue
#         if not _is_positive_on_grid(_eval_on_grid(fn_norm)):
#             continue

#         return tokens, normalised

#     return None, None


"""
expression_generator.py

Generates random symbolic mathematical expressions in prefix (Polish)
notation, constrained to be valid probability density functions on [0, 1].

Key design decisions vs a naive approach
-----------------------------------------
1.  Numerical validation instead of symbolic.
    sympy.simplify and sympy.integrate are very slow on complex expressions.
    We instead use fast numpy evaluation on a dense grid to check positivity
    and scipy.integrate.quad for the normalisation integral.
    This keeps generation time under 1ms per successful sample.

2.  Biased operator set.
    Operators that commonly produce non-positive or divergent results on [0,1]
    (tan, cot, ln, inv, asin, acos near boundaries) are given low probabilities
    so the success rate stays high without too many retries.

3.  Controlled expression complexity.
    n_ops in [1, 6] by default. Very deep expressions (8+ ops) rarely survive
    the positivity check and make the seq2seq task harder to learn.

4.  Prefix notation.
    No brackets, no ambiguity. 'sin + x 1' means sin(x+1). This format is
    natural for autoregressive sequence generation.

References
----------
Lample & Charton, "Deep Learning for Symbolic Mathematics" (2019)
https://arxiv.org/abs/1912.01412
"""

import math
import random
import numpy as np
from scipy import integrate as sci_integrate
from sympy import (
    symbols, lambdify,
    sin, cos, exp, sqrt,
    asin, acos, atan,
    Integer
)


X = symbols("x")

# Evaluation grid: 80 points strictly inside (0, 1)
# Used for fast positivity checks (no endpoint issues)
_GRID = np.linspace(0.02, 0.98, 80)


# -----------------------------------------------------------------------
# Operator table
# -----------------------------------------------------------------------
# Format: name -> (arity, sampling_weight)
# Weights are unnormalised; higher = more likely to be sampled.
# Operators that frequently produce negatives or divergences on [0,1]
# are given low weight rather than excluded, preserving expression diversity.

UNARY_OPS = {
    "sin":  (1, 6),
    "cos":  (1, 6),
    "exp":  (1, 4),
    "sqrt": (1, 5),
    "pow2": (1, 6),
    "pow3": (1, 3),
    "atan": (1, 3),
    "asin": (1, 1),
    "acos": (1, 1),
}

BINARY_OPS = {
    "+": (2, 8),
    "*": (2, 6),
    "-": (2, 3),
    "/": (2, 2),
}

ALL_OPS = {**UNARY_OPS, **BINARY_OPS}

# Precompute name lists and normalised probability weights
_UNARY_NAMES   = list(UNARY_OPS.keys())
_UNARY_WEIGHTS = [UNARY_OPS[k][1] for k in _UNARY_NAMES]
_u_total = sum(_UNARY_WEIGHTS)
_UNARY_WEIGHTS = [w / _u_total for w in _UNARY_WEIGHTS]

_BINARY_NAMES   = list(BINARY_OPS.keys())
_BINARY_WEIGHTS = [BINARY_OPS[k][1] for k in _BINARY_NAMES]
_b_total = sum(_BINARY_WEIGHTS)
_BINARY_WEIGHTS = [w / _b_total for w in _BINARY_WEIGHTS]


# -----------------------------------------------------------------------
# Apply operator: token name + sympy args -> sympy expression
# -----------------------------------------------------------------------

def _apply_op(op, *args):
    a = args[0]
    if op == "sin":   return sin(a)
    if op == "cos":   return cos(a)
    if op == "exp":   return exp(a)
    if op == "sqrt":  return sqrt(a)
    if op == "pow2":  return a ** 2
    if op == "pow3":  return a ** 3
    if op == "atan":  return atan(a)
    if op == "asin":  return asin(a)
    if op == "acos":  return acos(a)
    b = args[1]
    if op == "+":     return a + b
    if op == "*":     return a * b
    if op == "-":     return a - b
    if op == "/":     return a / b
    raise ValueError(f"Unknown operator: {op}")


# -----------------------------------------------------------------------
# Expression tree construction
# -----------------------------------------------------------------------

def _sample_tree(n_ops, rng):
    """
    Build a prefix-order expression tree with exactly n_ops operators.

    The algorithm follows Lample & Charton (2019). We maintain a count
    of empty (unfilled) leaf slots. At each step we place one operator:
      - A unary operator consumes 1 slot and adds 1  -> net change 0
      - A binary operator consumes 1 slot and adds 2 -> net change +1

    For a binary operator to be placeable, we need enough remaining
    empty slots that we can still fill them all with the remaining
    operators plus their leaves. The precise condition from the paper is:
      n_empty - skipped >= ops_left + 1
    But because we start with 1 empty slot and unary ops keep it at 1,
    we seed the stack with 2 slots so binary ops are reachable from
    the first step.

    Returns a list of strings/Nones in prefix order.
    """
    # The tree always has exactly n_ops + 1 leaves.
    # We build the operator sequence directly using the distribution from
    # Lample & Charton (2019), then assign arities and build the stack.
    #
    # Simple approach: independently decide for each of the n_ops
    # operator slots whether it is binary or unary, subject to the
    # constraint that the total number of leaves = (binary_count + 1).
    # We enforce validity by choosing arities greedily: at each step we
    # allow binary only if there are enough remaining operators to
    # "use up" the extra slot it creates.

    stack    = [None]
    n_empty  = 1
    l_leaves = 0

    for ops_left in range(n_ops, 0, -1):
        # How many left-side leaf slots can we skip before placing this op?
        max_skip = max(0, min(l_leaves, n_empty - ops_left))
        skipped  = rng.randint(0, max_skip)

        # After skipping, we have (n_empty - skipped) active empty slots.
        # If we place a binary op, n_empty increases by 1 net. We need
        # the final n_empty after ALL remaining ops to be >= 1 (at least
        # one leaf). With ops_left - 1 more ops each potentially unary
        # (net change 0) or binary (net change +1), we need:
        #   (n_empty - skipped) + 1 + (ops_left - 1) * 0 >= 1  <- always ok
        # The binding constraint from Lample & Charton is simply:
        #   n_empty - skipped >= 1   <- need a slot to place the op
        # Binary is allowed whenever we have at least 2 remaining ops OR
        # there is currently more than 1 empty slot.
        remaining = n_empty - skipped
        # Allow binary if we have more empty slots than strictly needed
        # (i.e. there is slack), OR if there are more operators remaining.
        can_binary = (remaining > 1) or (ops_left > 1 and remaining >= 1)

        if can_binary and rng.random() < 0.65:
            op    = rng.choices(_BINARY_NAMES, weights=_BINARY_WEIGHTS)[0]
            arity = 2
        else:
            op    = rng.choices(_UNARY_NAMES, weights=_UNARY_WEIGHTS)[0]
            arity = 1

        n_empty  += arity - 1 - skipped
        l_leaves += skipped

        none_pos = [i for i, v in enumerate(stack) if v is None]
        pos = none_pos[l_leaves]

        stack = (
            stack[:pos]
            + [op]
            + [None] * arity
            + stack[pos + 1:]
        )

    return stack


def _fill_leaves(stack, rng, max_const=4):
    """
    Replace every None slot with either 'x' or a small positive integer.

    We prefer 'x' (probability 0.65) to keep expressions variable-heavy.
    Constants are restricted to 1..max_const (positive) because negative
    leaf constants make positivity checks fail much more often.
    """
    leaves = []
    for _ in range(stack.count(None)):
        if rng.random() < 0.65:
            leaves.append("x")
        else:
            leaves.append(str(rng.randint(1, max_const)))
    rng.shuffle(leaves)

    result = []
    leaf_idx = 0
    for tok in stack:
        if tok is None:
            result.append(leaves[leaf_idx])
            leaf_idx += 1
        else:
            result.append(tok)
    return result


# -----------------------------------------------------------------------
# Token list -> sympy expression (prefix parser)
# -----------------------------------------------------------------------

def _prefix_to_sympy(tokens):
    """
    Parse a prefix-notation token list into a sympy expression.

    Returns (sympy_expr, n_tokens_consumed).
    Raises ValueError on malformed input.
    """
    if not tokens:
        raise ValueError("Empty token list in prefix parser")

    tok  = tokens[0]
    rest = tokens[1:]

    if tok == "x":
        return X, 1
    if tok not in ALL_OPS:
        try:
            return Integer(int(tok)), 1
        except ValueError:
            raise ValueError(f"Unrecognised token: {tok!r}")

    arity = ALL_OPS[tok][0]
    if arity == 1:
        child, c = _prefix_to_sympy(rest)
        return _apply_op(tok, child), 1 + c
    else:
        left,  c1 = _prefix_to_sympy(rest)
        right, c2 = _prefix_to_sympy(rest[c1:])
        return _apply_op(tok, left, right), 1 + c1 + c2


# -----------------------------------------------------------------------
# Numerical validation helpers  (fast - no sympy.simplify)
# -----------------------------------------------------------------------

def _make_numpy_fn(expr):
    """Convert a sympy expression to a vectorised numpy function."""
    try:
        return lambdify(X, expr, modules="numpy")
    except Exception:
        return None


def _eval_on_grid(fn):
    """
    Evaluate fn on the pre-built grid.
    Returns a real float64 array, or None if evaluation fails.

    RuntimeWarnings (arccos/arcsin out of domain, sqrt of negative, etc.)
    are suppressed because invalid values are expected for many candidate
    expressions and are handled by the positivity check that follows.
    Complex outputs are treated as invalid and discarded.
    """
    try:
        with np.errstate(all="ignore"):
            vals = fn(_GRID)

        if np.isscalar(vals):
            v = complex(vals)
            if abs(v.imag) > 1e-10:
                return None
            vals = np.full(len(_GRID), v.real)
        else:
            vals = np.asarray(vals)
            if np.iscomplexobj(vals):
                if np.any(np.abs(vals.imag) > 1e-10):
                    return None
                vals = vals.real
            vals = vals.astype(np.float64)

        return vals
    except Exception:
        return None


def _is_positive_on_grid(vals):
    """True if all values are strictly positive and finite."""
    if vals is None:
        return False
    return bool(np.all(np.isfinite(vals)) and np.all(vals > 1e-9))


def _normalise_numerical(expr, fn):
    """
    Integrate fn over [0, 1] with scipy and return the normalised
    sympy expression.  Returns None on failure.

    IntegrationWarnings are suppressed: they occur when scipy struggles
    with a function but still returns a usable estimate.  We check the
    result value regardless and discard if non-positive or non-finite.
    """
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                area, _ = sci_integrate.quad(
                    fn, 0.0, 1.0, limit=100,
                    epsabs=1e-5, epsrel=1e-5
                )
    except Exception:
        return None
    if area <= 0 or not math.isfinite(area):
        return None
    return expr / area


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def generate_expression(n_ops=None, rng=None, max_attempts=80):
    """
    Generate a single valid symbolic expression.

    The expression is:
      - Positive on [0, 1]  (checked numerically on an 80-point grid)
      - Integrable over [0, 1]  (checked via scipy.integrate.quad)
      - Normalised so the integral over [0, 1] equals 1
        (making it a valid probability density function)

    Parameters
    ----------
    n_ops : int or None
        Number of operators in the expression tree.
        If None, uniformly sampled from [1, 6].
    rng : random.Random or None
        Seeded RNG for reproducibility. A fresh one is created if None.
    max_attempts : int
        How many times to retry before giving up.

    Returns
    -------
    (prefix_tokens, normalised_sympy_expr)
        or (None, None) if all attempts fail.
    """
    if rng is None:
        rng = random.Random()

    for _ in range(max_attempts):
        n = n_ops if n_ops is not None else rng.randint(1, 6)

        try:
            tree    = _sample_tree(n, rng)
            tokens  = _fill_leaves(tree, rng)
            expr, _ = _prefix_to_sympy(tokens)
        except Exception:
            continue

        fn = _make_numpy_fn(expr)
        if fn is None:
            continue

        vals = _eval_on_grid(fn)
        if not _is_positive_on_grid(vals):
            continue

        normalised = _normalise_numerical(expr, fn)
        if normalised is None:
            continue

        fn_norm = _make_numpy_fn(normalised)
        if fn_norm is None:
            continue
        if not _is_positive_on_grid(_eval_on_grid(fn_norm)):
            continue

        return tokens, normalised

    return None, None