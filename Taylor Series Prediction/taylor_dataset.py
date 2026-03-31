"""
Generates a dataset of (function, Taylor expansion) pairs using SymPy.

Each sample is:
  - input_tokens  : prefix-notation token list of the original function
  - target_tokens : prefix-notation token list of the Taylor expansion
                    up to 4th order around a chosen expansion point a

Family of Functions used:
  Polynomials   : x, x^2, x^3, x^4, linear combinations
  Trigonometric : sin, cos, tan
  Exponential   : exp
  Logarithmic   : log (shifted to avoid log(0))
  Composites    : sin(x^2), exp(cos(x)), cos(sin(x)), ...

Expansion points:
  a in {0, 1/2, 1} sampled randomly per example.
  a=0 gives the standard Maclaurin series.

Output representation:
  Both the input function and its Taylor expansion are serialised in prefix (Polish) notation 
  so the same tokenizer handles both sides.
  Example:
    sin(x)  ->  ['sin', 'x']
    x - x^3/6  ->  ['+', 'x', '*', '-1/6', 'pow3', 'x']

  SymPy produces the expanded polynomial and then we call expr_to_prefix() to serialise it into tokens.
"""

import random
import warnings
import json
import os
import torch
import sympy as sp
from sympy import symbols, sin, cos, tan, exp, log, series, sqrt, Rational, nsimplify, Integer

x = symbols("x")
EXPANSION_POINTS = [0, Rational(1, 2), 1]
# Maximum absolute value of a rational coefficient we are willing to tokenize.
# Coefficients outside this range are too large for the vocabulary and the sample is discarded.
MAX_COEFF = 120

def _prefix_to_sympy(tokens):
    """
    Inverse of expr_to_prefix. Converts a prefix token list back 
    to a SymPy expression.
    
    Returns:
        (expr, remaining_tokens)
    """
    if not tokens:
        raise ValueError("Empty token list during prefix parsing")
    token_list = list(tokens) # Work with a copy or just use index-based recursion

    def pop_and_parse(data):
        if not data:
            raise ValueError("Unexpected end of tokens")
        
        token = data.pop(0)

        # Variables and Constants
        if token == 'x':
            return sp.Symbol('x')
        
        # Rational strings like "1/2" or "-1/6"
        if '/' in token and not token in ['/', 'sin', 'cos', 'exp']: 
            try:
                p, q = map(int, token.split('/'))
                return Rational(p, q)
            except ValueError:
                pass # fall through to other checks

        # Handle Integers
        try:
            return Integer(int(token))
        except ValueError:
            pass

        # Unary Functions (Arity 1)
        unary_map = {
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
            "neg": lambda e: -e
        }
        if token in unary_map:
            arg = pop_and_parse(data)
            return unary_map[token](arg)

        # 3Named Powers (Arity 1)
        pow_names = {"pow2": 2, "pow3": 3, "pow4": 4}
        if token in pow_names:
            arg = pop_and_parse(data)
            return sp.Pow(arg, pow_names[token])

        # Binary Operators (Arity 2)
        if token == "+":
            left = pop_and_parse(data)
            right = pop_and_parse(data)
            return sp.Add(left, right)
        
        if token == "*":
            left = pop_and_parse(data)
            right = pop_and_parse(data)
            return sp.Mul(left, right)
        
        if token == "/":
            left = pop_and_parse(data)
            right = pop_and_parse(data)
            return left / right
        
        if token == "pow":
            base = pop_and_parse(data)
            exponent = pop_and_parse(data)
            return sp.Pow(base, exponent)

        raise ValueError(f"Unknown token in prefix expression: {token}")

    result_expr = pop_and_parse(token_list)
    return result_expr, token_list



def custom_collate_fn(batch): # batch -> list of tuples (src, tgt, lbl, src_tokens)
    src_list, tgt_list, lbl_list, tokens_list = zip(*batch)
    src_stacked = torch.stack(src_list)
    tgt_stacked = torch.stack(tgt_list)
    lbl_stacked = torch.stack(lbl_list)
    
    return src_stacked, tgt_stacked, lbl_stacked, list(tokens_list)


# Prefix notation serialiser for sympy expressions
def expr_to_prefix(expr):
    """
    Convert a sympy expression to a prefix-notation token list.
    Handles: Add, Mul, Pow, sin, cos, tan, exp, log, Integer, Rational, Symbol (x).
    Returns a list of strings or None
    """
    if isinstance(expr, sp.Symbol):
        return [str(expr)]
    if isinstance(expr, Integer):
        return [str(int(expr))]
    if isinstance(expr, Rational):
        # Represent as "p/q" string token
        return [f"{expr.p}/{expr.q}"]
    if isinstance(expr, sp.Float):
        # Try to convert to a nearby rational
        r = nsimplify(expr, rational=True, tolerance=1e-9)
        return expr_to_prefix(r)
    if isinstance(expr, sp.core.numbers.NegativeOne):
        return ["-1"]
    if isinstance(expr, sp.core.numbers.Half):
        return ["1/2"]
    if isinstance(expr, sp.core.numbers.One):
        return ["1"]
    if isinstance(expr, sp.core.numbers.Zero):
        return ["0"]
    if isinstance(expr, sp.core.numbers.NaN):
        return None

    # Unary functions
    func_map = {
        sp.sin:  "sin",
        sp.cos:  "cos",
        sp.tan:  "tan",
        sp.exp:  "exp",
        sp.log:  "log",
        sp.sqrt: "sqrt",
    }
    if type(expr) in func_map:
        arg_tokens = expr_to_prefix(expr.args[0])
        if arg_tokens is None:
            return None
        return [func_map[type(expr)]] + arg_tokens



    if isinstance(expr, sp.Pow):
        base, exp_val = expr.args
        base_tokens = expr_to_prefix(base)
        if base_tokens is None:
            return None
        pow_names = {2: "pow2", 3: "pow3", 4: "pow4"}
        if isinstance(exp_val, Integer) and int(exp_val) in pow_names:
            return [pow_names[int(exp_val)]] + base_tokens
        # Negative integer exponents -> division by power
        if isinstance(exp_val, Integer) and int(exp_val) < 0:
            n = -int(exp_val)
            if n in pow_names:
                return ["/", "1"] + [pow_names[n]] + base_tokens
        # Rational exponent 1/2 -> sqrt
        if exp_val == Rational(1, 2):
            return ["sqrt"] + base_tokens
        # Generic: represent as pow token
        exp_tokens = expr_to_prefix(exp_val)
        if exp_tokens is None:
            return None
        return ["pow"] + base_tokens + exp_tokens

    # Multiplication: handle as a left-fold of binary * over all args
    if isinstance(expr, sp.Mul):
        args = list(expr.args)
        coeff = None
        rest = []
        for a in args:
            if coeff is None and isinstance(
                a, (Integer, sp.Rational, sp.Float,
                    sp.core.numbers.NegativeOne,
                    sp.core.numbers.Half,
                    sp.core.numbers.One)):
                coeff = a
            else:
                rest.append(a)

        if not rest:
            total = Integer(1)
            for a in args:
                total = total * a
            return expr_to_prefix(total)

        # Fold rest into a binary tree of * tokens without calling sp.Mul
        def fold_mul(terms):
            if len(terms) == 1:
                return expr_to_prefix(terms[0])
            left = expr_to_prefix(terms[0])
            right = fold_mul(terms[1:])
            if left is None or right is None:
                return None
            return ["*"] + left + right

        rest_tokens = fold_mul(rest)
        if rest_tokens is None:
            return None

        if coeff is None or coeff == Integer(1):
            return rest_tokens
        if coeff == Integer(-1) or coeff == sp.core.numbers.NegativeOne():
            return ["neg"] + rest_tokens

        coeff_tokens = expr_to_prefix(coeff)
        if coeff_tokens is None:
            return None
        return ["*"] + coeff_tokens + rest_tokens

    if isinstance(expr, sp.Add):
        args = list(expr.args)
        result = expr_to_prefix(args[-1])
        if result is None:
            return None
        for a in reversed(args[:-1]):
            a_tokens = expr_to_prefix(a)
            if a_tokens is None:
                return None
            result = ["+"] + a_tokens + result
        return result
    return None


# Function Families -> Return a list of (sympy_expr, label_str) pairs covering all required function families.
def _base_functions():
    fns = []

    # Polynomials
    for n in range(1, 5):
        fns.append((x**n, f"x^{n}"))
    for a, b in [(1, 1), (2, -1), (1, 3), (-1, 2)]:
        fns.append((a*x + b, f"{a}x+{b}"))
    for a, b, c in [(1, 1, 1), (1, -1, 2), (2, 0, -1)]:
        fns.append((a*x**2 + b*x + c, f"poly2"))

    # Trig
    for scale in [1, 2, 3]:
        fns.append((sin(scale * x),  f"sin({scale}x)"))
        fns.append((cos(scale * x),  f"cos({scale}x)"))
    fns.append((tan(x), "tan(x)"))
    fns.append((sin(x)**2, "sin^2(x)"))
    fns.append((cos(x)**2, "cos^2(x)"))
    fns.append((sin(x) * cos(x), "sin*cos"))

    # Exponential
    for scale in [1, 2, -1]:
        fns.append((exp(scale * x),   f"exp({scale}x)"))
    fns.append((exp(x**2), "exp(x^2)"))
    fns.append((exp(-x**2), "exp(-x^2)"))

    # Log
    fns.append((log(1 + x),   "log(1+x)"))
    fns.append((log(2 + x),   "log(2+x)"))
    fns.append((log(1 + x**2), "log(1+x^2)"))

    # Composites
    fns.append((sin(x**2),      "sin(x^2)"))
    fns.append((cos(x**2),      "cos(x^2)"))
    fns.append((exp(sin(x)),    "exp(sin(x))"))
    fns.append((exp(cos(x)),    "exp(cos(x))"))
    fns.append((sin(exp(x)),    "sin(exp(x))"))
    fns.append((cos(sin(x)),    "cos(sin(x))"))
    fns.append((sin(x) + exp(x), "sin+exp"))
    fns.append((cos(x) * exp(x), "cos*exp"))
    fns.append((log(1 + sin(x)**2), "log(1+sin^2)"))
    fns.append((exp(x) / (1 + x),  "exp/(1+x)"))

    return fns


# ----- Dataset generation -----
def _compute_taylor(fn_expr, a, order=4):
    """
    Compute the Taylor expansion of fn_expr around x=a up to given order.
    Returns the polynomial as a sympy expression
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = series(fn_expr, x, x0=a, n=order + 1)
        # Remove the O(x^n) remainder term
        poly = s.removeO()
        poly = sp.expand(poly)
        if poly == Integer(0):
            return None
        return poly
    except Exception:
        return None


def _tokens_valid(tokens, vocab): 
    return all(t in vocab for t in tokens) # Returns True if every token in the list is in the known vocabulary


def generate_taylor_dataset(n_samples=5000, order=4, seed=42, verbose=True): # order -> max pow of terms
    """
    Generate a dataset of (input_tokens, target_tokens, expansion_point) triplet

    Returns:
    list of dicts with keys:
        input_tokens, target_tokens  : list of str
        fn_str        : str  (human-readable function string)
        expansion_pt  : str  (in "0", "1/2", "1")
    """
    rng = random.Random(seed)
    vocab = _build_vocab()

    base_fns = _base_functions()
    samples = []
    attempts = 0

    while len(samples) < n_samples:
        attempts += 1
        fn_expr, fn_label = rng.choice(base_fns)
        a = rng.choice(EXPANSION_POINTS)
        taylor = _compute_taylor(fn_expr, a, order)
        if taylor is None:
            continue

        in_tokens = expr_to_prefix(fn_expr)
        tgt_tokens = expr_to_prefix(taylor)

        if in_tokens is None or tgt_tokens is None:
            continue

        # Discard if any token is outside the vocabulary
        if not _tokens_valid(in_tokens, vocab) or not _tokens_valid(tgt_tokens, vocab):
            continue

        if len(in_tokens) > 20 or len(tgt_tokens) > 50: # Discard very long sequences
            continue

        a_str = str(a) if isinstance(a, int) else f"{a.p}/{a.q}"

        samples.append({
            "input_tokens":  in_tokens,
            "target_tokens": tgt_tokens,
            "fn_str":        fn_label,
            "expansion_pt":  a_str,
        })

        if verbose and len(samples) % 100 == 0:
            print(f"  {len(samples)}/{n_samples} samples "
                  f"(attempt {attempts}, "
                  f"success rate {len(samples)/attempts*100:.0f}%)")

    if verbose:
        print(f"Done. {len(samples)} samples from {attempts} attempts.")

    return samples


def _build_vocab():
    """Build the full token vocabulary as a set for membership checks."""
    from taylor.taylor_tokenizer import TaylorTokenizer
    tok = TaylorTokenizer()
    return set(tok.token_to_id.keys())


def save_taylor_dataset(samples, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(samples, f)
    print(f"Saved {len(samples)} samples to {path}")


def load_taylor_dataset(path):
    with open(path) as f:
        return json.load(f)