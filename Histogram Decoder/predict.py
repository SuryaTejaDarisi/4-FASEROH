"""
predict.py

Run inference with a trained HistoDecoder model.

Accepts a histogram from a JSON file or generates a fresh one from a
known function (for quick demos), then decodes it using greedy,
beam-search, and BFGS-refined strategies.

Usage
-----
Demo mode (generates a histogram and shows predictions):
    python predict.py --model-dir outputs --demo

From a histogram file (JSON list of floats summing to 1):
    python predict.py --model-dir outputs --histogram my_histogram.json

Compare all three decoding strategies:
    python predict.py --model-dir outputs --demo --compare-strategies

Save output plot:
    python predict.py --model-dir outputs --demo --save-plot pred.png
"""

import os
import json
import argparse
import numpy as np
import torch

from data.tokenizer import Tokenizer
from data.histogram_generator import (
    expression_to_histogram, histogram_to_float
)
from data.expression_generator import generate_expression
from transformer import HistoDecoder
from decode import greedy_decode, beam_search_decode, bfgs_refine_constants, substitute_constants, tokens_to_sympy
from metrics import histogram_chi2, r2_score_hist
from utils.training_utils import load_checkpoint
from utils.plot_utils import plot_prediction


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HistoDecoder inference on a histogram input"
    )
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing best_model.pt and model_config.json")
    parser.add_argument("--histogram", type=str, default=None,
                        help="Path to JSON file: a list of normalised bin fractions.")
    parser.add_argument("--n-bins", type=int, default=50,
                        help="Number of histogram bins (must match training).")
    parser.add_argument("--beam-width", type=int, default=5,
                        help="Beam width for beam search decoding.")
    parser.add_argument("--demo", action="store_true",
                        help="Generate a random histogram for demonstration.")
    parser.add_argument("--compare-strategies", action="store_true",
                        help="Run greedy, beam, and BFGS and print all results.")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Save prediction plot to this path.")
    parser.add_argument("--n-ops", type=int, default=None,
                        help="Operator count for demo histogram (default: random).")
    return parser.parse_args()


def load_model(model_dir, device):
    """Load model config and weights from a training output directory."""
    config_path = os.path.join(model_dir, "model_config.json")
    ckpt_path = os.path.join(model_dir, "best_model.pt")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"model_config.json not found in {model_dir}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"best_model.pt not found in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)

    model = HistoDecoder(config).to(device)
    load_checkpoint(ckpt_path, model, optimizer=None)
    model.eval()
    return model, config


def decode_all_strategies(model, histogram_float, tokenizer, device,
                          beam_width, histogram_raw=None, n_bins=50):
    """
    Run greedy, beam, and BFGS on the same histogram.
    Returns a dict of results.
    """
    results = {}

    # Greedy
    g_ids, g_consts = greedy_decode(model, histogram_float, tokenizer, device)
    g_tokens = tokenizer.decode(g_ids)
    g_filled = substitute_constants(g_tokens, g_consts)
    g_expr = tokens_to_sympy(g_filled)
    results["greedy"] = {
        "tokens": g_tokens, "filled": g_filled, "expr": str(g_expr)
    }

    # Beam search
    b_ids, b_consts = beam_search_decode(
        model, histogram_float, tokenizer, device,
        beam_width=beam_width
    )
    b_tokens = tokenizer.decode(b_ids)
    b_filled = substitute_constants(b_tokens, b_consts)
    b_expr = tokens_to_sympy(b_filled)
    results["beam"] = {
        "tokens": b_tokens, "filled": b_filled, "expr": str(b_expr)
    }

    # BFGS refinement on beam result
    if histogram_raw is not None:
        ref_consts, success = bfgs_refine_constants(
            b_tokens, b_consts, histogram_raw, n_bins=n_bins
        )
        ref_filled = substitute_constants(b_tokens, ref_consts)
        ref_expr = tokens_to_sympy(ref_filled)
        results["bfgs"] = {
            "tokens": b_tokens, "filled": ref_filled,
            "expr": str(ref_expr), "converged": success
        }

    return results


def main():
    args = parse_args()
    tokenizer = Tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Version: {torch.__version__}")
    print("GPU name:", torch.cuda.get_device_name(0))

    model, config = load_model(args.model_dir, device)
    n_bins = config.get("n_bins", args.n_bins)

    # ------------------------------------------------------------------
    # Obtain histogram
    # ------------------------------------------------------------------
    true_expr_str = None
    histogram_raw = None

    if args.demo:
        print("Generating a demo histogram from a random expression ...")
        prefix_tokens, normalised_expr = generate_expression(n_ops=args.n_ops)
        if prefix_tokens is None:
            print("Expression generation failed. Try again.")
            return
        histogram_raw, _, _ = expression_to_histogram(
            normalised_expr, n_bins=n_bins, total_count=500
        )
        histogram_float = histogram_to_float(histogram_raw)
        true_expr_str = str(normalised_expr)
        print(f"True expression: {true_expr_str}")
        print(f"True prefix tokens: {prefix_tokens}")

    elif args.histogram:
        with open(args.histogram) as f:
            hist_data = json.load(f)
        histogram_float = np.array(hist_data, dtype=np.float32)
        if len(histogram_float) != n_bins:
            raise ValueError(
                f"Histogram has {len(histogram_float)} bins, "
                f"model expects {n_bins}"
            )
        histogram_raw = (histogram_float * 500).astype(int)

    else:
        print("Provide --demo or --histogram. Use --help for options.")
        return

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    if args.compare_strategies:
        results = decode_all_strategies(
            model, histogram_float, tokenizer, device,
            beam_width=args.beam_width,
            histogram_raw=histogram_raw,
            n_bins=n_bins
        )
        print("\n--- Decoding Results ---")
        for strategy, info in results.items():
            print(f"\n{strategy.upper()}")
            print(f"  Tokens : {info['tokens']}")
            print(f"  Filled : {info['filled']}")
            print(f"  Expr   : {info['expr']}")
    else:
        # Default: beam + BFGS
        b_ids, b_consts = beam_search_decode(
            model, histogram_float, tokenizer, device,
            beam_width=args.beam_width
        )
        b_tokens = tokenizer.decode(b_ids)
        b_filled = substitute_constants(b_tokens, b_consts)
        b_expr = tokens_to_sympy(b_filled)
        print(f"\nBeam search prediction : {b_filled}")
        print(f"Sympy expression       : {b_expr}")

        if histogram_raw is not None and b_consts:
            ref_consts, success = bfgs_refine_constants(
                b_tokens, b_consts, histogram_raw, n_bins=n_bins
            )
            ref_filled = substitute_constants(b_tokens, ref_consts)
            ref_expr = tokens_to_sympy(ref_filled)
            print(f"After BFGS refinement  : {ref_filled}")
            print(f"BFGS converged         : {success}")
        else:
            ref_expr = b_expr
            ref_filled = b_filled

        if args.save_plot:
            plot_prediction(
                histogram_float, ref_expr,
                true_expr_str=true_expr_str,
                n_bins=n_bins,
                save_path=args.save_plot
            )
            print(f"Plot saved to {args.save_plot}")


if __name__ == "__main__":
    main()