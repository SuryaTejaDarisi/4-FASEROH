"""
evaluate.py

Evaluates a trained HistoDecoder model on a held-out test dataset.

Computes:
  - Mean chi-squared distance between observed and predicted histograms
  - Mean R^2 score
  - Token-level accuracy
  - Improvement from BFGS constant refinement vs raw beam search

Usage
-----
    python evaluate.py --model-dir outputs --test-data data/test.json

    python evaluate.py --model-dir outputs --n-test 300 --n-bins 50
"""

import os
import json
import argparse
import numpy as np
import torch

from data.dataset import generate_dataset, load_dataset, save_dataset
from data.tokenizer import Tokenizer
# from data.histogram_generator import histogram_to_float
from transformer import HistoDecoder
from decode import beam_search_decode, substitute_constants, tokens_to_sympy, bfgs_refine_constants
from metrics import histogram_chi2, r2_score_hist, token_accuracy
from utils.training_utils import load_checkpoint
from utils.plot_utils import plot_evaluation_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Histogram Decoder on a test dataset")
    parser.add_argument("--model_dir", type=str, default="./outputs")
    parser.add_argument("--test_data", type=str, default=None, help="Path to saved test dataset JSON.")
    parser.add_argument("--n_test", type=int, default=1000, help="Test samples to generate if --test-data not given.")
    parser.add_argument("--save_data", action="store_true", help="Save generated datasets for reuse")
    parser.add_argument("--n-bins", type=int, default=50)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--use_bfgs", action="store_true", default=True, help="Also evaluate with BFGS constant refinement.")
    parser.add_argument("--save_plot", type=str, default="./outputs", help="Path to save evaluation summary plot.")
    parser.add_argument("--out_file", type=str, default=None, help="Save evaluation results to a JSON file.")
    return parser.parse_args()


def load_model(model_dir, device):
    config_path = os.path.join(model_dir, "model_config.json")
    ckpt_path = os.path.join(model_dir, "best_model.pt")
    with open(config_path) as f:
        config = json.load(f)
    model = HistoDecoder(config).to(device)
    load_checkpoint(ckpt_path, model, optimizer=None)
    model.eval()
    return model, config

# Evaluates one sample. Returns a dict of per-sample metrics.
def evaluate_sample(model, sample, tokenizer, device, beam_width, use_bfgs, n_bins, total_count=500):
    histogram_float = sample["encoder_input"].numpy()
    histogram_raw = (histogram_float * total_count).astype(int)
    true_ids = sample["decoder_target"].tolist()

    # Beam search
    b_ids, b_consts = beam_search_decode(model, histogram_float, tokenizer, device, beam_width=beam_width)

    b_tokens = tokenizer.decode(b_ids)
    b_filled = substitute_constants(b_tokens, b_consts)
    b_expr = tokens_to_sympy(b_filled)

    result = {
        "token_acc": token_accuracy(b_ids, true_ids, pad_id=tokenizer.pad_id),
        "chi2_beam": float("inf"),
        "r2_beam": -float("inf"),
        "chi2_bfgs": float("inf"),
        "r2_bfgs": -float("inf"),
    }

    if b_expr is not None:
        result["chi2_beam"] = histogram_chi2(histogram_raw, b_expr, n_bins)
        result["r2_beam"] = r2_score_hist(histogram_raw, b_expr, n_bins)

    if use_bfgs and b_consts:
        ref_consts, _ = bfgs_refine_constants(
            b_tokens, b_consts, histogram_raw, n_bins=n_bins
        )
        ref_filled = substitute_constants(b_tokens, ref_consts)
        ref_expr = tokens_to_sympy(ref_filled)
        if ref_expr is not None:
            result["chi2_bfgs"] = histogram_chi2(histogram_raw, ref_expr, n_bins)
            result["r2_bfgs"] = r2_score_hist(histogram_raw, ref_expr, n_bins)

    return result

def fmt(v):
    return f"{v:.4f}" if v == v else "N/A (no BFGS samples)"

def main():
    args = parse_args()
    tokenizer = Tokenizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Version: {torch.__version__}")
    print("GPU name:", torch.cuda.get_device_name(0))

    model, config = load_model(args.model_dir, device)
    n_bins = config.get("n_bins", args.n_bins)

    if args.test_data and os.path.exists(args.test_data):
        print(f"Loading test data from {args.test_data}")
        from data.dataset import load_dataset
        test_dataset = load_dataset(args.test_data)
    else:
        print(f"Generating {args.n_test} test samples .....")
        test_dataset = generate_dataset(args.n_test, n_bins=n_bins, total_count=500, seed=9999)
        if args.save_data:
            save_dataset(test_dataset, os.path.join(args.out_dir, "test.json"))

    print(f"Evaluating on {len(test_dataset)} samples .....")

    all_results = []
    for i, sample in enumerate(test_dataset.samples):
        res = evaluate_sample(
            model, sample, tokenizer, device,
            beam_width=args.beam_width,
            use_bfgs=args.use_bfgs,
            n_bins=n_bins
        )
        all_results.append(res)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(test_dataset.samples)}")

    # Aggregate
    def safe_mean(vals):
        finite = [v for v in vals if v != float("inf") and v != -float("inf") and v==v]
        return float(np.mean(finite)) if finite else float("nan")

    summary = {
        "n_samples": len(all_results),
        "mean_token_accuracy": safe_mean([r["token_acc"] for r in all_results]),
        "mean_chi2_beam": safe_mean([r["chi2_beam"] for r in all_results]),
        "mean_r2_beam": safe_mean([r["r2_beam"] for r in all_results]),
        "mean_chi2_bfgs": safe_mean([r["chi2_bfgs"] for r in all_results]),
        "mean_r2_bfgs": safe_mean([r["r2_bfgs"] for r in all_results]),
        "pct_finite_beam": 100 * sum(
            1 for r in all_results if r["chi2_beam"] < float("inf")
        ) / len(all_results),
    }

    print("\n----- Evaluation Summary -----")
    print(f"  Samples evaluated    : {summary['n_samples']}")
    print(f"  Token accuracy       : {summary['mean_token_accuracy']:.4f}")
    print(f"  Mean chi2 (beam)     : {summary['mean_chi2_beam']:.4f}")
    print(f"  Mean R2   (beam)     : {summary['mean_r2_beam']:.4f}")
    print(f"  Mean chi2 (BFGS)     : {fmt(summary['mean_chi2_bfgs'])}")
    print(f"  Mean R2   (BFGS)     : {fmt(summary['mean_r2_bfgs'])}")
    print(f"  Valid predictions %  : {summary['pct_finite_beam']:.1f}%%")

    if args.out_file:
        with open(args.out_file, "w") as f:
            json.dump({"summary": summary, "per_sample": all_results}, f, indent=2)
        print(f"Results saved to {args.out_file}")

    if args.save_plot:
        plot_evaluation_summary(all_results, save_path=args.save_plot)
        print(f"Evaluation plot saved to {args.save_plot}")


if __name__ == "__main__":
    main()