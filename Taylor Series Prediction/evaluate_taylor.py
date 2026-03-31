"""
Metrics
  Token accuracy  : fraction of non-PAD positions with the correct token
  Sequence accuracy: fraction of examples where the full output is correct
  BLEU-4          : corpus-level BLEU score (treats tokens as "words")
  Loss            : cross-entropy on held-out samples
"""

import os
import sys
import json
import argparse
import random
from collections import Counter
import torch
import torch.nn as nn
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from taylor_tokenizer import TaylorTokenizer
from taylor_dataset import load_taylor_dataset, generate_taylor_dataset, save_taylor_dataset, _prefix_to_sympy, custom_collate_fn
from train_taylor import TaylorSeqDataset
from lstm_model import Seq2SeqLSTM
from transformer_model import Seq2SeqTransformer
from training_utils import load_checkpoint
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LSTM, Transformer on Taylor expansion series")
    parser.add_argument("--model", nargs="+", choices=["lstm", "transformer"], default=["lstm", "transformer"], help="Which model(s) to evaluate.")
    parser.add_argument("--out_dir", type=str, default=".\outputs")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=1000, help="Test samples to generate if --data_path not given.")
    parser.add_argument("--save_data", action="store_true", help="Save generated dataset")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-examples", type=int, default=16, help="Number of prediction examples to print.")
    parser.add_argument("--save_results", type=str, default=".\outputs\eval_comparisons.json", help="Path to save comparison JSON.")
    # Model size args (match with training settings)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0, help="Use 0 for eval, disable it")
    parser.add_argument("--seed", type=int, default=99)
    return parser.parse_args()


# ----- BLEU-4 ------
def _ngrams(sequence, n):
    return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]


def corpus_bleu(hypotheses, references, max_n=4):
    """
    Corpus-level BLEU score.

    Parameters
    hypotheses, references : list of list of int

    Returns           bleu : float in [0, 1]
    """
    clip_counts   = Counter()
    total_counts  = Counter()
    hyp_len = ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            hyp_ng  = Counter(_ngrams(hyp, n))
            ref_ng  = Counter(_ngrams(ref, n))
            for gram, cnt in hyp_ng.items():
                clip_counts[n]  += min(cnt, ref_ng[gram])
                total_counts[n] += cnt

    log_bleu = 0.0
    for n in range(1, max_n + 1):
        if total_counts[n] == 0:
            return 0.0
        if clip_counts[n] == 0:
            return 0.0
        log_bleu += math.log(clip_counts[n] / total_counts[n])

    log_bleu /= max_n

    # Brevity penalty
    if hyp_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0

    return bp * math.exp(log_bleu)


def token_accuracy(pred_ids, true_ids, pad_id):
    """Position-wise accuracy excluding PAD positions."""
    correct = total = 0
    for p, t in zip(pred_ids, true_ids):
        if t == pad_id:
            continue
        total += 1
        if p == t:
            correct += 1
    return correct / total if total > 0 else 0.0

def prefix_to_readable(tokens): # Prefix -> Infix
    try:
        expr, _ = _prefix_to_sympy(tokens)
        return str(expr)
    except Exception:
        return " ".join(tokens)   # fallback -> show raw tokens


def evaluate_model(model, loader, tokenizer, device, n_examples=5):
    model.eval()
    all_hyp = []
    all_ref = []
    all_src_tokens = []

    for src, tgt, lbl, src_tokens in loader:
        src = src.to(device)
        preds = model.greedy_decode(src, tokenizer.bos_id, tokenizer.eos_id)
        for i in range(src.size(0)):
            ref = [t for t in lbl[i].tolist() if t != tokenizer.pad_id]
            hyp = preds[i]
            all_hyp.append(hyp)
            all_ref.append(ref)
            all_src_tokens.append(src_tokens[i])

    acc_vals = []
    for hyp, ref in zip(all_hyp, all_ref):
        min_len = min(len(hyp), len(ref))
        if min_len == 0:
            continue
        correct = sum(h == r for h, r in zip(hyp[:min_len], ref[:min_len]))
        acc_vals.append(correct / len(ref))
    token_acc = 100 * sum(acc_vals) / len(acc_vals) if acc_vals else 0.0

    seq_matches = 0
    for hyp, ref in zip(all_hyp, all_ref):
        if tokenizer.decode(hyp, skip_special=True) == tokenizer.decode(ref, skip_special=True):
            seq_matches += 1
    seq_acc = 100 * seq_matches / len(all_hyp)

    bleu = corpus_bleu(all_hyp, all_ref) * 100

    examples = []
    for src_toks, hyp, ref in zip(
        all_src_tokens[:n_examples],
        all_hyp[:n_examples],
        all_ref[:n_examples]
    ):
        examples.append({
            "input":     prefix_to_readable(src_toks),
            "predicted": prefix_to_readable(tokenizer.decode(hyp, skip_special=True)),
            "reference": prefix_to_readable(tokenizer.decode(ref, skip_special=True)),
        })

    return {
        "token_accuracy":    round(token_acc, 2),
        "sequence_accuracy": round(seq_acc, 2),
        "bleu4":             round(bleu, 2),
        "examples":          examples,
    }


def _load_model(model_name, tokenizer, args, device):
    ckpt = os.path.join(args.out_dir, f"best_{model_name}.pt")
    if not os.path.exists(ckpt):
        print(f"  Checkpoint not found: {ckpt}")
        return None

    if model_name == "lstm":
        model = Seq2SeqLSTM(
            vocab_size=tokenizer.vocab_size,
            embed_dim=args.embed_dim,
            hidden_size=args.hidden_size,
            n_layers=args.n_layers,
            dropout=args.dropout,
            pad_id=tokenizer.pad_id,
        )
    else:
        model = Seq2SeqTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            n_heads=4,
            n_enc_layers=args.n_layers,
            n_dec_layers=args.n_layers,
            d_ff=args.d_model * 2,
            dropout=args.dropout,
            pad_id=tokenizer.pad_id,
        )

    load_checkpoint(ckpt, model)
    model = model.to(device)
    model.eval()
    print(f"  Loaded {model_name.upper()} ({sum(p.numel() for p in model.parameters() if p.requires_grad):,} params) from {ckpt}")
    return model


def main():
    args   = parse_args()
    tokenizer    = TaylorTokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Version: {torch.__version__}")
    print("GPU name:", torch.cuda.get_device_name(0))

    # Test data
    if args.data_path and os.path.exists(args.data_path):
        samples = load_taylor_dataset(args.data_path)
        # Use last 10% as test set (same split convention as train_taylor)
        # n_test = max(1, int(len(samples) * 0.1))
        # samples = samples[:n_test]
    else:
        print(f"Generating {args.n_samples} test samples .....")
        samples = generate_taylor_dataset(args.n_samples, seed=args.seed)
        if args.save_data:
            data_dir = os.path.join(args.out_dir, "data")
            save_taylor_dataset(samples, os.path.join(data_dir, "test_dataset.json"))

    ds     = TaylorSeqDataset(samples, tokenizer)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    print(f"Test samples: {len(samples)}\n")

    all_results = {}

    for model_name in args.model:
        print(f"--- {model_name.upper()} ---")
        model = _load_model(model_name, tokenizer, args, device)
        if model is None:
            continue

        results = evaluate_model(model, loader, tokenizer, device, args.n_examples)
        all_results[model_name] = results

        print(f"  Token accuracy    : {results['token_accuracy']}")
        print(f"  Sequence accuracy : {results['sequence_accuracy']}")
        print(f"  BLEU-4            : {results['bleu4']}")
        print()
        print(f"  Prediction examples:")
        for i, ex in enumerate(results["examples"]):
            print(f"    [{i+1}] Input      : {ex['input']}")
            print(f"         Predicted     : {ex['predicted']}")
            print(f"         Ground Truth  : {ex['reference']}")
        print()

    # Side-by-side comparison if both models are evaluated
    if "lstm" in all_results and "transformer" in all_results:
        print("--- Comparison ---")
        metrics = ["token_accuracy", "sequence_accuracy", "bleu4"]
        print(f"  {'Metric':<22} {'LSTM':>10} {'Transformer':>14}")
        print("  " + "-" * 48)
        for m in metrics:
            l_val = all_results["lstm"][m]
            t_val = all_results["transformer"][m]
            winner = " (LSTM wins)" if l_val > t_val else " (Transformer wins)"
            print(f"  {m:<22} {l_val:>10.2f} {t_val:>14.2f}{winner}")

    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()