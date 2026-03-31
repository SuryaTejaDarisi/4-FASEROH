import os
import sys
import argparse
import random
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from taylor.taylor_dataset import generate_taylor_dataset, load_taylor_dataset
from taylor.taylor_tokenizer import TaylorTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Visualise Taylor dataset")
    parser.add_argument("--data_path", type=str, default=None, help="Path to saved dataset JSON. If not given, a small dataset is generated.")
    parser.add_argument("--n_samples", type=int, default=1000, help="Samples to generate if no data path given.")
    parser.add_argument("--n-examples", type=int, default=16, help="Number of examples to print.")
    parser.add_argument("--save_dir", type=str, default=".\outputs\plots", help="Where to save plots.")
    return parser.parse_args()


def plot_distributions(samples, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Count function types
    fn_counts = Counter(s["fn_str"] for s in samples)
    pt_counts = Counter(s["expansion_pt"] for s in samples)
    in_lens  = [len(s["input_tokens"])  for s in samples]
    tgt_lens = [len(s["target_tokens"]) for s in samples]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Function type distribution
    top_fns = fn_counts.most_common(15)
    axes[0, 0].barh([f for f, _ in top_fns], [c for _, c in top_fns],color="steelblue")
    axes[0, 0].set_title("Function Type Distribution (Top 15)")
    axes[0, 0].set_xlabel("Count")

    pts = list(pt_counts.keys())
    axes[0, 1].bar(pts, [pt_counts[p] for p in pts], color="coral")
    axes[0, 1].set_title("Expansion Point Distribution")
    axes[0, 1].set_xlabel("Expansion Point (a)")
    axes[0, 1].set_ylabel("Count")

    # Input sequence length histogram
    axes[1, 0].hist(in_lens, bins=range(1, max(in_lens) + 2), color="steelblue", edgecolor="white")
    axes[1, 0].set_title(f"Input Sequence Length (mean={sum(in_lens)/len(in_lens):.1f})")
    axes[1, 0].set_xlabel("Tokens")
    axes[1, 0].set_ylabel("Count")

    # Target sequence length histogram
    axes[1, 1].hist(tgt_lens, bins=range(1, min(max(tgt_lens) + 2, 60)), color="coral", edgecolor="white")
    axes[1, 1].set_title(f"Target Sequence Length (mean={sum(tgt_lens)/len(tgt_lens):.1f})")
    axes[1, 1].set_xlabel("Tokens")
    axes[1, 1].set_ylabel("Count")

    plt.suptitle(f"Taylor Dataset Overview ({len(samples)} samples)", fontsize=13)
    plt.tight_layout()

    path = os.path.join(save_dir, "dataset_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def print_examples(samples, tokenizer, n=8, rng_seed=0):
    rng = random.Random(rng_seed)
    chosen = rng.sample(samples, min(n, len(samples)))

    print(f"\n----- {len(chosen)} Random Dataset Examples -----\n")
    print(f"{'#':<4} {'Expansion pt':<14} {'Input function':<28} {'Taylor expansion (prefix)'}")
    print("-" * 90)
    for i, s in enumerate(chosen):
        in_str  = " ".join(s["input_tokens"])
        tgt_str = " ".join(s["target_tokens"])
        pt_str  = s["expansion_pt"]
        print(f"{i+1:<4} a={pt_str:<12} {in_str:<28} {tgt_str}")
    print()


def main():
    args = parse_args()
    tok  = TaylorTokenizer()

    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading dataset from {args.data_path}")
        samples = load_taylor_dataset(args.data_path)
    else:
        print(f"Generating {args.n_samples} samples for visualisation ....")
        samples = generate_taylor_dataset(args.n_samples, verbose=False)

    print(f"Total samples: {len(samples)}")
    print(f"Vocabulary size: {tok.vocab_size}")

    plot_distributions(samples, args.save_dir)
    print_examples(samples, tok, n=args.n_examples)


if __name__ == "__main__":
    main()