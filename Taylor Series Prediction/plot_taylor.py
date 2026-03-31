import os
import sys
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Taylor training results")
    parser.add_argument("--out_dir", type=str, default=".\outputs")
    parser.add_argument("--save_dir", type=str, default=".\outputs\plots")
    return parser.parse_args()


def plot_training_curves(out_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"lstm": "steelblue", "transformer": "coral"}

    for model_name, color in colors.items():
        hist_path = os.path.join(out_dir, f"history_{model_name}.json")
        if not os.path.exists(hist_path):
            continue
        with open(hist_path) as f:
            h = json.load(f)
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["train_loss"], color=color, label=f"{model_name.upper()} train", linewidth=1.5)
        axes[1].plot(epochs, h["val_loss"],   color=color, label=f"{model_name.upper()} val", linewidth=1.5, linestyle="--")

    for ax, title in zip(axes, ["Training Loss", "Validation Loss"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.25)

    plt.suptitle("LSTM vs Transformer: Training Curves", fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_metric_comparison(results_path, save_dir):
    if not os.path.exists(results_path):
        return
    with open(results_path) as f:
        results = json.load(f)

    metrics = ["token_accuracy", "sequence_accuracy", "bleu4"]
    labels  = ["Token Accuracy", "Sequence Accuracy", "BLEU-4"]
    models  = [m for m in ["lstm", "transformer"] if m in results]

    if len(models) < 2:
        return

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (model, color) in enumerate(zip(models, ["steelblue", "coral"])):
        vals = [results[model][m] for m in metrics]
        bars = ax.bar([v + i * width for v in x], vals, width, label=model.upper(), color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks([v + width / 2 for v in x])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("LSTM vs Transformer: Evaluation Metrics")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(save_dir, "metric_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    args = parse_args()
    plot_training_curves(args.out_dir, args.save_dir)
    results_path = os.path.join(args.out_dir, "eval_results.json")
    plot_metric_comparison(results_path, args.save_dir)

if __name__ == "__main__":
    main()