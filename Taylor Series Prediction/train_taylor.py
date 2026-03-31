import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from taylor_tokenizer import TaylorTokenizer
from taylor_dataset import generate_taylor_dataset, save_taylor_dataset, load_taylor_dataset, custom_collate_fn
from lstm_model import Seq2SeqLSTM
from transformer_model import Seq2SeqTransformer
from training_utils import set_seed, save_checkpoint, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM or Transformer on Taylor expansion series")

    parser.add_argument("--model", choices=["lstm", "transformer"], required=True, help="Which model to train.")
    parser.add_argument("--data_path", type=str, default=None, help="Path to saved dataset JSON. If not given, data is generated.")
    parser.add_argument("--n_samples", type=int, default=12000)
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data used for validation.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10, help="Patience counter for Early Stopping")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256, help="LSTM hidden size.")
    parser.add_argument("--d-model", type=int, default=128, help="Transformer d_model.")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of LSTM layers / Transformer enc+dec layers.")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--out_dir", type=str, default=".\outputs", help="Directory for checkpoints and logs.")
    parser.add_argument("--save_data", action="store_true", help="Save generated dataset")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


class TaylorSeqDataset(Dataset): # Pytorch dataset
    """
    Wraps a list of (input_tokens, target_tokens) pairs and tensorises
    them with padding so they can be batched.
    """
    def __init__(self, samples, tokenizer, max_src=25, max_tgt=55):
        self.tokenizer = tokenizer
        self.max_src   = max_src
        self.max_tgt   = max_tgt

        self.src_tensors = []
        self.tgt_tensors = []   # decoder input  (BOS + tokens, no EOS)
        self.lbl_tensors = []   # decoder target (tokens + EOS, no BOS)
        self.src_token_lists = []   # keep raw input tokens for display

        for s in samples:
            src_ids = tokenizer.encode(s["input_tokens"])
            full_ids = tokenizer.wrap(s["target_tokens"])   # BOS+tokens+EOS

            src_ids  = tokenizer.pad_sequence(src_ids, max_src)
            tgt_ids  = tokenizer.pad_sequence(full_ids[:-1], max_tgt)  # drop EOS
            lbl_ids  = tokenizer.pad_sequence(full_ids[1:],  max_tgt)  # drop BOS

            self.src_tensors.append(torch.tensor(src_ids, dtype=torch.long))
            self.tgt_tensors.append(torch.tensor(tgt_ids, dtype=torch.long))
            self.lbl_tensors.append(torch.tensor(lbl_ids, dtype=torch.long))
            self.src_token_lists.append(s["input_tokens"])

    def __len__(self):
        return len(self.src_tensors)

    def __getitem__(self, idx):
        return (self.src_tensors[idx],
            self.tgt_tensors[idx],
            self.lbl_tensors[idx],
            self.src_token_lists[idx]) # Added this to print samples in evaluate.py


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, is_train, pad_id):
    model.train() if is_train else model.eval()
    meter = AverageMeter()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for src, tgt, lbl, _ in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            lbl = lbl.to(device)

            logits = model(src, tgt)           # (B, T, V)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B * T, V), lbl.reshape(B * T))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            meter.update(loss.item(), src.size(0))

    return meter.avg

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Version: {torch.__version__}")
    print("GPU name:", torch.cuda.get_device_name(0))

    tokenizer = TaylorTokenizer()

    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading dataset from {args.data_path}")
        all_samples = load_taylor_dataset(args.data_path)
    else:
        print(f"Generating {args.n_samples} samples ...")
        all_samples = generate_taylor_dataset(args.n_samples, seed=args.seed)
        if args.save_data:
            data_dir = os.path.join(args.out_dir, "data")
            save_taylor_dataset(all_samples, os.path.join(data_dir, "dataset.json"))

    # Train / val split
    random.Random(args.seed).shuffle(all_samples)
    n_val   = max(1, int(len(all_samples) * args.val_split))
    val_samples   = all_samples[:n_val]
    train_samples = all_samples[n_val:]
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    train_ds = TaylorSeqDataset(train_samples, tokenizer)
    val_ds   = TaylorSeqDataset(val_samples,   tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    if args.model == "lstm":
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

    model = model.to(device)
    print(f"Model: {args.model.upper()} | Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience_counter = 0
    ckpt_path = os.path.join(args.out_dir, f"best_{args.model}.pt")

    print(f"\nTraining {args.model.upper()} for {args.epochs} epochs ...")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True, pad_id=tokenizer.pad_id)
        val_loss   = run_epoch(model, val_loader, criterion, optimizer, device, is_train=False, pad_id=tokenizer.pad_id)
        scheduler.step()

        print(f"  Epoch {epoch:3d}/{args.epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, best_val, ckpt_path)
            patience_counter = 0
            print("NEW BEST MODEL !!!")
        else:
            patience_counter+=1
            print(f"No improvement [{patience_counter}/{args.patience}]")
            if (patience_counter == args.patience):
                print("Early Stopping triggered!!!")
                break

        # if epoch % 5 == 0 or epoch == 1:
        # print(f"  Epoch {epoch:3d}/{args.epochs} | train={train_loss:.4f}  val={val_loss:.4f}")
        print("-"*100)

    # Save history for plots
    hist_path = os.path.join(args.out_dir, f"history_{args.model}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"History: {hist_path}")


if __name__ == "__main__":
    main()