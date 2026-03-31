import os
import json
import argparse
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset import generate_dataset, save_dataset, load_dataset, get_dataloader
from data.tokenizer import Tokenizer
from transformer import HistoDecoder, SymbolicLoss
from utils.training_utils import set_seed, save_checkpoint, load_checkpoint, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description="Symbolic Regression from Histograms")

    parser.add_argument("--train_data", type=str, default=None, help="Path to saved training dataset JSON. If not provided, data is generated.")
    parser.add_argument("--val_data", type=str, default=None, help="Path to saved validation dataset JSON")
    parser.add_argument("--n_train", type=int, default=10000, help="Number of training samples to generate")
    parser.add_argument("--n_val", type=int, default=2000, help="Number of validation samples to generate.")
    parser.add_argument("--n-bins", type=int, default=50, help="Number of histogram bins (encoder input length)")
    parser.add_argument("--total_count", type=int, default=500, help="Total counts per histogram (controls noise level)")
    parser.add_argument("--save_data", action="store_true", help="Save generated datasets for reuse")
    # Model hyperparameters
    parser.add_argument("--d-model", type=int, default=256, help="Transformer hidden dimension")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n-enc-layers", type=int, default=3, help="Number of encoder layers")
    parser.add_argument("--n-dec-layers", type=int, default=3, help="Number of decoder layers")
    parser.add_argument("--d-ff", type=int, default=256, help="Feed-forward layer dimension inside Transformer")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    # Training settings
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--lambda-const", type=float, default=0.5, help="Weight of constant regression loss")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Number of linear warmup steps for the LR")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="Gradient clipping norm. use 0 for no clipping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="outputs", help="Directory for checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50, help="Print training loss every N batches.")
    parser.add_argument("--patience", type=int, default=10, help="No. of epcohs for Early Stopping")

    return parser.parse_args()

def warmup_lambda(step, warmup_steps):
    """Linear warmup schedule factor."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


def run_epoch(model, loader, criterion, optimizer, scheduler, device, is_train, clip_grad, log_interval, epoch):
    if is_train:
        model.train()
        mode_str = "train"
    else:
        model.eval()
        mode_str = "val"

    total_meter = AverageMeter()
    ce_meter = AverageMeter()
    mse_meter = AverageMeter()
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch_idx, batch in enumerate(loader):
            histogram = batch["encoder_input"].to(device)
            dec_input = batch["decoder_input"].to(device)
            dec_target = batch["decoder_target"].to(device)
            constants = batch["constants"].to(device)

            logits, const_preds = model(histogram, dec_input, constants)
            total_loss, ce_loss, mse_loss = criterion(logits, const_preds, dec_target, constants)

            if is_train:
                optimizer.zero_grad()
                total_loss.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                scheduler.step()

            bs = histogram.size(0)
            total_meter.update(total_loss.item(), bs)
            ce_meter.update(ce_loss.item(), bs)
            mse_meter.update(mse_loss.item(), bs)

            if is_train and (batch_idx + 1) % log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                      f"loss={total_meter.avg:.4f}  "
                      f"ce={ce_meter.avg:.4f}  "
                      f"mse={mse_meter.avg:.4f}  "
                      f"lr={lr:.2e}")

    return total_meter.avg, ce_meter.avg, mse_meter.avg


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Version: {torch.__version__}")
    print("GPU name:", torch.cuda.get_device_name(0))

    tokenizer = Tokenizer()

    # Data Loading and Generation
    if args.train_data and os.path.exists(args.train_data):
        print(f"Loading training data from {args.train_data}")
        train_dataset = load_dataset(args.train_data)
    else:
        print(f"Generating {args.n_train} training samples ...")
        train_dataset = generate_dataset(
            args.n_train, n_bins=args.n_bins,
            total_count=args.total_count, seed=args.seed
        )
        if args.save_data:
            save_dataset(train_dataset, os.path.join(args.out_dir, "train.json"))

    if args.val_data and os.path.exists(args.val_data):
        print(f"Loading validation data from {args.val_data}")
        val_dataset = load_dataset(args.val_data)
    else:
        print(f"Generating {args.n_val} validation samples ...")
        val_dataset = generate_dataset(
            args.n_val, n_bins=args.n_bins,
            total_count=args.total_count, seed=args.seed + 1
        )
        if args.save_data:
            save_dataset(val_dataset, os.path.join(args.out_dir, "val.json"))

    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=args.num_workers)
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

    # ------------------------------------------------------------------
    # Model, loss, optimiser
    # ------------------------------------------------------------------
    config = {
        "n_bins": args.n_bins,
        "vocab_size": tokenizer.vocab_size,
        "const_id": tokenizer.const_id,
        "pad_id": tokenizer.pad_id,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_encoder_layers": args.n_enc_layers,
        "n_decoder_layers": args.n_dec_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "max_seq": 64,
    }
    model = HistoDecoder(config).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    criterion = SymbolicLoss(
        pad_id=tokenizer.pad_id,
        const_id=tokenizer.const_id,
        lambda_const=args.lambda_const,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    total_steps = args.epochs * len(train_loader)
    # Warmup then cosine anneal
    warmup_sched = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: warmup_lambda(s, args.warmup_steps))
    cosine_sched = CosineAnnealingLR(optimizer, T_max=total_steps - args.warmup_steps, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[args.warmup_steps]
    )

    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0

    if args.resume and os.path.exists(args.resume):
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer
        )
        print(f"\nResumed from epoch {start_epoch} | Best val loss = {best_val_loss:.4f}\n")

    # Save config
    with open(os.path.join(args.out_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Training Loop
    history = {"train_loss": [], "val_loss": [], "train_ce": [], "val_ce": []}
    print("\nStarting training ...")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_ce, train_mse = run_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, is_train=True,
            clip_grad=args.clip_grad,
            log_interval=args.log_interval,
            epoch=epoch
        )

        val_loss, val_ce, val_mse = run_epoch(
            model, val_loader, criterion, optimizer, scheduler,
            device, is_train=False,
            clip_grad=args.clip_grad,
            log_interval=args.log_interval,
            epoch=epoch
        )

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train={train_loss:.4f} (ce={train_ce:.4f} mse={train_mse:.4f}) | "
              f"val={val_loss:.4f} (ce={val_ce:.4f} mse={val_mse:.4f}) | "
              f"time={elapsed:.1f}s")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_ce"].append(train_ce)
        history["val_ce"].append(val_ce)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, best_val_loss,
                os.path.join(args.out_dir, "best_model.pt")
            )
            patience_counter=0
            print(f"-----NEW BEST MODEL saved (val_loss={best_val_loss:.4f})-----")
        else:
            patience_counter+=1
            if patience_counter == args.patience:
                print("Early Stopping Encountered")
                break
            else:
                print(f"No improvement [{patience_counter}/{args.patience}]")
        print("-"*100)

        # Save latest checkpoint (for resuming)
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            os.path.join(args.out_dir, "latest_model.pt")
        )
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining completed. Best val loss: {best_val_loss:.4f}")
    print(f"Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()