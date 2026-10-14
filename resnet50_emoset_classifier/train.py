from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50

from emoset import EMOTIONS, EmoSetDataset, build_transforms, default_dataset_root
from engine import evaluate, train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-50 on EmoSet-118K")
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--output-dir", type=Path, default=Path("runs/resnet50_emoset"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--limit", type=int, help="Limit each split for a quick smoke run")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(pretrained: bool):
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(EMOTIONS))
    return model


def save_confusion_plot(matrix: np.ndarray, path: Path):
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("EmoSet-118K test confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_tf, eval_tf = build_transforms(args.image_size)
    common = dict(root=args.dataset_root, verify_paths=args.verify_paths, limit=args.limit)
    train_set = EmoSetDataset(split="train", transform=train_tf, **common)
    val_set = EmoSetDataset(split="val", transform=eval_tf, **common)
    test_set = EmoSetDataset(split="test", transform=eval_tf, **common)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=amp,
        persistent_workers=args.workers > 0,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    model = build_model(not args.no_pretrained).to(device)
    if args.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    weights = None
    if not args.no_class_weights:
        counts = torch.tensor(train_set.class_counts, dtype=torch.float32)
        weights = counts.sum() / (len(counts) * counts.clamp_min(1))
        weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    start_epoch = 0
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        target_model = model.module if isinstance(model, nn.DataParallel) else model
        target_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_macro_f1 = checkpoint["best_macro_f1"]

    history_path = args.output_dir / "history.csv"
    history_exists = history_path.exists() and start_epoch > 0
    with history_path.open("a", newline="", encoding="utf-8") as history_stream:
        writer = csv.DictWriter(
            history_stream,
            fieldnames=["epoch", "lr", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_macro_f1", "val_balanced_accuracy"],
        )
        if not history_exists:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs):
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, amp)
            val_metrics, _ = evaluate(model, val_loader, criterion, device, amp, len(EMOTIONS))
            scheduler.step()
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            }
            writer.writerow(row)
            history_stream.flush()
            print(json.dumps(row, ensure_ascii=False))

            target_model = model.module if isinstance(model, nn.DataParallel) else model
            state = {
                "epoch": epoch,
                "model": target_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_macro_f1": max(best_macro_f1, val_metrics["macro_f1"]),
                "class_names": EMOTIONS,
                "args": vars(args),
            }
            torch.save(state, args.output_dir / "last.pt")
            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                epochs_without_improvement = 0
                torch.save(state, args.output_dir / "best.pt")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

    best = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    target_model.load_state_dict(best["model"])
    test_metrics, matrix = evaluate(model, test_loader, criterion, device, amp, len(EMOTIONS))
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    np.savetxt(args.output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    save_confusion_plot(matrix, args.output_dir / "confusion_matrix.png")
    print("Test:", json.dumps(test_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
