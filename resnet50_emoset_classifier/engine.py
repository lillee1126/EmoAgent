from __future__ import annotations

from contextlib import nullcontext

import torch
from tqdm import tqdm

from metrics import confusion_matrix, metrics_from_confusion


def _autocast(enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, amp: bool):
    model.train()
    running_loss = 0.0
    correct = 0
    seen = 0
    progress = tqdm(loader, desc="train", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        seen += batch_size
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        progress.set_postfix(loss=f"{running_loss / seen:.4f}", acc=f"{correct / seen:.3f}")
    return {"loss": running_loss / seen, "accuracy": correct / seen}


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp: bool, num_classes: int):
    model.eval()
    running_loss = 0.0
    seen = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with _autocast(amp):
            logits = model(images)
            loss = criterion(logits, targets)
        batch_size = targets.size(0)
        seen += batch_size
        running_loss += loss.item() * batch_size
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
    matrix = confusion_matrix(all_targets, all_predictions, num_classes)
    result = metrics_from_confusion(matrix)
    result["loss"] = running_loss / seen
    return result, matrix

