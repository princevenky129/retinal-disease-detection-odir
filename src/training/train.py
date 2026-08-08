"""
Main training loop.

Handles: AMP mixed precision, gradient clipping, TensorBoard logging,
checkpointing on best validation macro-F1, and early stopping.

Run via: python scripts/run_training.py
"""

import os
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.training.metrics import compute_metrics


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, grad_clip_norm=1.0):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                             enabled=(scaler is not None)):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_labels, all_probs = [], []

    for images, labels in tqdm(loader, desc="Val", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    metrics = compute_metrics(all_labels, all_probs, threshold=threshold)
    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, metrics


def train_model(model, train_dataset, val_dataset, optimizer, scheduler, criterion, cfg):
    device = torch.device(cfg["project"]["device"] if torch.cuda.is_available() else "cpu")
    model.to(device)

    t_cfg = cfg["training"]
    train_loader = DataLoader(
        train_dataset, batch_size=t_cfg["batch_size"], shuffle=True,
        num_workers=t_cfg["num_workers"], pin_memory=True,
    )
    # NOTE: pass a WeightedRandomSampler (src/data/sampler.py) via `sampler=`
    # instead of shuffle=True once the sampler has been built and verified --
    # shuffle and sampler are mutually exclusive in DataLoader.
    val_loader = DataLoader(
        val_dataset, batch_size=t_cfg["batch_size"], shuffle=False,
        num_workers=t_cfg["num_workers"], pin_memory=True,
    )

    scaler = torch.cuda.amp.GradScaler() if (t_cfg["amp"] and device.type == "cuda") else None
    writer = SummaryWriter(log_dir=t_cfg["log_dir"])

    best_metric = -float("inf")
    best_state = None
    epochs_no_improve = 0
    monitor = t_cfg["checkpoint"]["monitor"]

    for epoch in range(1, t_cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            grad_clip_norm=t_cfg["grad_clip_norm"],
        )
        val_loss, val_metrics = evaluate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        current_metric = val_metrics["macro_f1"]
        print(f"Epoch {epoch}/{t_cfg['epochs']} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_macro_f1={current_metric:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metrics/val_macro_f1", val_metrics["macro_f1"], epoch)
        writer.add_scalar("Metrics/val_macro_auc", val_metrics["macro_auc"], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        for c, m in val_metrics["per_class"].items():
            writer.add_scalar(f"F1_per_class/{c}", m["f1"], epoch)

        if current_metric > best_metric:
            best_metric = current_metric
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(t_cfg["checkpoint"]["save_path"]), exist_ok=True)
            torch.save({
                "model_state_dict": best_state,
                "epoch": epoch,
                "val_macro_f1": best_metric,
                "config": cfg,
            }, t_cfg["checkpoint"]["save_path"])
            print(f"  -> New best model saved (val_macro_f1={best_metric:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= t_cfg["early_stopping"]["patience"]:
            print(f"Early stopping at epoch {epoch} (no improvement for "
                  f"{t_cfg['early_stopping']['patience']} epochs).")
            break

    writer.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metric
