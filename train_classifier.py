#!/usr/bin/env python3
"""
train_classifier.py

Fine-tunes ResNet50 on select/reject racing footage classification.
Handles class imbalance with weighted loss.

Usage:
    python train_classifier.py --dataset_dir ./dataset --model_output /path/to/models/pca.pt
"""

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def load_dataset(dataset_dir: Path, batch_size: int = 32):
    train_tf, val_tf = get_transforms()
    train_dataset = ImageFolder(dataset_dir / "train", transform=train_tf)
    test_dataset  = ImageFolder(dataset_dir / "test",  transform=val_tf)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader, train_dataset.classes, train_dataset.class_to_idx


def load_class_weights(dataset_dir: Path, class_to_idx: dict) -> torch.Tensor:
    stats = json.loads((dataset_dir / "dataset_stats.json").read_text())
    w = stats["class_weights"]
    weights = torch.zeros(len(class_to_idx), dtype=torch.float32)
    for name, idx in class_to_idx.items():
        weights[idx] = w[name]
    return weights


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def train_classifier(
    dataset_dir: str | Path,
    model_output: str | Path = "model.pt",
    epochs: int = 15,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    progress_cb: Optional[Callable[[dict], None]] = None,
) -> dict:
    """
    Fine-tune ResNet50 on the prepared dataset.

    Progress events emitted via progress_cb:
        {type: "setup",  device: str, train_total: N, test_total: M}
        {type: "epoch",  epoch: N, epochs: total,
                         train_loss: f, train_acc: f,
                         test_loss: f,  test_acc: f,
                         elapsed_s: f,  saved: bool}
        {type: "done",   best_acc: f, model_path: str}

    Returns {"best_acc": float, "model_path": str}.
    """

    def emit(event: dict):
        if progress_cb:
            progress_cb(event)

    dataset_dir  = Path(dataset_dir)
    model_output = Path(model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Data
    train_loader, test_loader, class_names, class_to_idx = load_dataset(dataset_dir, batch_size)
    class_weights = load_class_weights(dataset_dir, class_to_idx).to(device)
    train_total = len(train_loader.dataset)
    test_total  = len(test_loader.dataset)
    print(f"Train: {train_total}  Test: {test_total}  Classes: {class_names}")
    emit({"type": "setup", "device": str(device),
          "train_total": train_total, "test_total": test_total,
          "class_names": class_names})

    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training {epochs} epochs  lr={learning_rate}  batch={batch_size}")

    best_acc = 0.0
    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        saved = test_acc > best_acc
        if saved:
            best_acc = test_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names":      class_names,
                "class_to_idx":     class_to_idx,
                "epoch":            epoch,
                "best_acc":         best_acc,
            }, model_output)

        print(
            f"Epoch {epoch+1:2d}/{epochs}  "
            f"train loss={train_loss:.4f} acc={train_acc:.1f}%  "
            f"test loss={test_loss:.4f} acc={test_acc:.1f}%  "
            f"{'→ saved' if saved else ''} ({elapsed:.1f}s)"
        )
        emit({
            "type":       "epoch",
            "epoch":      epoch + 1,
            "epochs":     epochs,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  2),
            "test_loss":  round(test_loss,  4),
            "test_acc":   round(test_acc,   2),
            "elapsed_s":  round(elapsed,    1),
            "saved":      saved,
        })

    result = {"best_acc": round(best_acc, 2), "model_path": str(model_output)}
    print(f"Done. Best test acc: {best_acc:.2f}%  Model: {model_output}")
    emit({"type": "done", **result})
    return result


def main():
    parser = argparse.ArgumentParser(description="Train select/reject classifier")
    parser.add_argument("--dataset_dir",    required=True)
    parser.add_argument("--model_output",   default="model.pt")
    parser.add_argument("--epochs",         type=int,   default=15)
    parser.add_argument("--learning_rate",  type=float, default=1e-4)
    parser.add_argument("--batch_size",     type=int,   default=32)
    args = parser.parse_args()

    train_classifier(
        args.dataset_dir,
        model_output=args.model_output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
