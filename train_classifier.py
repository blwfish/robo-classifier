#!/usr/bin/env python3
"""
train_classifier.py

Fine-tunes ResNet50 on interesting vs. boring racing footage classification.
Handles class imbalance with weighted loss.

Usage:
    python train_classifier.py --dataset_dir ./dataset --model_output model.pt
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import argparse
import json
from pathlib import Path
import time

def get_transforms():
    """Return train and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform

def load_dataset(dataset_dir, batch_size=32):
    """Load training and test datasets."""
    dataset_dir = Path(dataset_dir)
    train_transform, val_transform = get_transforms()
    
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    test_dataset = ImageFolder(test_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Class names and indices
    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    
    print(f"Classes: {class_names}")
    print(f"Class indices: {class_to_idx}")
    
    return train_loader, test_loader, class_names, class_to_idx

def load_class_weights(dataset_dir):
    """Load precomputed class weights from dataset_stats.json."""
    stats_file = Path(dataset_dir) / "dataset_stats.json"
    with open(stats_file, 'r') as f:
        stats = json.load(f)

    weights = stats['class_weights']
    # Order matters: alphabetically, "reject" comes first, "select" second
    weight_tensor = torch.tensor([weights['reject'], weights['select']], dtype=torch.float32)

    print(f"Class weights: {weight_tensor}")
    return weight_tensor

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels

def train_classifier(dataset_dir, model_output="model.pt", epochs=15, learning_rate=1e-4, batch_size=32):
    """
    Fine-tune ResNet50 on the classification task.
    """
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader, class_names, class_to_idx = load_dataset(dataset_dir, batch_size=batch_size)
    class_weights = load_class_weights(dataset_dir)
    class_weights = class_weights.to(device)
    
    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Replace final layer (ImageNet has 1000 classes, we have 2)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    model = model.to(device)
    
    # Loss with class weights to handle imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nTraining ResNet50 for {epochs} epochs")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}\n")
    
    best_test_acc = 0.0
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}% | "
              f"Time: {elapsed:.1f}s")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'class_to_idx': class_to_idx,
                'epoch': epoch
            }, model_output)
            print(f"  → Saved model (best test acc: {best_test_acc:.2f}%)")
    
    print(f"\nTraining complete. Best test accuracy: {best_test_acc:.2f}%")
    print(f"Model saved to: {model_output}")

def main():
    parser = argparse.ArgumentParser(description="Train interesting/boring classifier")
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Path to dataset directory (from prepare_training_data.py)"
    )
    parser.add_argument(
        "--model_output",
        default="model.pt",
        help="Output path for trained model (default: model.pt)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    
    args = parser.parse_args()
    
    train_classifier(
        args.dataset_dir,
        model_output=args.model_output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
