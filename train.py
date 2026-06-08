import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import GTSRB
from tqdm import tqdm

from model import STN_CNN, SubsetWithTransform


# --- 1. Argument Parsing ---
def get_args():
    parser = argparse.ArgumentParser(description="Train a road sign recogniser.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory for storing the dataset.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--model_path", type=str, default="model.pt", help="Path to save the trained model.")
    return parser.parse_args()


# --- 2. Main Training Logic ---
def main():
    args = get_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Data Augmentation & Normalization
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669])
    ])

    print("Loading GTSRB dataset via torchvision...")
    try:
        full_dataset = GTSRB(root=args.data_dir, split='train', download=True, transform=None)
    except Exception as e:
        print(f"Error downloading dataset using torchvision: {e}")
        return

    # --- Improvement: Calculate Class Weights for Imbalanced Dataset ---
    print("Calculating class weights to handle imbalance...")
    class_counts = np.zeros(43)
    for _, label in full_dataset:
        class_counts[label] += 1
    
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * 43 # Normalize
    class_weights = class_weights.to(device)
    print("Class weights calculated.")
    # ---

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(range(len(full_dataset)), [train_size, val_size])

    train_dataset = SubsetWithTransform(torch.utils.data.Subset(full_dataset, train_indices), transform=train_transform)
    val_dataset = SubsetWithTransform(torch.utils.data.Subset(full_dataset, val_indices), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Using device: {device}")

    model = STN_CNN(num_classes=43).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # --- Use the calculated weights in the loss function ---
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_pbar.set_postfix({'loss': running_loss / (train_pbar.n + 1)})

        # Evaluation Loop
        model.eval()
        correct = 0
        total = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_pbar.set_postfix({'acc': f'{100 * correct / total:.2f}%'})

        print(f"Epoch [{epoch + 1}/{args.epochs}], Validation Accuracy: {100 * correct / total:.2f}%")

    print("Finished Training")
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
