import argparse
import random

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import GTSRB
from tqdm import tqdm

from model import STN_CNN, SubsetWithTransform
from torchvision.datasets import CIFAR10
from torch.utils.data import ConcatDataset, Dataset

class BackgroundDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        return img, 43

class NoiseBackgroundDataset(Dataset):
    def __init__(self, num_samples=10000):
        self.num_samples = num_samples
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        if random.random() < 0.5:
            # Solid color to prevent hallucinating on zoomed-in sign patches
            color = np.random.randint(0, 255, size=(3,))
            img = (np.ones((50, 50, 3)) * color).astype(np.uint8)
        else:
            # Random static noise
            img = np.random.randint(0, 255, size=(50, 50, 3), dtype=np.uint8)
        return Image.fromarray(img), 43


# --- Domain-bridging augmentation: GTSRB (German white signs) → Polish (yellow signs) ---
class RandomWhiteToYellow:
    """Randomly recolors white/light-gray sign regions to yellow.

    Polish warning signs use a yellow background, while the GTSRB training
    data (German signs) uses white. This transform detects neutral bright
    pixels and shifts them toward yellow by suppressing the blue channel,
    teaching the model to be invariant to this domain-specific color difference.
    """
    def __init__(self, p=0.3, brightness_thresh=170, spread_thresh=60):
        self.p = p
        self.brightness_thresh = brightness_thresh
        self.spread_thresh = spread_thresh

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_array = np.array(img, dtype=np.float32)
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

        # Detect white-ish pixels: high average brightness + low color spread
        brightness = (r + g + b) / 3.0
        spread = np.max(img_array, axis=2) - np.min(img_array, axis=2)
        white_mask = (brightness > self.brightness_thresh) & (spread < self.spread_thresh)

        # Turn white → yellow: keep red, slightly warm green, strongly reduce blue
        img_array[white_mask, 2] *= 0.3   # suppress blue
        img_array[white_mask, 1] *= 0.92  # slightly warm green

        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))


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

    # Data Augmentation & Normalization (with 48x48 resolution)
    # Aggressive 3D geometric augmentation to force STN to learn heavy deskewing
    train_transform = transforms.Compose([
        transforms.Resize((52, 52)),              # Resize slightly larger to allow small crops

        # EXTREME 3D camera-angle simulation
        transforms.RandomPerspective(distortion_scale=0.5, p=0.8, fill=128),

        # Combined extreme geometric: rotation + translation + zoom + shear
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=30,                           # Tilt the camera ±30°
                translate=(0.15, 0.15),               # Shift up to 15% of image size
                scale=(0.8, 1.2),                     # Zoom in/out heavily
                shear=(-20, 20, -20, 20),             # Extreme Oblique viewing
                fill=128,                             # Neutral gray fill
            )
        ], p=0.8),

        transforms.RandomCrop((48, 48)),          # Crop back to 48x48 AFTER geometric transforms

        # Domain bridging: GTSRB white signs → Polish yellow signs
        RandomWhiteToYellow(p=0.3),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669])
    ])

    print("Loading GTSRB dataset via torchvision...")
    try:
        gtsrb_dataset = GTSRB(root=args.data_dir, split='train', download=True, transform=None)
    except Exception as e:
        print(f"Error downloading GTSRB: {e}")
        return

    print("Loading CIFAR-10 to use as Real-World Background (Class 43)...")
    cifar_dataset = CIFAR10(root=args.data_dir, train=True, download=True, transform=None)
    bg_dataset = BackgroundDataset(cifar_dataset)
    noise_dataset = NoiseBackgroundDataset(num_samples=10000)

    # Combine everything: 43 sign classes + 1 massive background class
    full_dataset = ConcatDataset([gtsrb_dataset, bg_dataset, noise_dataset])

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(range(len(full_dataset)), [train_size, val_size])

    train_dataset = SubsetWithTransform(torch.utils.data.Subset(full_dataset, train_indices), transform=train_transform)
    val_dataset = SubsetWithTransform(torch.utils.data.Subset(full_dataset, val_indices), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Training set size: {len(train_dataset)} (including Backgrounds)")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Using device: {device}")

    model = STN_CNN(num_classes=44).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # Compute Class Weights to fix the massive imbalance
    # GTSRB classes have ~600 images each. Class 43 has 60,000 images.
    # To mathematically balance this, Class 43 needs a weight 100x smaller.
    class_weights = torch.ones(44).to(device)
    class_weights[43] = 0.01

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # STN parameters MUST have a tiny learning rate (1e-5) because the affine grid is highly sensitive.
    # At standard 1e-3 lr, the STN gradients explode/vanish and it collapses to Identity (no transformation).
    stn_params = list(model.localization.parameters()) + list(model.fc_loc.parameters())
    base_params = [p for n, p in model.named_parameters() if 'localization' not in n and 'fc_loc' not in n]

    optimizer = optim.AdamW([
        {'params': stn_params, 'lr': 1e-4},
        {'params': base_params}
    ], lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training Loop
    for epoch in range(args.epochs):
        # Prevent STN collapse by freezing it during the first 3 epochs (Warmup)
        if epoch < 3:
            if epoch == 0:
                print("\n[Epoch 1-3] STN is FROZEN (Identity). CNN is learning basic features...")
            for p in stn_params:
                p.requires_grad = False
        elif epoch == 3:
            print("\n[Epoch 4+] STN is now UNFROZEN. Learning to deskew and center...")
            for p in stn_params:
                p.requires_grad = True

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
        scheduler.step()

    print("Finished Training")
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
