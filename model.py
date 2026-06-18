import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# --- Target Architecture: STN + 4-Layer CNN (48x48 input, high capacity) ---
class STN_CNN(nn.Module):
    def __init__(self, num_classes=44):
        super(STN_CNN, self).__init__()

        # Spatial Transformer Network (STN) Localization Network
        # Input: 48x48
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),      # -> 42x42
            nn.MaxPool2d(2, stride=2),          # -> 21x21
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),     # -> 17x17
            nn.MaxPool2d(2, stride=2),          # -> 8x8
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 8 * 8, 32), # Adjusted for 48x48 input
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # Main 4-Layer CNN Classification Network
        # Massive Capacity Upgrade: 128 -> 256 channels
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)    # -> 128 x 48x48
        self.bn1   = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)   # -> 128 x 48x48
        self.bn2   = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)                            # -> 128 x 24x24

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # -> 256 x 24x24
        self.bn3   = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # -> 256 x 24x24
        self.bn4   = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)                            # -> 256 x 12x12
        self.pool3 = nn.MaxPool2d(2, 2)                            # -> 256 x  6x6

        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def stn(self, x):
        # Extract features for localization
        xs = self.localization(x)
        xs = xs.reshape(-1, 10 * 8 * 8) # Adjusted for 48x48 input

        # Calculate affine transformation parameters
        theta = self.fc_loc(xs)
        theta = theta.reshape(-1, 2, 3)

        # Create affine grid and sample
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def forward(self, x):
        # Apply STN
        x = self.stn(x)

        # Pass through CNN with BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.pool3(x)

        x = x.reshape(-1, 256 * 6 * 6)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --- Helper for applying transforms to subsets ---
class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
