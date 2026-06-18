# Polish Road-Sign Recognizer: A Lightweight STN-CNN Approach

This repository contains a deep learning pipeline designed to recognize road signs in real-time. It was trained on the German Traffic Sign Recognition Benchmark (GTSRB) and heavily engineered to operate reliably on Polish roads using classical computer vision heuristics, a Spatial Transformer Network (STN), and a shallow CNN.

## Problem & Approach
Our goal is to build a highly robust road-sign recognizer that works on unscaled, real-world high-resolution photos and live webcam feeds. Instead of relying on massive, computationally expensive object detectors (like YOLO), we explicitly decoupled localization from classification:

1. **High-Performance Sliding Window (Region Proposal)**: Instead of a massive YOLO network, we implemented a pure mathematical sliding window algorithm. To ensure real-time performance on a 1080p webcam feed, we use a custom Variance Pre-Filter (to instantly skip flat backgrounds like sky or asphalt) and a batched CNN execution strategy.
2. **Spatial Transformer Network (STN)**: Automatically learns to crop, rotate, and heavily deskew traffic signs before passing them to the classifier.
3. **CNN Classifier**: A 4-layer convolutional network that handles 43 sign classes plus a dedicated Background class.
4. **Feasibility**: By combining intelligent window filtering with a shallow network, the total parameter count is kept to ~5.8 Million—a fraction of the size of standard modern detectors, allowing for real-time inference on standard laptop CPUs.

## Training Innovations & The Domain Gap
We utilized the **GTSRB** (German Traffic Sign Recognition Benchmark) dataset, but deployed the model for Polish roads. We resolved several major machine learning challenges during training:

- **Bridging the Domain Gap (White to Yellow)**: Polish warning signs use a *yellow* interior, while GTSRB signs use a *white* interior. We wrote a custom mathematical augmentation (`RandomWhiteToYellow`) that dynamically shifts the chromaticity of bright white pixels toward warm yellow during training, forcing the CNN to become invariant to background color.
- **Extreme Geometric Augmentations**: We trained the STN against extreme 3D camera-angle simulations, shearing up to 20°, and random scale changes to force it to learn aggressive deskewing.
- **The "Background" Class (CIFAR-10)**: To prevent the CNN from hallucinating road signs out of trees and cars, we injected thousands of generic CIFAR-10 images into the dataset as a 44th "Background" class.
- **100x Loss Penalty**: Because the CIFAR-10 background images vastly outnumbered the actual road signs, the network initially collapsed into "always guessing background". We solved this mathematically by applying a hard 100x dynamic weight penalty to the Background class inside the Cross-Entropy loss function.
- **Differential STN Learning Rates**: To prevent the STN from collapsing to an Identity matrix, the STN parameters were restricted to a precise $1e^{-4}$ learning rate, while the base CNN was trained at $1e^{-3}$.

## Results
- **Validation Accuracy**: **98.69%** on the GTSRB validation set (including the CIFAR-10 background class).
- **Real-World Performance**: The model successfully identifies and completely deskews heavily angled signs held up to webcams in real-time, instantly isolating them from messy backgrounds via the HSV heuristic.

## How to Reproduce

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**:
   This will download the GTSRB dataset, load the CIFAR-10 backgrounds, train the STN-CNN, and save `model.pt`.
   ```bash
   python train.py --epochs 20 --batch_size 64
   ```

3. **Run the Demo**:
   To run the high-performance sliding-window pipeline on your webcam (1080p by default):
   ```bash
   python demo.py
   ```
   To run on a static image:
   ```bash
   python demo.py --image path/to/image.jpg
   ```
