# Polish (and German) Road-Sign Recognizer

This repository contains a deep learning pipeline designed to recognize road signs in real-time. It was trained on the German Traffic Sign Recognition Benchmark (GTSRB) and designed to fulfill the criteria of a practical, fast-training Computer Vision application.

## Problem & Approach
Our goal is to build a robust road-sign recognizer that works on real-world photos and live webcam feeds. Instead of using large, pre-trained models (like YOLO), we engineered a **custom Convolutional Neural Network (CNN) equipped with a Spatial Transformer Network (STN)**. 
- **STN**: Automatically learns to crop, rotate, and deskew traffic signs before passing them to the classifier.
- **CNN**: A lightweight 4-layer convolutional network that handles the 43-class classification.
- The total parameter count is kept below 240,000, allowing it to train end-to-end on a single GPU in under 2 minutes (and well under an hour on a standard CPU).

## Dataset
We utilized the **GTSRB** (German Traffic Sign Recognition Benchmark) dataset, containing 43 classes of traffic signs. To bridge the domain gap between pristine dataset crops and messy real-world webcam feeds, we heavily augmented the training data with:
- `RandomPerspective` (to mimic dashboard camera angles)
- `GaussianBlur` (to mimic motion blur)
- `ColorJitter` (to handle varying lighting and weather conditions)

## The "Wow Angle": Live Sliding Window Detection
To demonstrate the model in the real world without a bounding-box dataset, we implemented a pure **sliding-window detection pipeline**. 
Because GTSRB does not contain a "Background" class, naive sliding windows generate massive false positives on sky and trees. We solved this mathematically without adding new classes:
1. **Variance Filter**: Fast pre-processing step that skips windows with low pixel variance (flat sky, blank walls).
2. **Entropy Filter**: Analyzes the CNN's softmax output. If the probability distribution is "flat" (high Shannon entropy), it means the model is uncertain, and the window is rejected as background noise.

## Results & Error Analysis
- **Validation Accuracy**: ~95% on the GTSRB validation set.
- **Failure Modes / Domain Gap**: While the model works perfectly on German signs, testing it on live Polish signs via webcam revealed a critical domain gap. German Warning and Yield signs have a **white interior**, while Polish signs have a **yellow interior**. Because the CNN had never seen a yellow Yield sign, it confidently misclassified it (e.g., as a "No entry" sign). This highlights how brittle CNNs can be to specific color features when transferring across geographical domains.

## How to Reproduce

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**:
   This will download the GTSRB dataset, train the STN-CNN, and save `model.pt`.
   ```bash
   python train.py --epochs 20 --batch_size 64
   ```

3. **Run the Demo**:
   To run the sliding-window detector on an image:
   ```bash
   python demo.py --image path/to/image.jpg
   ```
   To run on a live internet URL:
   ```bash
   python demo.py --url "https://example.com/sign.jpg"
   ```
