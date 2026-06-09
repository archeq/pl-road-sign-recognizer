# Polish Road-Sign Recogniser

A project to design, train, and demonstrate a working solution for a Polish Road-Sign Recogniser.

## 1. Problem

The goal of this project is to recognize Polish road signs from images. Due to the high visual similarity between Polish and German road signs, the German Traffic Sign Recognition Benchmark (GTSRB) dataset is used as a proxy for training and evaluation.

The project is developed with several constraints:
*   Training time under 1 hour on a single CPU/modest GPU.
*   Raw dataset size $\le 1\text{ GB}$.
*   Trained model checkpoint size $\le 100\text{ MB}$.
*   Training must be reproducible with fixed random seeds.

## 2. Approach

### Dataset

The [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/) is used. It contains ~50,000 images across 43 classes of road signs. Data augmentation techniques such as rotations, skews, and brightness adjustments are applied to improve model robustness. All input images are resized to a 48x48 resolution.

### Model Architecture

The model consists of a **Spatial Transformer Network (STN)** followed by a **4-layer Convolutional Neural Network (CNN)**.

*   **STN:** Learns an affine transformation (scaling, cropping, rotation) to normalize the input image and focus on the sign. This helps the model become invariant to transformations.
*   **CNN:** A 4-layer CNN with 3 MaxPool layers extracts features from the STN-transformed image for classification.

The total number of parameters is approximately 240,000.

### Optimizer

The model is trained using the **AdamW** optimizer, which is Adam with decoupled weight decay. This prevents the L2 regularization from interfering with the adaptive gradient scaling.

## 3. Results

*(This section will be updated after model training and evaluation.)*

Initial baseline models will be compared against the final STN+CNN architecture. Ablation studies will be performed to quantify the impact of the STN. An error analysis will document failure modes.

## 4. Reproduction Steps

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the model:**
    ```bash
    python train.py --epochs 20
    ```
    This will download the dataset, train the model, and save the checkpoint to `model.pt`.

3.  **Run the live webcam demo:**
    ```bash
    python demo.py
    ```
    This will launch a webcam feed.

4.  **Run the sliding window demo on a photo:**
    ```bash
    python demo.py --image path/to/image.jpg
    # or
    python demo.py --url http://example.com/image.jpg
    ```
    This will run sliding-window detection and display the detected signs along with their STN alignments.
