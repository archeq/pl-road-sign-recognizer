import argparse
import urllib.request

import cv2
import numpy as np
import torch
from torchvision import transforms

from model import STN_CNN

# GTSRB Class Mapping
CLASSES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}


# --- 1. Argument Parsing ---
def get_args():
    parser = argparse.ArgumentParser(description="Live demo for road sign recognition.")
    parser.add_argument("--model_path", type=str, default="model.pt", help="Path to the trained model checkpoint.")
    parser.add_argument("--width", type=int, default=640, help="Width of the webcam frame.")
    parser.add_argument("--height", type=int, default=480, help="Height of the webcam frame.")
    parser.add_argument("--url", type=str, default=None, help="URL of an image to test instead of webcam.")
    return parser.parse_args()


# --- 2. Main Demo Logic ---
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = STN_CNN(num_classes=43)

    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded model from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model not found at {args.model_path}. Please train the model first.")
        return

    model.to(device)
    model.eval()

    # Define the same transformations as validation
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669])
    ])

    if args.url:
        print(f"Downloading image from {args.url}...")
        try:
            # --- FIX: Add a User-Agent header to mimic a browser ---
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            
            with urllib.request.urlopen(args.url) as req:
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                frame = cv2.imdecode(arr, -1) # 'Load it as it is'

            if frame is None:
                print("Error: Could not decode the image from the provided URL.")
                return
                
            # Convert BGR to RGB (OpenCV uses BGR, torchvision expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Preprocess the frame
            input_tensor = val_transform(frame_rgb).unsqueeze(0).to(device)

            # Get model prediction
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output.data, 1)
                class_id = predicted.item()
                class_name = CLASSES.get(class_id, "Unknown")

                # Get the STN transformed image
                stn_out_t = model.stn(input_tensor)
                stn_out_vis = stn_out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                # De-normalize for visualization
                stn_out_vis = stn_out_vis * np.array([0.2724, 0.2608, 0.2669]) + np.array([0.3403, 0.3121, 0.3214])
                stn_out_vis = np.clip(stn_out_vis, 0, 1)
                stn_out_vis = (stn_out_vis * 255).astype(np.uint8)
                stn_out_vis = cv2.resize(stn_out_vis, (200, 200)) # Make it bigger for static image

            print(f"\n--- Prediction ---")
            print(f"Class ID: {class_id}")
            print(f"Class Name: {class_name}")
            print(f"------------------\n")
            
            # Convert back to BGR for OpenCV display
            stn_out_vis_bgr = cv2.cvtColor(stn_out_vis, cv2.COLOR_RGB2BGR)

            cv2.imshow("Original Image", frame)
            cv2.imshow("STN Transformed (What the model 'sees')", stn_out_vis_bgr)
            print("Press any key in the image windows to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"An error occurred while processing the URL: {e}")
            
    else:
        # Start webcam feed
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Starting webcam feed. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Preprocess the frame
            input_tensor = val_transform(frame_rgb).unsqueeze(0).to(device)

            # Get model prediction
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output.data, 1)
                class_id = predicted.item()
                class_name = CLASSES.get(class_id, "Unknown")

                # Get the STN transformed image
                stn_out_t = model.stn(input_tensor)
                stn_out_vis = stn_out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                # De-normalize for visualization
                stn_out_vis = stn_out_vis * np.array([0.2724, 0.2608, 0.2669]) + np.array([0.3403, 0.3121, 0.3214])
                stn_out_vis = np.clip(stn_out_vis, 0, 1)
                stn_out_vis = (stn_out_vis * 255).astype(np.uint8)
                stn_out_vis = cv2.resize(stn_out_vis, (100, 100))
                
                # Convert back to BGR for OpenCV display
                stn_out_vis_bgr = cv2.cvtColor(stn_out_vis, cv2.COLOR_RGB2BGR)

            # Display the results
            cv2.putText(frame, f"Class: {class_name} #{class_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Create a combined view
            combined = np.zeros((args.height, args.width + 120, 3), dtype=np.uint8)
            combined[0:args.height, 0:args.width] = frame

            # Add STN visualization
            cv2.putText(combined, "STN Output:", (args.width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            combined[40:140, args.width + 10:args.width + 110] = stn_out_vis_bgr

            cv2.imshow("Road Sign Recognition", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
