import argparse
import urllib.request
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import STN_CNN

CLASSES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)', 9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield', 14: 'Stop',
    15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited', 17: 'No entry', 18: 'General caution',
    19: 'Dangerous curve to the left', 20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work', 26: 'Traffic signals',
    27: 'Pedestrians', 28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Beware of ice/snow',
    31: 'Wild animals crossing', 32: 'End of all speed and passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left',
    38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

def non_max_suppression(boxes, confidences, threshold):
    if len(boxes) == 0:
        return []
    pick = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(confidences)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # Use Intersection over Minimum Area (IoM) to strongly suppress
        # overlapping boxes of different scales (e.g. 64x64 inside 128x128).
        intersection = w * h
        min_area = np.minimum(area[i], area[idxs[:last]])
        overlap = intersection / min_area
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))
    
    # To strictly ensure we only detect the best sign (as requested), 
    # we can limit the output to the absolute best detection per image.
    # We will return just the top 1 index.
    if len(pick) > 0:
        # pick is already sorted by highest confidence (since idxs was sorted ascending, we take from the end)
        # So pick[0] is the highest confidence.
        return [pick[0]]
    return pick

def get_args():
    parser = argparse.ArgumentParser(description="Live demo for road sign recognition.")
    parser.add_argument("--model_path", type=str, default="model.pt", help="Path to the trained model checkpoint.")
    parser.add_argument("--width", type=int, default=640, help="Width of the webcam frame.")
    parser.add_argument("--height", type=int, default=480, help="Height of the webcam frame.")
    parser.add_argument("--url", type=str, default=None, help="URL of an image to test sliding window.")
    parser.add_argument("--image", type=str, default=None, help="Local image path to test sliding window.")
    parser.add_argument("--conf", type=float, default=0.999, help="Confidence threshold for sliding window detection.")
    return parser.parse_args()

def process_batch(model, device, batch_windows, batch_coords, conf_threshold, boxes, confidences, class_ids, stn_visuals, max_entropy=2.0):
    batch_tensor = torch.stack(batch_windows).to(device)
    with torch.no_grad():
        outputs = model(batch_tensor)
        probs = F.softmax(outputs, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
        
        # Calculate Entropy: -sum(p * log(p))
        # High entropy = uncertain model (flat distribution) = background
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        mask = (max_probs > conf_threshold) & (entropy < max_entropy)
        if mask.any():
            passed_indices = torch.nonzero(mask).squeeze(1)
            # Handle case where passed_indices is 0-d tensor
            if passed_indices.dim() == 0:
                passed_indices = passed_indices.unsqueeze(0)
                
            stn_out_t = model.stn(batch_tensor[passed_indices])
            
            for i, idx in enumerate(passed_indices.cpu().numpy()):
                boxes.append(batch_coords[idx])
                confidences.append(max_probs[idx].item())
                class_ids.append(preds[idx].item())
                
                vis = stn_out_t[i].permute(1, 2, 0).cpu().numpy()
                vis = vis * np.array([0.2724, 0.2608, 0.2669]) + np.array([0.3403, 0.3121, 0.3214])
                vis = np.clip(vis, 0, 1)
                vis = (vis * 255).astype(np.uint8)
                stn_visuals.append(vis)

def run_sliding_window(model, frame_rgb, device, val_transform, conf_threshold, max_entropy=2.0):
    height, width = frame_rgb.shape[:2]
    min_dim = min(height, width)
    
    base_sizes = [48, 64, 96, 128, 192, 256, 384, 512, 768]
    window_sizes = [s for s in base_sizes if s <= min_dim]
    if not window_sizes:
        window_sizes = [min_dim]

    boxes = []
    confidences = []
    class_ids = []
    stn_visuals = []

    model.eval()
    batch_windows = []
    batch_coords = []
    
    # Pre-calculate grayscale for fast variance check
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    
    for win_size in window_sizes:
        step_size = max(16, win_size // 4)
        for y in range(0, height - win_size + 1, step_size):
            for x in range(0, width - win_size + 1, step_size):
                window_gray = gray[y:y+win_size, x:x+win_size]
                
                # Fast variance check: skip flat regions (sky, road surface, walls)
                # Traffic signs have sharp edges and high variance
                if np.var(window_gray) < 500:
                    continue
                    
                window = frame_rgb[y:y+win_size, x:x+win_size]
                input_tensor = val_transform(window)
                batch_windows.append(input_tensor)
                batch_coords.append((x, y, x+win_size, y+win_size))
                
                if len(batch_windows) >= 128:
                    process_batch(model, device, batch_windows, batch_coords, conf_threshold, boxes, confidences, class_ids, stn_visuals, max_entropy)
                    batch_windows, batch_coords = [], []
                    
    if len(batch_windows) > 0:
        process_batch(model, device, batch_windows, batch_coords, conf_threshold, boxes, confidences, class_ids, stn_visuals, max_entropy)

    if len(boxes) > 0:
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        
        keep_idxs = non_max_suppression(boxes, confidences, threshold=0.1)
        return boxes[keep_idxs], confidences[keep_idxs], [class_ids[i] for i in keep_idxs], [stn_visuals[i] for i in keep_idxs]
    else:
        return [], [], [], []

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = STN_CNN(num_classes=43)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded model from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model not found at {args.model_path}. Please train the model first.")
        return

    model.to(device)
    model.eval()

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669])
    ])

    if args.url or args.image:
        print("Running Sliding-Window Detection on Image...")
        if args.url:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            with urllib.request.urlopen(args.url) as req:
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            frame = cv2.imread(args.image, cv2.IMREAD_COLOR)

        if frame is None:
            print("Error: Could not decode the image.")
            return
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        boxes, confs, cids, stns = run_sliding_window(model, frame_rgb, device, val_transform, args.conf)
        
        if len(boxes) == 1:
            print("Detected 1 sign.")
        else:
            print(f"Detected {len(boxes)} signs.")
            
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            class_name = CLASSES.get(cids[i], "Unknown")
            
            print(f" - Sign {i+1}: '{class_name}' (Confidence: {confs[i]:.2f}, Window Size: {w}x{h})")
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({confs[i]:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show STN output for each detected box
            stn_bgr = cv2.cvtColor(stns[i], cv2.COLOR_RGB2BGR)
            stn_bgr = cv2.resize(stn_bgr, (100, 100))
            cv2.imshow(f"STN Transformation {i+1} ({class_name})", stn_bgr)
            
        cv2.imshow("Detection Results", frame)
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
    else:
        # Webcam feed
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

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = val_transform(frame_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output.data, 1)
                class_id = predicted.item()
                class_name = CLASSES.get(class_id, "Unknown")

                stn_out_t = model.stn(input_tensor)
                stn_out_vis = stn_out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                stn_out_vis = stn_out_vis * np.array([0.2724, 0.2608, 0.2669]) + np.array([0.3403, 0.3121, 0.3214])
                stn_out_vis = np.clip(stn_out_vis, 0, 1)
                stn_out_vis = (stn_out_vis * 255).astype(np.uint8)
                stn_out_vis = cv2.resize(stn_out_vis, (100, 100))
                stn_out_vis_bgr = cv2.cvtColor(stn_out_vis, cv2.COLOR_RGB2BGR)

            cv2.putText(frame, f"Class: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            combined = np.zeros((args.height, args.width + 120, 3), dtype=np.uint8)
            # Ensure frame fits in combined in case dimensions differ slightly
            h, w = frame.shape[:2]
            combined[0:min(h, args.height), 0:min(w, args.width)] = frame[0:min(h, args.height), 0:min(w, args.width)]
            cv2.putText(combined, "STN:", (args.width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            combined[40:140, args.width + 10:args.width + 110] = stn_out_vis_bgr

            cv2.imshow("Road Sign Recognition", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
