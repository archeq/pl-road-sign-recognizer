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

def extract_rois_and_predict(model, frame_rgb, device, val_transform, conf_threshold):
    # Convert RGB to HSV
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    
    # Masks for Traffic Sign Colors (Red, Blue, Yellow)
    mask_red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    mask_blue = cv2.inRange(hsv, np.array([100, 100, 50]), np.array([130, 255, 255]))
    mask_yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
    
    mask = mask_red1 | mask_red2 | mask_blue | mask_yellow
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 400: # Minimum sign area
            x, y, w, h = cv2.boundingRect(c)
            # Aspect ratio check
            aspect_ratio = float(w)/h
            if 0.5 <= aspect_ratio <= 2.0:
                boxes.append([x, y, x+w, y+h])
                
    if not boxes:
        return [], [], [], []
        
    final_boxes, final_confs, final_cids, final_stns = [], [], [], []
    model.eval()
    
    for box in boxes:
        x1, y1, x2, y2 = box
        # Add 15% margin to capture edges for the STN
        margin_x = int((x2 - x1) * 0.15)
        margin_y = int((y2 - y1) * 0.15)
        
        cx1 = max(0, x1 - margin_x)
        cy1 = max(0, y1 - margin_y)
        cx2 = min(frame_rgb.shape[1], x2 + margin_x)
        cy2 = min(frame_rgb.shape[0], y2 + margin_y)
        
        roi_rgb = frame_rgb[cy1:cy2, cx1:cx2]
        if roi_rgb.size == 0: continue
            
        input_tensor = val_transform(roi_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            max_prob, pred = torch.max(probs, dim=1)
            
            if max_prob.item() >= conf_threshold:
                final_boxes.append([cx1, cy1, cx2, cy2])
                final_confs.append(max_prob.item())
                final_cids.append(pred.item())
                
                stn_out_t = model.stn(input_tensor)
                vis = stn_out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                vis = vis * np.array([0.2724, 0.2608, 0.2669]) + np.array([0.3403, 0.3121, 0.3214])
                vis = np.clip(vis, 0, 1)
                vis = (vis * 255).astype(np.uint8)
                final_stns.append(vis)
                
    if final_boxes:
        keep_idxs = non_max_suppression(np.array(final_boxes), np.array(final_confs), threshold=0.1)
        return (np.array(final_boxes)[keep_idxs], 
                np.array(final_confs)[keep_idxs], 
                [final_cids[i] for i in keep_idxs], 
                [final_stns[i] for i in keep_idxs])
    
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
        print("Running ROI Extraction & Detection on Image...")
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
        
        boxes, confs, cids, stns = extract_rois_and_predict(model, frame_rgb, device, val_transform, args.conf)
        
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
