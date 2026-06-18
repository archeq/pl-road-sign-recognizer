import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import argparse
import time

from model import STN_CNN

# Class Names definition
class_names = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
    'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
    'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
    'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
    'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road',
    'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
    'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only',
    'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
    'End of no passing', 'End of no passing by vehicles over 3.5 metric tons', 'Background'
]

# Pre-allocate normalization tensors (Global)
MEAN_TENSOR = None
STD_TENSOR = None

def init_tensors(device):
    global MEAN_TENSOR, STD_TENSOR
    MEAN_TENSOR = torch.tensor([0.3403, 0.3121, 0.3214]).view(1, 3, 1, 1).to(device)
    STD_TENSOR = torch.tensor([0.2724, 0.2608, 0.2669]).view(1, 3, 1, 1).to(device)

def non_max_suppression(boxes, confidences, threshold=0.1):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(confidences)
    pick = []
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
        intersection = w * h
        min_area = np.minimum(area[i], area[idxs[:last]])
        overlap = intersection / (min_area + 1e-6)
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))
    return pick

def run_edge_proposal(model, frame_rgb, device, conf_threshold=0.95):
    """Region Proposal mode using Edge Detection. Color-agnostic, fast, and handles colorless signs perfectly."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to close broken boundaries on signs
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)
    
    # Use RETR_LIST instead of RETR_EXTERNAL. 
    # If a sign is displayed on a phone screen, RETR_EXTERNAL only finds the phone's boundary 
    # and ignores the sign inside it! RETR_LIST finds all contours, internal and external.
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes, confidences, class_ids, stn_visuals = [], [], [], []
    batch_windows_np, batch_coords = [], []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800: continue # Sign must be reasonably sized
        
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Structural Filter 1: Aspect Ratio (signs are roughly square)
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.6 or aspect_ratio > 1.4: continue
            
        # Structural Filter 2: Fill Ratio
        # A solid circle in Canny gives a filled contour (because we dilate/close). 
        # Prevents random L-shaped or diagonal background edges from passing.
        fill_ratio = area / (w * h)
        if fill_ratio < 0.4: continue
            
        pad = int(0.1 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_rgb.shape[1], x + w + pad)
        y2 = min(frame_rgb.shape[0], y + h + pad)
        
        window = frame_rgb[y1:y2, x1:x2]
        window_resized = cv2.resize(window, (48, 48))
        
        batch_windows_np.append(window_resized)
        batch_coords.append((x1, y1, x2, y2))
        
    if len(batch_windows_np) > 0:
        batch_np = np.stack(batch_windows_np)
        batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous().float().to(device) / 255.0
        batch_tensor = (batch_tensor - MEAN_TENSOR) / STD_TENSOR
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = F.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            
            mask_conf = (max_probs > conf_threshold) & (preds != 43)
            if mask_conf.any():
                passed_indices = torch.nonzero(mask_conf).squeeze(1)
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
                    
    if len(boxes) > 0:
        boxes_arr = np.array(boxes)
        confs_arr = np.array(confidences)
        keep_idxs = non_max_suppression(boxes_arr, confs_arr, threshold=0.1)
        return [boxes[i] for i in keep_idxs], [confidences[i] for i in keep_idxs], [class_ids[i] for i in keep_idxs], [stn_visuals[i] for i in keep_idxs]
    else:
        return [], [], [], []

def run_sliding_window(model, frame_rgb, device, conf_threshold=0.85):
    """Dense Sliding Window Mode."""
    scale = 640.0 / frame_rgb.shape[1]
    small_w = int(frame_rgb.shape[1] * scale)
    small_h = int(frame_rgb.shape[0] * scale)
    small_frame = cv2.resize(frame_rgb, (small_w, small_h))
    
    gray_img = cv2.cvtColor(small_frame, cv2.COLOR_RGB2GRAY)
    window_sizes = [(64, 64), (96, 96)]
    boxes, confidences, class_ids, stn_visuals = [], [], [], []
    batch_windows_np, batch_coords = [], []

    def process_fast_batch():
        nonlocal batch_windows_np, batch_coords, boxes, confidences, class_ids, stn_visuals
        if len(batch_windows_np) == 0: return
            
        batch_np = np.stack(batch_windows_np)
        batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous().float().to(device) / 255.0
        batch_tensor = (batch_tensor - MEAN_TENSOR) / STD_TENSOR
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = F.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            
            mask_conf = (max_probs > conf_threshold) & (preds != 43)
            if mask_conf.any():
                passed_indices = torch.nonzero(mask_conf).squeeze(1)
                if passed_indices.dim() == 0:
                    passed_indices = passed_indices.unsqueeze(0)
                    
                stn_out_t = model.stn(batch_tensor[passed_indices])
                for i, idx in enumerate(passed_indices.cpu().numpy()):
                    orig_x1, orig_y1, orig_x2, orig_y2 = batch_coords[idx]
                    boxes.append((int(orig_x1 / scale), int(orig_y1 / scale), int(orig_x2 / scale), int(orig_y2 / scale)))
                    confidences.append(max_probs[idx].item())
                    class_ids.append(preds[idx].item())
                    
                    vis = stn_out_t[i].permute(1, 2, 0).cpu().numpy()
                    vis = vis * np.array([0.2724, 0.2608, 0.2669]) + np.array([0.3403, 0.3121, 0.3214])
                    vis = np.clip(vis, 0, 1)
                    vis = (vis * 255).astype(np.uint8)
                    stn_visuals.append(vis)
                    
        batch_windows_np.clear()
        batch_coords.clear()

    for (w, h) in window_sizes:
        stride = w // 2
        for y in range(0, small_h - h + 1, stride):
            for x in range(0, small_w - w + 1, stride):
                window_gray = gray_img[y:y+h, x:x+w]
                if window_gray.std() < 25.0:
                    continue
                    
                window = small_frame[y:y+h, x:x+w]
                window_resized = cv2.resize(window, (48, 48))
                
                batch_windows_np.append(window_resized)
                batch_coords.append((x, y, x + w, y + h))
                
                if len(batch_windows_np) >= 128:
                    process_fast_batch()
                    
    if len(batch_windows_np) > 0:
        process_fast_batch()

    if len(boxes) > 0:
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        keep_idxs = non_max_suppression(boxes, confidences, threshold=0.1)
        return boxes[keep_idxs], confidences[keep_idxs], [class_ids[i] for i in keep_idxs], [stn_visuals[i] for i in keep_idxs]
    else:
        return [], [], [], []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model.pt")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_tensors(device)
    
    model = STN_CNN(num_classes=44)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded {args.model_path}")
    except FileNotFoundError:
        print(f"Error: {args.model_path} not found.")
        return
        
    model.to(device)
    model.eval()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    mode = "edge"
    cv2.namedWindow("Dashboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dashboard", 1680, 720)
    
    print("Press 'm' to toggle mode between Edge Proposals and Sliding Window.")
    print("Press 'q' to quit.")
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if mode == "edge":
            boxes, confs, cids, stns = run_edge_proposal(model, frame_rgb, device, conf_threshold=0.90)
        else:
            boxes, confs, cids, stns = run_sliding_window(model, frame_rgb, device, conf_threshold=0.85)
            
        # Draw bounding boxes cleanly on raw frame
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Short label
            label = f"{class_names[cids[i]][:15]}.. ({confs[i]:.2f})"
            cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # UI SPLIT VIEW RENDERING
        # Left Panel: 1280x720 (Live Camera)
        # Right Panel: 400x720 (STN Dashboard)
        frame_resized = cv2.resize(frame, (1280, 720))
        dashboard = np.zeros((720, 1680, 3), dtype=np.uint8)
        dashboard[0:720, 0:1280] = frame_resized
        
        # Draw Sidebar Division and Header
        cv2.line(dashboard, (1280, 0), (1280, 720), (255, 255, 255), 2)
        cv2.putText(dashboard, "STN TRANSFORM GALLERY", (1300, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.line(dashboard, (1300, 50), (1660, 50), (100, 100, 100), 1)
        
        y_offset = 80
        for i, stn_img in enumerate(stns):
            if y_offset > 600: break # don't run off screen
            
            # Upscale 48x48 STN output to beautiful 100x100 thumbnail
            stn_bgr = cv2.cvtColor(stn_img, cv2.COLOR_RGB2BGR)
            stn_disp = cv2.resize(stn_bgr, (100, 100))
            
            # Place on Dashboard
            dashboard[y_offset:y_offset+100, 1300:1400] = stn_disp
            cv2.rectangle(dashboard, (1300, y_offset), (1400, y_offset+100), (0, 255, 0), 2)
            
            # Draw Data Next to it
            text = class_names[cids[i]]
            # Split long class names dynamically so they fit in the sidebar
            if len(text) > 22:
                cv2.putText(dashboard, text[:22]+"-", (1415, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(dashboard, text[22:], (1415, y_offset+55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(dashboard, f"Conf: {confs[i]:.3f}", (1415, y_offset+85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(dashboard, text, (1415, y_offset+45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(dashboard, f"Conf: {confs[i]:.3f}", (1415, y_offset+80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            y_offset += 130
                
        # Main View Telemetry
        fps = 1.0 / (time.time() - start_time + 1e-6)
        mode_text = "Edge-Guided Region Proposal" if mode == "edge" else "Dense Sliding Window"
        cv2.putText(dashboard, f"Algorithm: {mode_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        cv2.putText(dashboard, f"FPS: {fps:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(dashboard, "Press 'm' to toggle algorithm", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Dashboard", dashboard)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mode = "sliding" if mode == "edge" else "edge"
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
