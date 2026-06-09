import pytest
import cv2
import urllib.request
import numpy as np
import torch
from torchvision import transforms

from model import STN_CNN
from demo import run_sliding_window, CLASSES

# Populate this dictionary with URLs you want to test!
# If a list is empty, the test will automatically be skipped.
TEST_URLS = {
    'Speed limit (20km/h)': [],
    'Speed limit (30km/h)': [],
    'Speed limit (50km/h)': [],
    'Speed limit (60km/h)': [],
    'Speed limit (70km/h)': [],
    'Speed limit (80km/h)': [],
    'End of speed limit (80km/h)': [],
    'Speed limit (100km/h)': [],
    'Speed limit (120km/h)': [],
    'No passing': [],
    'No passing for vehicles over 3.5 metric tons': [],
    'Right-of-way at the next intersection': [],
    'Priority road': [],
    'Yield': [],
    'Stop': [],
    'No vehicles': [],
    'Vehicles over 3.5 metric tons prohibited': [],
    'No entry': [],
    'General caution': [],
    'Dangerous curve to the left': [],
    'Dangerous curve to the right': [],
    'Double curve': [],
    'Bumpy road': [],
    'Slippery road': [],
    'Road narrows on the right': [],
    'Road work': [],
    'Traffic signals': [],
    'Pedestrians': [],
    'Children crossing': [],
    'Bicycles crossing': [],
    'Beware of ice/snow': [],
    'Wild animals crossing': [],
    'End of all speed and passing limits': [],
    'Turn right ahead': [],
    'Turn left ahead': [],
    'Ahead only': [],
    'Go straight or right': [],
    'Go straight or left': [],
    'Keep right': [],
    'Keep left': [],
    'Roundabout mandatory': [],
    'End of no passing': [],
    'End of no passing by vehicles over 3.5 metric tons': []
}

# Generate parameter list for pytest
test_cases = []
for class_name, urls in TEST_URLS.items():
    if not urls:
        test_cases.append((class_name, None))
    else:
        for url in urls:
            test_cases.append((class_name, url))


@pytest.fixture(scope="session")
def model_setup():
    """Loads the model and transform pipeline once for all tests."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STN_CNN().to(device)
    
    try:
        model.load_state_dict(torch.load("model.pt", map_location=device, weights_only=True))
    except FileNotFoundError:
        pytest.fail("model.pt not found. Please train the model first.")
        
    model.eval()
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669])
    ])
    
    return model, device, val_transform


@pytest.mark.parametrize("expected_class, url", test_cases)
def test_detect_sign(model_setup, expected_class, url):
    if url is None:
        pytest.skip(f"No URLs provided for class: '{expected_class}'. Add URLs to TEST_URLS to enable.")
        
    model, device, val_transform = model_setup
    
    # Download and decode image
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    
    try:
        with urllib.request.urlopen(url) as req:
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        pytest.fail(f"Failed to download URL: {url}\nError: {e}")
        
    assert frame is not None, f"Failed to decode image from URL: {url}"
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run sliding window detection (using a strict confidence threshold to match demo.py)
    boxes, confs, cids, stns = run_sliding_window(model, frame_rgb, device, val_transform, conf_threshold=0.999)
    
    assert len(boxes) > 0, f"No signs detected with high confidence in image: {url}"
    
    # Check if the expected class matches what was detected
    detected_classes = [CLASSES.get(cid, "Unknown") for cid in cids]
    
    assert expected_class in detected_classes, (
        f"Model failed to detect '{expected_class}'. It instead detected: {detected_classes}."
    )
