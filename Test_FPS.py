import cv2
import time
import torch

# ==========================
# CONFIGURATION
# ==========================
VIDEO_PATH  = "test.mp4"                     # Input video path
OUTPUT_PATH = "output_ssd_cpu_fps.mp4"       # Output video path
MODEL_PATH  = "ssdlite320_mbv3_pruned_finetuned_full.pth"  # Loaded model file

device = torch.device("cpu")                 # Force CPU inference

# Class ID to name mapping
label_to_name = {
    1: 'vehicle',
    2: 'bus',
    3: 'car',
    4: 'motorbike',
    5: 'truck',
}

# ==========================
# LOAD MODEL
# ==========================
print("Loading model on CPU...")
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.to(device)
model.eval()
print("Model loaded on:", device)

# ==========================
# OPEN INPUT VIDEO
# ==========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps_in  = cap.get(cv2.CAP_PROP_FPS)
width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_in, (width, height))

print(f"Input video: {width}x{height} at {fps_in:.2f} fps")

# ==========================
# INFERENCE FUNCTION
# ==========================
@torch.no_grad()
def infer_one_frame(frame_bgr, score_thresh=0.5):
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Convert to tensor [C,H,W] normalized to [0,1]
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.to(device)

    # SSD model expects a list of tensors
    outputs = model([img_tensor])[0]

    boxes  = outputs["boxes"]
    labels = outputs["labels"]
    scores = outputs["scores"]

    # Filter by confidence score
    keep = scores >= score_thresh
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    frame_out = frame_bgr.copy()

    # Draw bounding boxes and labels
    for box, lab, sc in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cls_name = label_to_name.get(int(lab.item()), f"id{int(lab.item())}")
        text = f"{cls_name} {sc:.2f}"
