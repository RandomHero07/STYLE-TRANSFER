import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import os
from PIL import Image
import numpy as np
from model import TransformNet  # Import the model class from model.py

# ========== CONFIG ==========
MODEL_PATH = "style_model.pth"           # Path to the saved model
INPUT_VIDEO_PATH = "input.mp4"           # Path to input video
OUTPUT_VIDEO_PATH = "output_styled.mp4"  # Path to save styled output
IMAGE_SIZE = 256                         # Resize each frame to this size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================

# ----- Preprocessing & Postprocessing --------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

def preprocess(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)
    return img

def postprocess(tensor):
    tensor = tensor.squeeze().cpu().clamp(0, 1)
    img = transforms.ToPILImage()(tensor)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ----- Load Model -------------------------------------------
model = TransformNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ----- Read Video ---------------------------------------
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))

# VideoWriter: size must match processed frame size
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (IMAGE_SIZE, IMAGE_SIZE))

print("Stylizing video frames...")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_frame = postprocess(output_tensor)

    out.write(output_frame)
    frame_count += 1
    print(f"\rProcessed frame {frame_count}", end='')

cap.release()
out.release()
print("\nVideo stylization complete!")
