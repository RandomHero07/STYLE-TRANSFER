import torch
from torchvision import transforms
from PIL import Image
from model import TransformNet  #  IMPORTING OUR MODEL,MAKE SURE THEY ARE IN SAME FOLDER

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CONFIG ---
MODEL_PATH = "hexa.pth"   # Trained model path
INPUT_IMAGE = "test.jpg"        # Your input image
OUTPUT_IMAGE = "output.jpg"      # Output styled image
IMAGE_SIZE = 256                 # Should match training size
# ---------------

# --- Transform -------------------------------------------------------------------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

# --- Load Model ------------------------------------------------------------------------------------------------------------------------------------
model = TransformNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- Load and Style Image --------------------------------------------------------------------------------------------------------------------------
image = Image.open(INPUT_IMAGE).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(image).cpu().squeeze(0)
    output = output.clamp(0, 255).div(255)
    output_img = transforms.ToPILImage()(output)

# --- Save Output ------------------------------------------------------------------------------------------------------------------------------------
output_img.save(OUTPUT_IMAGE)
print(f"🎉 Output saved as {OUTPUT_IMAGE}")
