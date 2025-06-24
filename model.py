import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from PIL import Image
import os
# Define the style transfer model architecture using convolution and transpose convolution layers
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)

class TransformNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 9, 1, 4),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),

            ResidualBlock(128),
            ResidualBlock(128),

            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 3, 9, 1, 4)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Set configuration: paths, image size, device, epochs, etc.
    IMAGE_DIR = "data"                    # this "data" folder should be oresent in the same folder as this script
    STYLE_IMAGE = "hexagons.png"          # similar to image directory this should also be present in same directory
    EPOCHS = 50
    BATCH_SIZE = 4
    IMAGE_SIZE = 256
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_MODEL_PATH = "style_model.pth"
# Image preprocessing: Resize and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
# Dataset class to load images from a flat directory
    class FlatImageDataset(Dataset):
        def __init__(self, image_dir, transform=None):
            self.image_paths = [os.path.join(image_dir, f)
                                for f in os.listdir(image_dir)
                                if f.lower().endswith(('.jpg', '.png'))]
            self.transform = transform
            self.loader = default_loader

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = self.loader(self.image_paths[idx])
            if self.transform:
                image = self.transform(image)
            return image, 0
# Initialize DataLoader with the dataset
    dataset = FlatImageDataset(IMAGE_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# Load and preprocess the style image
    style_img = Image.open(STYLE_IMAGE).convert("RGB")
    style_img = transform(style_img).unsqueeze(0).to(DEVICE)
# Load pretrained VGG19 model and extract features up to conv4_1
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:21].to(DEVICE).eval()
# Function to extract intermediate features from VGG19
    def extract_features(x, model):
        features = []
        for i, layer in enumerate(model):
            x = layer(x)
            if i in {0, 5, 10, 19}:
                features.append(x)
        return features
# Compute Gram matrix used for style representation
    def gram_matrix(tensor):
        b, c, h, w = tensor.size()
        features = tensor.view(b, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G / (c * h * w)

    style_features = extract_features(style_img, vgg)
    style_grams = [gram_matrix(f).detach() for f in style_features]
# Initialize model, optimizer, and MSE loss function
    model = TransformNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    total_images = len(loader.dataset)
    total_steps = total_images * EPOCHS
    step_count = 0
# Start training loop and set up progress tracking
    print("Training started...\n")

    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (content, _) in enumerate(loader):
            content = content.to(DEVICE)
            optimizer.zero_grad()

            output = model(content)
            content_feats = extract_features(content, vgg)
            output_feats = extract_features(output, vgg)

            content_loss = mse(output_feats[2], content_feats[2])
            style_loss = 0
            for of, sg in zip(output_feats, style_grams):
                gm = gram_matrix(of)
                style_loss += mse(gm, sg.expand_as(gm))

            loss = content_loss + 1e2 * style_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step_count += len(content)

            percent = step_count / total_steps * 100
            bar_len = 30
            done = int(bar_len * percent / 100)
            bar = "█" * done + '-' * (bar_len - done)
            # Update training progress bar in the terminal
            print(f"\rEpoch {epoch+1}/{EPOCHS} | [{bar}] {percent:.2f}% complete", end='')

        print(f"\n Epoch {epoch+1} finished | Avg Loss: {total_loss:.2f}\n")

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f" Model saved to: {SAVE_MODEL_PATH}")
    # Save the trained model weights to file
          
