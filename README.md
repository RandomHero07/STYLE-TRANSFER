# Real-Time Style Transfer using PyTorch

This project performs **artistic style transfer** on both **images** and **videos** using a fast **feed-forward convolutional neural network** trained with **PyTorch**. It uses **VGG19** to extract deep content and style features and allows you to stylize any image or video with a custom style.

---

## Requirements

Install the necessary packages using `pip`:

```bash
pip install torch torchvision pillow opencv-python tqdm
```
If you want to use getimages.py to download a dataset of 1000 training images automatically:
```bash
pip install fiftyone
```
## Quick Start Guide

### Step 1: Prepare Training Data

- Place all your training images (JPEG/PNG) into a folder named `data/`.
- Place your style image (e.g., `style.png`) in the root directory.
- Example: if you are using a hexagonal pattern, rename it to `hexagons.png` and update `STYLE_IMAGE` inside `model.py`.

---

### Step 2: Train the Style Model

Run the training script:

```bash
python model.py
```

#### You can modify inside model.py:
`EPOCHS` to control training duration.
`STYLE_IMAGE` to use a different style.
`SAVE_MODEL_PATH` to change the output filename for the trained model.
Training can take several minutes to hours, depending on your system. GPU is recommended for faster training.
After training completes, your model will be saved as 
```bash
style_model.pth
```
### Step 3: Stylize an Image
To apply the trained style to an image:
```bash
python output.py
```
#### Inside output.py:
Change `input_img = "your_image.png"` to the image you want to stylize.
Output will be saved as `output.png` by default.
Make sure `style_model.pth` is in the same directory.

### Step 4: Stylize a Video

To apply style transfer to a video, run:

```bash
python video.py
```
#### Inside video.py:

Modify the line:
`input_video = "your_video.mp4"`
to your actual input video path.
You can also change the output filename, for example:
`output_video = "styled_video.mp4"`
Every frame is resized to 256x256 and stylized individually.
Note: Video stylization is slower and depends on video length and system specifications.

### Notes

- Training uses **VGG19** to extract both content and style features.
- The model architecture is based on:  
  _"Perceptual Losses for Real-Time Style Transfer" by Johnson et al._
- You can use any clear and consistent style image by replacing `style.png`.
- Train multiple models for different styles and switch between them.

---

### Tips

- Use more training epochs (e.g., **50-100**) for better stylization results.
- Larger and more diverse datasets improve output quality.
- To apply different styles:
  - Train and save multiple models.
  - Load your chosen `.pth` model in `output.py` or `video.py`.






