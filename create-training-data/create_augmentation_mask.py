import argparse
import numpy as np
import os
import random
import torch

from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from tqdm import tqdm
from transformers import SamModel
from transformers import SamProcessor

# Check if CUDA is available and set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the SAM model and processor
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

def process_image_once(image_path):
    raw_image = Image.open(image_path).convert("RGB").resize((512, 512))
    input_points = [[[256, 256]]]  # Center of the image

    inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )

    scores = outputs.iou_scores
    selected_idx = torch.argmax(scores).item()
    
    # Convert mask to a PIL image
    mask = Image.fromarray((masks[0][0].numpy()[selected_idx] * 255).astype('uint8'))

    # Apply dilation to the mask
    mask = mask.filter(ImageFilter.MaxFilter(size=1))

    # Extract the foreground subject using the mask
    foreground = Image.composite(raw_image, Image.new("RGB", raw_image.size), mask)
    return foreground, mask

def inpaint_foreground_on_white_background(foreground, mask, raw_image, output_folder, image_basename, idx):
    # Create a white background
    background_image = Image.new("RGB", (512, 512), (255, 255, 255))
    background_mask = Image.new("RGB", (512, 512), (0,0,0))

    # Randomly resize the foreground subject
    new_size = random.randint(200, 512)
    foreground_resized = foreground.resize((new_size, new_size), Image.LANCZOS)
    mask_resized = mask.resize((new_size, new_size), Image.LANCZOS)
    raw_image_resized = raw_image.resize((new_size, new_size), Image.LANCZOS)  # Resize the whole image
    # breakpoint()
    # Calculate random position to place the foreground and the corresponding original image
    x = random.randint(0, background_image.size[0] - new_size)
    y = random.randint(0, background_image.size[1] - new_size)

    # Paste the resized original image onto the background
    background_image.paste(raw_image_resized, (x, y))
    background_mask.paste(mask_resized, (x, y))

    # Paste the resized foreground onto the background using the resized mask
    # background_image.paste(foreground_resized, (x, y), mask_resized)

    # Save the final image
    os.makedirs(os.path.join(output_folder, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'foreground'), exist_ok=True)
    output_image_path = os.path.join(output_folder, 'foreground', f"{image_basename}_{idx}.png")
    background_image.save(output_image_path)
    output_mask_path = os.path.join(output_folder, 'mask', f"{image_basename}_{idx}.png")
    background_mask.save(output_mask_path)

def main():
    parser = argparse.ArgumentParser(description="Process images with SAM model and generate multiple inpainted images.")
    parser.add_argument("--image_folder", type=str, help="Path to the folder containing images.")
    parser.add_argument("--output_folder", type=str, help="Path to the folder where output images will be saved.")
    parser.add_argument("--num_inpaintings_per_image", type=int, default=120, help="Number of inpainting images to generate per image.")

    args = parser.parse_args()

    image_folder = args.image_folder
    output_folder = args.output_folder
    num_inpaintings_per_image = args.num_inpaintings_per_image

    os.makedirs(output_folder, exist_ok=True)

    for filename in tqdm(os.listdir(image_folder)):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(image_folder, filename)
            foreground, mask = process_image_once(image_path)

            # Pass the raw_image along with other parameters
            raw_image = Image.open(image_path).convert("RGB").resize((512, 512))

            for i in range(num_inpaintings_per_image):
                inpaint_foreground_on_white_background(foreground, mask, raw_image, output_folder, os.path.splitext(filename)[0], i)


if __name__ == "__main__":
    main()
