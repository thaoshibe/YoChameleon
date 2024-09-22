import argparse
import json
import numpy as np
import os
import random
import torch

from PIL import Image
from PIL import ImageFilter
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from tqdm import tqdm

# Setup argparse
parser = argparse.ArgumentParser(description="Inpainting with Stable Diffusion XL")
parser.add_argument('--image_folder', type=str, required=True, help='Folder containing input images')
parser.add_argument('--mask_folder', type=str, required=True, help='Folder containing input masks')
parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the output images')
args = parser.parse_args()

# Ensure output folder exists
os.makedirs(args.output_folder, exist_ok=True)

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
                                                 torch_dtype=torch.float16, variant="fp16").to("cuda")

with open('./system-prompts/background-captions.json', 'r') as file:
    prompts = json.load(file)
prompts = prompts['captions']

img_size = 512

for img_name in tqdm(os.listdir(args.image_folder)):
    # check if the file is an image
    if not img_name.endswith('.jpg') and not img_name.endswith('.png'):
        continue
    img_url = os.path.join(args.image_folder, img_name)
    mask_url = os.path.join(args.mask_folder, img_name)  # Assuming mask has the same name as image

    image = load_image(img_url).resize((img_size, img_size))
    fg_mask = load_image(mask_url).resize((img_size, img_size))
    bg_mask = (255 - (np.array(fg_mask) > 0) * 255).astype('uint8')
    bg_mask = Image.fromarray(bg_mask)
    bg_mask = bg_mask.filter(ImageFilter.GaussianBlur(radius=1))

    # prompt = "A field of sunflowers"
    
    prompt_idx = np.random.randint(0, len(prompts)-1)
    prompt = prompts[prompt_idx]
    # rd_seed = np.random.randint(0, 100000)
    # generator = torch.Generator(device="cuda").manual_seed(rd_seed)
    
    result_image = pipe(prompt=prompt, image=image, mask_image=bg_mask, 
                        guidance_scale=7.5, num_inference_steps=20, 
                        strength=1).images
    
    output_path = os.path.join(args.output_folder, img_name)
    result_image[0].save(output_path)
