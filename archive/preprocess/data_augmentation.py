import argparse
import os
import random
import torch

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Set up argument parser
parser = argparse.ArgumentParser(description="Image augmentation script.")
parser.add_argument('--image_folder', type=str, required=True, help="Path to the folder containing original images.")
parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder where augmented images will be saved.")
parser.add_argument('--num_augmented_images', type=int, default=500, help="Number of augmented images to create.")

args = parser.parse_args()

# Use the arguments from the command line
image_folder = args.image_folder
output_folder = args.output_folder
num_augmented_images = args.num_augmented_images

os.makedirs(output_folder, exist_ok=True)

# List of augmentation transforms to apply
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
])

# Load all the images from the folder
image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]

# Counter to keep track of saved images
counter = 0
# Generate augmented images
for i in tqdm(range(num_augmented_images)):
    # Randomly select an image from the original images
    img_path = random.choice(image_files)
    image = Image.open(img_path).convert('RGB')
    
    # Apply augmentation transforms
    augmented_image = augmentation_transforms(image)
    
    # Save the augmented image to the output folder
    output_path = os.path.join(output_folder, f'aug_{counter:04d}.png')
    augmented_image.save(output_path)
    counter += 1

print(f'Generated {counter} augmented images and saved them in {output_folder}.')