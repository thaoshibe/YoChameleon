import matplotlib.pyplot as plt
import os
import torch

from PIL import Image
from torch.nn.functional import cosine_similarity
from transformers import pipeline

# Define the folder path for fake images
fake_images_folder = "/mnt/localssd/YoLLaVA/example_training_data/bo/laion"

# Define the paths for the four real/reference images
real_image_paths = [
    "/mnt/localssd/code/data/minibo/7.png",
    "/mnt/localssd/code/data/minibo/8.png",
    "/mnt/localssd/code/data/minibo/9.png",
    "/mnt/localssd/code/data/minibo/6.png"
]

# Load the real images
real_images = [Image.open(path).convert("RGB") for path in real_image_paths]

# Initialize the pipeline for feature extraction
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipe = pipeline(task="image-feature-extraction", model="google/vit-base-patch16-384", device=DEVICE, pool=True)

# Extract features for the real images
real_features_list = [pipe(image_real) for image_real in real_images]

# Get all the fake images from the folder
fake_image_paths = [os.path.join(fake_images_folder, f) for f in os.listdir(fake_images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Initialize a list to store average similarity scores and corresponding image paths
scores_and_images = []

# Loop through each fake image and compute average similarity score with all reference images
for fake_image_path in fake_image_paths:
    # Load the fake image
    image_fake = Image.open(fake_image_path).convert("RGB")
    
    # Extract features for the fake image
    fake_features = pipe(image_fake)
    
    # Initialize a list to store the similarity scores with each reference image
    similarities = []
    
    # Compute the cosine similarity between fake image and each reference image
    for real_features in real_features_list:
        similarity_score = cosine_similarity(torch.Tensor(real_features[0]), torch.Tensor(fake_features[0]), dim=0).item()
        similarities.append(similarity_score)
    
    # Compute the average similarity score
    avg_similarity_score = sum(similarities) / len(similarities)
    
    # Store the average score and image path
    scores_and_images.append((avg_similarity_score, fake_image_path))
    breakpoint()
# Sort the images by average similarity score (high to low)
scores_and_images.sort(reverse=True, key=lambda x: x[0])
# print(scores_and_images)

# Visualization: Display images in a 10x10 grid in a zigzag pattern
fig, axes = plt.subplots(10, 10, figsize=(20, 20))
fig.tight_layout()

for i, (score, img_path) in enumerate(scores_and_images[:100]):
    print(img_path)
    row, col = divmod(i, 10)
    if row % 2 == 1:  # Zigzag: reverse row order
        col = 9 - col
    
    # Load the image
    img = Image.open(img_path).convert("RGB")
    
    # Display the image
    axes[row, col].imshow(img)
    axes[row, col].axis('off')
    axes[row, col].set_title(f"Score: {score:.2f}")

# Remove extra padding between plots
plt.savefig('test.png')
