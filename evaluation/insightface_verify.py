import cv2
import glob
import insightface
import numpy as np
import os

from scipy.spatial.distance import cosine
from tqdm import tqdm

# Initialize the face recognition model
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=-1)  # Set ctx_id to 0 for GPU, or -1 for CPU

# Function to compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def compare_images(fake_image, real_images):
    # Read the fake image
    fake_image = cv2.imread(fake_image)
    fake_img = model.get(fake_image)
    
    avg_distances = []
    for real_image in real_images:
        # Read each real image
        try:
          real_image = cv2.imread(real_image)
          real_img = model.get(real_image)
          
          # Calculate cosine similarity between the first face embedding
          distance = cosine_similarity(fake_img[0].embedding, real_img[0].embedding)
          print(distance)
          avg_distances.append(distance)  # Since cosine returns similarity, we use (1 - similarity)
        except Exception as e:
          print(f"Error processing {fake_image} with {real_image}: {e}")
    if len(avg_distances) == 0:
        return None
    else:
      avg_distance = sum(avg_distances) / len(avg_distances)
      return avg_distance

# fake_folder = '/sensei-fs/users/thaon/code/generated_images/neg-128-16/700'
fake_folder = '/sensei-fs/users/thaon/data/yollava-data/train/thao'
real_folder = '/sensei-fs/users/thaon/data/yollava-data/train/thao'

fake_images = glob.glob(fake_folder + "/*.png")
real_images = glob.glob(real_folder + "/*.png")[1:5]

# Function to process each fake image
def process_image(fake_image):
    avg_dist = compare_images(fake_image, real_images)
    if avg_dist is None:
        return {'fake_image': fake_image, 'avg_distance': None}
    return {'fake_image': fake_image, 'avg_distance': avg_dist}

# Sequential processing
result_list = []
for fake_image in tqdm(fake_images):
    rs = process_image(fake_image)
    result_list.append(rs)
    # print(result)
    # result_list.append(result)

# Output the result_list
print(result_list)
