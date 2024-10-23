import argparse
import cv2
import glob
import insightface
import json
import numpy as np
import os
import sklearn

from datetime import datetime

from scipy.spatial.distance import cosine
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Face verification using InsightFace")
    parser.add_argument("--fake_folder", type=str, default="/sensei-fs/users/thaon/code/generated_images/neg-16-4-v2/700",
                        help="Path to the folder containing fake images")
    parser.add_argument("--real_folder", type=str, default="/mnt/localssd/code/data/yollava-data/train/thao",
                        help="Path to the folder containing real images")
    parser.add_argument("--output_file", type=str, default="face_compare.json",
                        help="Path to the output JSON file")
    return parser.parse_args()

def prepare_model():
    # Initialize the face recognition model
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=-1)  # Set ctx_id to 0 for GPU, or -1 for CPU
    return model

# Function to compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    vec1 = vec1/np.linalg.norm(vec1)
    vec2 = vec2/np.linalg.norm(vec2)
    cosine_value = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cosine_value

def l2_distance(vec1, vec2):
    # normalization
    # vec1 = sklearn.preprocessing.normalize(vec1).reshape(1, -1)
    # vec2 = sklearn.preprocessing.normalize(vec2).reshape(1, -1)
    vec1 = vec1/np.linalg.norm(vec1)
    vec2 = vec2/np.linalg.norm(vec2)
    diff = np.subtract(vec1, vec2)
    dist = np.sum(np.square(diff))
    # print(dist)
    return dist

# Precompute real image embeddings
def precompute_real_embeddings(real_images, model):
    real_image_features = {}
    for real_image in real_images:
        try:
            img = cv2.imread(real_image)
            faces = model.get(img)
            if faces:
                real_image_features[real_image] = faces[0].embedding
            else:
                print(f"No face detected in real image: {real_image}")
        except Exception as e:
            print(f"Error processing real image {real_image}: {e}")
    return real_image_features

# Compare a fake image with precomputed real image embeddings
def compare_images(fake_image_path, real_image_features, model):
    # Read the fake image
    fake_image = cv2.imread(fake_image_path)
    fake_faces = model.get(fake_image)
    if not fake_faces:
        print(f"No face detected in fake image: {fake_image_path}")
        return None

    fake_embedding = fake_faces[0].embedding
    # breakpoint()
    avg_distances = []
    for real_img_path, real_embedding in real_image_features.items():
        try:
            # Calculate cosine similarity between the fake and real image embeddings
            distance = cosine_similarity(fake_embedding, real_embedding)
            # distance = l2_distance(fake_embedding, real_embedding)
            avg_distances.append(distance)
            # print(f"Distance between {fake_image_path} and {real_img_path}: {distance}")
        except Exception as e:
            print(f"Error processing {fake_image_path} with {real_img_path}: {e}")
    
    if len(avg_distances) == 0:
        return None
    else:
        avg_distance = sum(avg_distances) / len(avg_distances)
        return avg_distance

def process_image(fake_image, model, real_image_features):
    avg_dist = compare_images(fake_image, real_image_features, model)
    if avg_dist is None:
        return {'fake_image': fake_image, 'avg_distance': None}
    return {'fake_image': fake_image, 'avg_distance': avg_dist}

def face_verification(fake_folder, real_folder, output_file):
    model = prepare_model()

    # List of images
    fake_images = glob.glob(fake_folder + "/*.png")
    # TODO: Thao: Currently verify on 4 real images only
    real_images = glob.glob(real_folder + "/*.png")[:4]

    # Precompute the real image embeddings
    real_image_features = precompute_real_embeddings(real_images, model)

    # Sequential processing
    result_list = []
    no_face_count = 0
    for fake_image in tqdm(fake_images):
        rs = process_image(fake_image, model, real_image_features)
        result_list.append(rs)
        if rs['avg_distance'] is None:
            no_face_count += 1

    # Calculate the overall average distance of all images that have a valid distance
    valid_distances = [res['avg_distance'] for res in result_list if res['avg_distance'] is not None]
    if valid_distances:
        overall_avg_distance = sum(valid_distances) / len(valid_distances)
    else:
        overall_avg_distance = None

    with open(output_file, "a") as f:
        f.write('\n')
        
        # Get the current date and time
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Write the date
        f.write(f"==================================================\nEvaluation on date: {current_date}\n")
        
        # Write folder info and overall average distance
        f.write(f'''
            {fake_folder}
            and
            {real_folder}
            overall_avg_distance: {overall_avg_distance}
            no_face_count: {no_face_count}
            total_images: {len(fake_images)}
        ==================================================
            ''')
    with open(output_file.replace('.txt', '.json'), "a") as f:
        # Prepare the output data as JSON string and add to log file
        output_data = {
            'results': result_list,
            'overall_avg_distance': overall_avg_distance,
            'no_face_count': no_face_count,
            'total_images': len(fake_images)
        }
        
        # Write the output data in a pretty-printed JSON format
        f.write(json.dumps(output_data, indent=4) + '\n')
    return output_file

if __name__ == "__main__":
    args = get_args()
    face_verification(args.fake_folder, args.real_folder, args.output_file)

