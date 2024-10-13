import argparse
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf

from deepface import DeepFace
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Facial similarity verification between real and fake images.")
    parser.add_argument("--real_folder", type=str, required=True, help="Path to the folder containing real images.")
    parser.add_argument("--fake_folder", type=str, required=True, help="Path to the folder containing fake images.")
    return parser.parse_args()

def main():
    args = get_args()

    # Limit real images to the first 4 for verification
    real_images = glob.glob(os.path.join(args.real_folder, "*.png"))[:4]

    fake_settings = os.listdir(args.fake_folder)
    verification_data = []
    overall_scores = {}

    # Iterate over each subfolder in the fake folder
    for setting in tqdm(fake_settings, desc="Processing settings"):
        fake_images = glob.glob(os.path.join(args.fake_folder, setting, "*.png"))
        distance_scores = []

        for fake_image in tqdm(fake_images, desc=f"Processing images in {setting}", leave=False):
            avg_distances = []
            for real_image in real_images:
                try:
                    import time
                    start_time = time.time()
                    # Perform DeepFace verification between real and fake image
                    result = DeepFace.verify(img1_path=real_image, img2_path=fake_image)
                    distance = result["distance"]
                    distance_scores.append(distance)
                    avg_distances.append(1 - distance)

                    # Store verification result
                    verification_data.append({
                        "image": os.path.basename(fake_image).split(".")[0],  # Image hash or identifier
                        "image_path": fake_image,
                        "distance": np.mean(avg_distances),  # Mean similarity score
                        "verified": result["verified"]  # Verification status
                    })
                    print(f"Processed {fake_image} with {real_image} in {time.time() - start_time} seconds")
                    breakpoint()
                except Exception as e:
                    print(f"Error processing {fake_image} with {real_image}: {e}")

        # Store the overall score (1 - mean distance) for the current setting
        if distance_scores:
            overall_scores[setting] = 1 - np.mean(distance_scores)

    # Save verification data and overall scores to JSON files
    save_json(verification_data, os.path.join(args.fake_folder, "face_scores.json"))
    save_json(overall_scores, os.path.join(args.fake_folder, "overall_scores.json"))

    # Plot similarity scores based on token length
    plot_similarity_scores(overall_scores, args.fake_folder)

def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f)
    print(f"Saved at {filepath}")

def plot_similarity_scores(data, output_folder):
    # Filter and sort data by keys that are numeric (token lengths)
    numeric_data = {k: v for k, v in data.items() if k.isdigit()}
    sorted_data = dict(sorted(numeric_data.items(), key=lambda item: int(item[0])))

    if not sorted_data:
        print("No valid numeric token length data to plot.")
        return

    # Plotting similarity scores
    keys = list(sorted_data.keys())
    values = list(sorted_data.values())

    plt.figure(figsize=(10, 6))
    plt.plot(keys, values, marker='o', linestyle='-', color='b')
    plt.xlabel('Token Length')
    plt.ylabel('Facial Similarity')
    plt.title('Facial Similarity by Token Length')
    plt.xticks(keys)
    plt.grid(True, axis='y')

    # Save the plot as an image
    plot_path = os.path.join(output_folder, "face_verification.png")
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free up memory
    print(f"Saved plot at {plot_path}")

if __name__ == "__main__":
    main()