import argparse
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from deepface import DeepFace
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_folder", type=str, required=True)
    parser.add_argument("--fake_folder", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    # Thao: Currently verify with 4 images only
    real_images = glob.glob(os.path.join(args.real_folder, "*.png"))[:4]
    
    settings = os.listdir(args.fake_folder)
    data = {}
    save_dict = []
    # breakpoint()
    # Iterate over each setting in the fake folder
    for setting in tqdm(settings):
        # breakpoint()
        fake_images = glob.glob(os.path.join(args.fake_folder, setting, "*.png"))
        scores = []
        for fake_image in tqdm(fake_images):
            avg_score = []
            for real_image in real_images:
                try:
                    # Perform DeepFace verification
                    result = DeepFace.verify(img1_path=real_image, img2_path=fake_image)
                    scores.append(result["distance"])
                    avg_score.append(1-result["distance"])
                    # Generate the image hash or use any identifier (optional, placeholder here)
                    image_hash = os.path.basename(fake_image).split(".")[0]
                    
                    # Generate save location or use the fake image path
                    save_location = fake_image
                    
                    # Append data in the requested format
                except Exception as e:
                    print(e)
            save_dict.append(
                {
                    "image": image_hash,  # Placeholder for image hash
                    "image_path": save_location,  # Save path of the fake image
                    # "clip_score": result["distance"],  # Using distance as clip_score equivalent
                    # "real_path": real_image,  # Real image path
                    # "fake_path": fake_image,  # Fake image path
                    "distance": np.mean(avg_score),  # Distance between real and fake image
                    "verified": result["verified"]  # Verification status
                }
            )
        data[setting] = 1 - np.mean(scores)
    
    # Save detailed results into face_verification.json
    with open(f"{args.fake_folder}/face_scores.json", "w") as f:
        json.dump(save_dict, f)
    
    # Save overall similarity scores into overall_scores.json
    with open(f"{args.fake_folder}/overall_scores.json", "w") as f:
        json.dump(data, f)
    
    print('Saved at', f"{args.fake_folder}/face_verification.json")
    
    # Sort data by token length and create plot
    data = {k: v for k, v in data.items() if k.isdigit()}
    sorted_data = dict(sorted(data.items(), key=lambda item: int(item[0])))

    # Extracting keys and values from the sorted dictionary
    keys = list(sorted_data.keys())
    values = list(sorted_data.values())

    # Creating a line plot for similarity scores
    plt.figure(figsize=(10, 6))
    plt.plot(keys, values, marker='o', linestyle='-', color='b')

    # Adding labels and title for plot
    plt.xlabel('Token Length')
    plt.ylabel('Facial Similarity')
    plt.title('Facial Similarity by Token Length')
    plt.xticks(keys)  # To show each key as an x-tick
    plt.grid(True, axis='y')

    # Save the plot as an image
    plt.savefig(f"{args.fake_folder}/face_verification.png")
    print(f"Saved at {args.fake_folder}/face_verification.png")

    print(sorted_data)
