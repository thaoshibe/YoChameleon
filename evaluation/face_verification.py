# https://github.com/serengil/deepface
import argparse
import glob
import os

import json
import matplotlib.pyplot as plt
import numpy as np

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
	for setting in tqdm(settings):
	# for setting in ["2","8","16"]:
		fake_images = glob.glob(os.path.join(args.fake_folder, setting, "*.png"))
		scores = []
		for real_image in tqdm(real_images):
			for fake_image in tqdm(fake_images):
				try:
					result = DeepFace.verify(img1_path=real_image, img2_path = fake_image)
					scores.append(result["distance"])
					# print(f"Real: {real_image}, Fake: {fake_image}, Distance: {result['distance']}")
					save_dict.append(
						{"real_path": real_image,
						"fake_path": fake_image,
						"distance": result["distance"],
						"verified": result["verified"]}
						)
					# breakpoint()
				except Exception as e:
					print(e)
		data[setting] = 1-np.mean(scores)
	with open(f"{args.fake_folder}/face_verification.json", "w") as f:
		json.dump(save_dict, f)
	with open(f"{args.fake_folder}/overall_scores.json", "w") as f:
		json.dump(data, f)
	print('Saved at', f"{args.fake_folder}/face_verification.json")
	# with open("overall_scores.json", "r") as f:
	# 	data = json.load(f)

	data = {k: v for k, v in data.items() if k.isdigit()}
	sorted_data = dict(sorted(data.items(), key=lambda item: int(item[0])))

	# Extracting keys and values from the sorted dictionary
	keys = list(sorted_data.keys())
	values = list(sorted_data.values())

	# Creating a bar plot
	plt.figure(figsize=(10, 6))
	plt.plot(keys, values, marker='o', linestyle='-', color='b')

	# Adding labels and title for bar plot
	plt.xlabel('Token Length')
	plt.ylabel('Facial Similarity')
	plt.title('Facial Similarity by Token Length')
	plt.xticks(keys)  # To show each key as an x-tick
	plt.grid(True, axis='y')

	# Show the bar plot
	plt.savefig(f"{args.fake_folder}/face_verification.png")
	print(f"Saved at {args.fake_folder}/face_verification.png")

	print(sorted_data)