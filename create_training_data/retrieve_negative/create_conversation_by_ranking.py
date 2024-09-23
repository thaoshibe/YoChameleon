import argparse
import glob
import json
import os

from tqdm import tqdm

def get_args():
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Create conversation by ranking CLIP similarity score")
	parser.add_argument("--input_folder", type=str, default='/mnt/localssd/code/data/minimam/', help="Path to the base image")
	parser.add_argument("--save_folder", type=str, default='/mnt/localssd/code/data/minimam/piat_retrieved', help="Path to the base image")
	parser.add_argument("--limit", type=int, default=150, help="Number of similar images to retrieve")
	parser.add_argument("--token_length", type=int, default=16, help="Token length")
	return parser.parse_args()
# Read the JSON file
def get_personalized_prompt(token_length=1):

	prefix_tokens = [f'<reserved{16301+i}>' for i in range(token_length)]
	personalized_tokens = [f'<reserved16300>']
	personalized_tokens.extend(prefix_tokens)
	sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}. A photo of <reserved16300>."
	return sks_prompt

def divide_list_into_k_parts(image_paths, k):
	# Calculate the approximate size of each part
	avg_size = len(image_paths) // k
	remainder = len(image_paths) % k

	# Start dividing the list
	parts = []
	start = 0
	for i in range(k):
		# Distribute the remainder across the first few parts
		part_size = avg_size + (1 if i < remainder else 0)
		parts.append(image_paths[start:start + part_size])
		start += part_size
	return parts

def duplicate_list_to_match_size(lst, k):
	# Repeat the list until it matches the size of k
	repeated_list = (lst * (k // len(lst) + 1))[:k]
	return repeated_list

if __name__ == "__main__":
	args = get_args()
	file_path = os.path.join(args.save_folder, 'scores.json')

	with open(file_path, 'r') as f:
		data = json.load(f)

	# Sort the list of dictionaries based on the clip_score
	sorted_data = sorted(data, key=lambda x: x['clip_score'])

	print(f'There are {len(sorted_data)} images in total~')
	image_paths = [d['image_path'] for d in sorted_data]
	# divided lists
	divided_lists = divide_list_into_k_parts(image_paths, args.token_length-1)

	# THAO: Currently use only 4 positive images for train
	real_images = glob.glob(os.path.join(args.input_folder, "*.png"))[:4]
	real_images = duplicate_list_to_match_size(real_images, len(divided_lists[0]))
	divided_lists.append(real_images)
	print(f'Plus {len(real_images)}')
	#--- Create conversation by ranking
	data = []
	for index, part in enumerate(tqdm(divided_lists)):
		sks_prompt = get_personalized_prompt(token_length=index+1)
		for image_path in part:
			conv = [
				{
				"from": "human",
				"value": f"{sks_prompt}"
				},
				{
				"from": "bot",
				"value": f"<image>"
				},
			]
			data.append({
				"image": [image_path],
				"conversations": conv
			})
	with open(f"{args.input_folder}/conversations.json", 'w') as f:
		json.dump(data, f)
	print(f"Saved conversation at: {args.input_folder}/conversations.json")