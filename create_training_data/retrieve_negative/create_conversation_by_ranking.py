import argparse
import glob
import json
import math
import os

from tqdm import tqdm

def get_args():
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Create conversation by ranking CLIP similarity score")
	parser.add_argument("--input_folder", type=str, default='/mnt/localssd/code/data/minimam/', help="Path to the base image")
	parser.add_argument("--save_folder", type=str, default='/mnt/localssd/code/data/minimam/piat_retrieved', help="Path to the base image")
	parser.add_argument("--token_length", type=int, default=16, help="Token length")
	parser.add_argument("--spacing", type=int, default=1, help="spacing")
	parser.add_argument("--num_of_real_images", type=int, default=-100, help="spacing")
	parser.add_argument("--version", type=str, default="v4")
	parser.add_argument("--negative_image", type=bool, default=False)
	parser.add_argument("--divide_before_positive", type=bool, default=False)
	parser.add_argument("--limit_negative", type=int, default=500)
	parser.add_argument("--consistent_prompt", type=bool, default=False)
	return parser.parse_args()

# Read the JSON file
def get_personalized_prompt(token_length=1, identifier=16200, index=16201):
	prefix_tokens = [f'<reserved{index+i}>' for i in range(token_length)]
	personalized_tokens = [f'<reserved{identifier}>']
	personalized_tokens.extend(prefix_tokens)
	if identifier == 16200:
		sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}. A photo of <reserved{identifier}>."
	else:
		sks_prompt = f"A photo of {''.join(personalized_tokens[1:])}."
	return sks_prompt

def divide_list_into_k_parts(image_paths, k):
	if k == 0:
		return [image_paths]

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
	# Thao: Uncomment this if you want to use the scores.json file
	if args.negative_image:
		file_path = os.path.join(args.input_folder, 'negative_example', 'scores.json')
		with open(file_path, 'r') as f:
			data = json.load(f)
		# print('Input Data: ', data)
		# Sort the list of dictionaries based on the clip_score
		sorted_data = sorted(data, key=lambda x: x['clip_score'])
		# Sort by face distance
		# filtered_data = [x for x in data if not math.isnan(x['distance'])]
		# sorted_data = sorted(filtered_data, key=lambda x: x['distance'])

		print(f'There are {len(sorted_data)} images in total~')
		image_paths = [d['image_path'] for d in sorted_data]
		image_paths = image_paths[:args.limit_negative]
		# divided lists
		num_of_part = int((args.token_length)/(args.spacing))-1
		print(f'Number of parts: {num_of_part}')
		divided_lists = divide_list_into_k_parts(image_paths, num_of_part)

		# # THAO: Currently use only 4 positive images for train
		real_images = glob.glob(os.path.join(args.input_folder, "*.png"))[:4]
		if args.num_of_real_images > 0:
			real_images = duplicate_list_to_match_size(real_images, args.num_of_real_images)
			divided_lists.append(real_images)
		elif args.num_of_real_images == 0:
			print('No added real images')
		else:
			real_images = duplicate_list_to_match_size(real_images, len(divided_lists[0]))
			divided_lists.append(real_images)
	else:
		real_images = glob.glob(os.path.join(args.input_folder, "*.png"))
		for ext in ['jpg', 'jpeg', 'jpeg', 'JPG', 'JPEG', 'JPEG']:
			real_images.extend(glob.glob(os.path.join(args.input_folder, f"*.{ext}")))
		real_images = real_images[:4] # THAO: Currently use only 4 positive
		real_images = duplicate_list_to_match_size(real_images, args.num_of_real_images)
		divided_lists = [real_images]

	print(f'Plus {len(real_images)} positive images')
	# print(len(divided_lists))

	#--- Create conversation by ranking
	data = []
	# divided_lists = [divided_lists[-1]]
	for index, part in enumerate(tqdm(divided_lists)):
		print(f'Length of part {index}: {len(part)}')

		# append all the part except index last one
		# flattened_list = [item for sublist in divided_lists[index:] for item in sublist]
		# part = flattened_list
		# for the idea of graudally added token
		# sks_prompt = get_personalized_prompt(token_length=args.spacing*(index+1))
		# for the idea of fixed token then finetune
		# sks_prompt = get_personalized_prompt(token_length=args.token_length)
		# --- for the idea of different identifier
		# sks_prompt = get_personalized_prompt(token_length=args.token_length, identifier=16200-len(divided_lists)+index+1)
		# This code will assigne different identifier for each chunk
		# sks_prompt = get_personalized_prompt(token_length=args.spacing*(index+1), identifier=16200-len(divided_lists)+index+1)
		# This code will simply use "A photo of ..." for different chunk
		if args.consistent_prompt:
			sks_prompt = get_personalized_prompt(token_length=args.spacing*(index+1), identifier=16200)
		else:
			sks_prompt = get_personalized_prompt(token_length=args.spacing*(index+1), identifier=16200-len(divided_lists)+index+1)
		print(f"Prompt: {sks_prompt}")
		print(part[0])
		# sks_prompt = 'A photo of <reserved16300>.'

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
	save_location = os.path.join(args.save_folder, f'{args.version}.json')
	with open(save_location, 'w') as f:
		json.dump(data, f)
	print(f"Saved conversation at: {save_location}")