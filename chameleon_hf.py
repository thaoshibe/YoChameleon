import argparse

import requests
import torch
from PIL import Image
from transformers import ChameleonForConditionalGeneration, ChameleonProcessor


def get_args():
	parser = argparse.ArgumentParser(description='Chameleon')
	parser.add_argument('--image', type=str, default='./chameleon/inference/examples/thao-bo.jpeg', help='Path to image')
	parser.add_argument('--prompt', type=str, default=" What is the color of Teddy? <image>", help='Prompt')
	parser.add_argument('--model_id', type=str, default='./chameleon-hf/chameleon-7b', help='Model ID')
	return parser.parse_args()

if __name__ == '__main__':
	args = get_args()
	model_id = args.model_id
	image = Image.open(args.image)

	processor = ChameleonProcessor.from_pretrained(model_id)
	model = ChameleonForConditionalGeneration.from_pretrained(model_id, device_map="auto") 
	prompt = "What color is the belt in this image?<image>"

	inputs = processor(prompt, image, return_tensors="pt").to(model.device)

	# autoregressively complete prompt
	output = model.generate(**inputs, max_new_tokens=50)
	print(processor.decode(output[0], skip_special_tokens=True))
