import requests
import torch

from PIL import Image
from transformers import ChameleonForCausalLM
from transformers import ChameleonProcessor

import argparse

def get_args():
	parser = argparse.ArgumentParser(description='Chameleon')
	parser.add_argument('--image', type=str, default='./chameleon/inference/examples/thao-bo.jpeg', help='Path to image')
	parser.add_argument('--prompt', type=str, default=" What is the color of Teddy? <image>", help='Prompt')
	parser.add_argument('--model_id', type=str, default='./chameleon-hf/chameleon-7b', help='Model ID')
	return parser.parse_args()

if __name__ == '__main__':
	args = get_args()
	model_id = args.model_id

	processor = ChameleonProcessor.from_pretrained(model_id)
	model = ChameleonForCausalLM.from_pretrained(model_id, device_map="auto")

	image = Image.open(args.image)
	# system_prompt = "What is the breed of the dog in this photo?"
	# prompt = system_prompt + args.prompt
	prompt = args.prompt
	print(prompt)
	inputs = processor(prompt, image, return_tensors="pt").to(model.device)

	# output = model.generate(**inputs, max_new_tokens=50)
	print('Generating...')
	output = model.generate(**inputs, max_new_tokens=2000)
	print(processor.decode(output[0], skip_special_tokens=True))
	# breakpoint()