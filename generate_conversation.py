import argparse
import glob
import os

import requests
import torch

from PIL import Image
from tqdm import tqdm
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor

human_questions = {
    "What is Riley hair color?",
    "What is Riley height?",
    "What is Riley skin tone?",
    "How would you describe Riley hairstyle?",
    "Does Riley wear glasses or any accessories?",
    "What style of clothing does Riley typically choose?",
    "Does Riley have any distinctive facial features?",
    "What is Riley overall build or physique?",
    "What is Riley general expression or demeanor?",
    "How would you describe Riley overall appearance?",
}

object_questions = {
    "What color is Riley?",
    "What shape does Riley have?",
    "What is the overall vibe of Riley?",
    "What material is Riley made of?",
    "What size is Riley?",
    "Does Riley have any patterns or markings?",
    "What type of object is Riley?",
    "Does Riley have any distinctive features or details?",
    "What's Riley general texture like?",
    "Can you describe Riley?",
}

def get_args():
	parser = argparse.ArgumentParser(description='Chameleon')
	parser.add_argument('--image_folder', type=str, default='./yollava-data/train/mam/', help='Path to image')
	# parser.add_argument('--prompt', type=str, default="What shape does <mam> has? <image>", help='Prompt')
	parser.add_argument('--model_id', type=str, default='./chameleon-hf/chameleon-7b', help='Model ID')
	parser.add_argument('--type', type=str, default='object', help='Object or Human')
	return parser.parse_args()

if __name__ == '__main__':
	args = get_args()
	model_id = args.model_id
	# image = Image.open(args.image)

	processor = ChameleonProcessor.from_pretrained(model_id)
	model = ChameleonForConditionalGeneration.from_pretrained(model_id, device_map="auto") 
	# prompt = "What color is the belt in this image?<image>"
	list_questions = object_questions if args.type == 'object' else human_questions
	list_imgs = glob.glob(os.path.join(args.image_folder, '*.png'))
	for image_path in tqdm(list_imgs):
		for question in list_questions:
			print('----------------------------')
			print(f'Processing {question} for {image_path}')
			image = Image.open(image_path)
			prompt = question + f"<image>."
			inputs = processor(question, image, return_tensors="pt").to(model.device)

			# autoregressively complete prompt
			output = model.generate(**inputs, max_new_tokens=1000, pad_token_id=processor.tokenizer.eos_token_id)
			# print(')
			print(processor.decode(output[0], skip_special_tokens=True))
			# breakpoint()

