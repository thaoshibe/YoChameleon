import requests
import torch

import json

from PIL import Image
from tqdm import tqdm
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor

processor = ChameleonProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
model = ChameleonForConditionalGeneration.from_pretrained("leloy/Anole-7b-v0.1-hf", torch_dtype=torch.bfloat16, device_map="cuda")

####################
#
#
# This is for QA with Image Prompt
#
#
#####################
# with open('text_qa_soft_2.json') as f:
# 	data = json.load(f)

# total_count = 0
# count_correct = 0
# for data_key in tqdm(data.keys()):
# 	info = data[data_key]
# 	image = [Image.open(f'../../yochameleon-data/train/{data_key}/0.png'),
# 			 Image.open(f'../../yochameleon-data/train/{data_key}/1.png'),
# 			 Image.open(f'../../yochameleon-data/train/{data_key}/2.png'),
# 			 Image.open(f'../../yochameleon-data/train/{data_key}/3.png')]
# 	for index in info.keys():
# 		prompt = info[index]['question'].replace('sks', 'this subject')
# 		options = info[index]['option']
# 		option_A = options['A']
# 		option_B = options['B']
# 		question = prompt + f'{option_A} or {option_B}?'
# 		image_with_prompt = question +'<image><image><image><image>'
# 		correct_answer = options[info[index]['correct_answer']]
# 		inputs = processor(images=image, text=image_with_prompt, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
# 		output = model.generate(**inputs, max_new_tokens=10)
# 		answer = processor.decode(output[0], skip_special_tokens=True)
# 		answer = answer.replace(question, '')
# 		print(prompt)
# 		print(correct_answer)
# 		print(answer)
# 		if correct_answer.lower() in answer.lower():
# 			count_correct += 1
# 		total_count += 1
# 		print('Current accuracy:', count_correct/total_count)

####################
#
#
# This is for QA with Text Prompt
#
#
#####################

# with open('text_qa_soft_2.json') as f:
# 	data = json.load(f)

# with open('subject-detailed-captions.json') as f:
# 	captions = json.load(f)

# total_count = 0
# count_correct = 0
# for data_key in tqdm(data.keys()):
# 	try:
# 		info = data[data_key]
# 		caption = captions[data_key]
# 		caption = caption.replace('<sks>', 'this subject')
# 		for index in info.keys():
# 			prompt = info[index]['question'].replace('sks', 'this subject')
# 			options = info[index]['option']
# 			option_A = options['A']
# 			option_B = options['B']
# 			# question = caption + ' ' + prompt + f'{option_A} or {option_B}?'
# 			question = prompt + f'{option_A} or {option_B}?'
# 			correct_answer = options[info[index]['correct_answer']]
# 			inputs = processor(text=question, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
# 			output = model.generate(**inputs, max_new_tokens=10)
# 			answer = processor.decode(output[0], skip_special_tokens=True)
# 			answer = answer.replace(question, '')
# 			print(prompt)
# 			print(correct_answer)
# 			print(answer)
# 			if correct_answer.lower() in answer.lower():
# 				count_correct += 1
# 			total_count += 1
# 			print('Current accuracy:', count_correct/total_count)
# 	except:
# 		pass

####################
#
#
# This is for VQA with Text Prompt
#
#
#####################

# with open('yollava-visual-qa.json') as f:
# 	data = json.load(f)

# with open('subject-detailed-captions.json') as f:
# 	captions = json.load(f)

# total_count = 0
# count_correct = 0

# for data_key in tqdm(data.keys()):
# 	try:
# 		info = data[data_key]
# 		caption = captions[data_key]
# 		caption = caption.replace('<sks>', 'this subject')
# 		for index in info.keys():
# 			image = Image.open(index.replace('./yollava-data/', '../../yochameleon-data/'))
# 			prompt = info[index]['question'].replace('sks', 'this subject')
# 			options = info[index]['options']
# 			option_A = options['A']
# 			option_B = options['B']
# 			question = caption + ' ' + prompt + f'{option_A} or {option_B}?'

# 			correct_answer = options[info[index]['correct_answer']]
# 			inputs = processor(images = image, text=question, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
# 			output = model.generate(**inputs, max_new_tokens=10)
# 			answer = processor.decode(output[0], skip_special_tokens=True)
# 			answer = answer.replace(question, '')
# 			print(question)
# 			print(correct_answer)
# 			print(answer)
# 			if correct_answer.lower() in answer.lower():
# 				count_correct += 1
# 			total_count += 1
# 			print('Current accuracy:', count_correct/total_count)
# 	except:
# 		pass

with open('yollava-visual-qa.json') as f:
	data = json.load(f)

with open('subject-detailed-captions.json') as f:
	captions = json.load(f)

total_count = 0
count_correct = 0

for data_key in tqdm(data.keys()):
	try:
		info = data[data_key]
		caption = captions[data_key]
		caption = caption.replace('<sks>', 'this subject')
		for index in info.keys():
			image = [
				Image.open(index.replace('./yollava-data/test/', '../../yochameleon-data/train/')),
				Image.open(index.replace('./yollava-data/', '../../yochameleon-data/')),
				]
			prompt = info[index]['question'].replace('<sks>', 'this subject')
			options = info[index]['options']
			option_A = options['A']
			option_B = options['B']
			question = prompt + f'{option_A} or {option_B}?'
			question = 'You are seeing a photo of an subject <image>' + question +'<image>'
			correct_answer = options[info[index]['correct_answer']]
			inputs = processor(images = image, text=question, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
			output = model.generate(**inputs, max_new_tokens=10)
			print(output)
			# breakpoint()
			answer = processor.decode(output[0], skip_special_tokens=True)
			query = question.replace('<image>', '')
			answer = answer.replace(query, '')
			print(question)
			print(correct_answer)
			print(answer)
			if correct_answer.lower() in answer.lower():
				count_correct += 1
			total_count += 1
			print('Current accuracy:', count_correct/total_count)
	except Exception as e:
		print(e)
		pass