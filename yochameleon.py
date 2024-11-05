import os
import torch
import wandb

import re

import numpy as np

import html

from PIL import Image
from evaluation.clip_image_similarity import CLIPEvaluator
from itertools import cycle
from tqdm import tqdm
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor
from transformers.image_transforms import to_pil_image
from utils import Config
from utils import chameleon_trim_answer

def save_generated_images(pixel_values, prompt_short, save_path, sks_name, index):
    """Save generated images to a specified directory."""
    for pixel_value in pixel_values:
        image = to_pil_image(pixel_value.detach().cpu())
        prompt_short = prompt_short.replace('<reserved16200>', sks_name).replace('.', '')
        os.makedirs(save_path, exist_ok=True)
        image.save(f'{save_path}/{prompt_short}_{index}.png')
        index += 1
    return index, image

class YoChameleonTrainer:
	def __init__(self, config):
		self.config = config
		self.get_model()
		self.prepare_personalized_tokens()
		self.get_optimizer_and_scheduler(config) # get optimizer and scheduler for pretraining
		self.setup_logger()
		self.sks_name = config.sks_name
		self.sks_prompt = f"{self.personalized_tokens[0]} is {''.join(self.personalized_tokens[1:])}."
		self.orig_embeds_params = self.model.get_input_embeddings().weight.data.clone()
		self.orig_lm_params = self.model.lm_head.weight.data.clone()
		self.index_no_updates = None
		self.iteration = 0
		self.clip_evaluator = CLIPEvaluator()
		self.weighted_acc = 0.0
		self.mean_clip = 0.0
		self.avg_metric = 0.0

	def get_personalized_prompt(self):
		return self.sks_prompt

	def get_understanding_prompt(self):
		if self.config.self_prompting:
			return self.understanding_prompt
		else:
			return None

	def get_generation_prompt(self):
		if self.config.self_prompting:
			return self.generation_prompt
		else:
			return None

	def prepare_personalized_tokens(self):
		if self.config.self_prompting:
			#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
			#          
			#  Attention: If follow this setting, prompt is: <sks> is <generation><understanding>
			#
			#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

			print('\n\n            Self-Prompting is enabled!\n\n')

			self.identifier = self.config.special_tokens["SKS_TOKEN"]
			identifier_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.identifier)

			self.latent_tokens_start_index = self.config.special_tokens["LATENT_TOKEN_START"]

			# generation tokens
			gen_prefix_tokens = [f'<reserved{self.latent_tokens_start_index+i}>' for i in range(self.config.prefix_token)]
			# understanding tokens
			understand_prefix_tokens = [f'<reserved{self.latent_tokens_start_index+self.config.prefix_token+i}>' for i in range(self.config.prefix_token)]
			personalized_tokens = [self.identifier]
			
			personalized_tokens.extend(gen_prefix_tokens)
			personalized_tokens.extend(understand_prefix_tokens)

			self.understanding_prompt = "".join(understand_prefix_tokens)
			self.generation_prompt = "".join(gen_prefix_tokens)

			self.personalized_tokens = personalized_tokens
			self.personalized_token_ids = self.processor.tokenizer.convert_tokens_to_ids(personalized_tokens)

			print(f'Personalized tokens: {self.personalized_tokens}')
			print(f'Personalized token ids: {self.personalized_token_ids}')
			print(f'There are {len(self.personalized_tokens)} personalized tokens')
		else:
			#--- This is train the SAME set of latent tokens for all the tasks
			self.latent_tokens_start_index = self.config.special_tokens["LATENT_TOKEN_START"]
			self.identifier = self.config.special_tokens["SKS_TOKEN"]

			prefix_tokens = [f'<reserved{self.latent_tokens_start_index+i}>' for i in range(self.config.prefix_token)]
			personalized_tokens = [self.identifier]
			personalized_tokens.extend(prefix_tokens)

			# --- This is for the negative identifier, which is not used anymore
			# if self.config.different_identifier:
			# 	# -1 for the identifier, then -1 for the first neagtive identifier
			# 	negative_identifier = [f'<reserved{self.latent_tokens_start_index-1-i}>' for i in range(1, self.config.prefix_token)]
			# 	personalized_tokens.extend(negative_identifier)
			# 	print(negative_identifier)
			# 	print(len(negative_identifier))

			self.personalized_tokens = personalized_tokens
			self.personalized_token_ids = self.processor.tokenizer.convert_tokens_to_ids(personalized_tokens)
			print(f'Personalized tokens: {self.personalized_tokens}')
			print(f'Personalized token ids: {self.personalized_token_ids}')
			print(f'There are {len(self.personalized_tokens)} personalized tokens')

	def get_model(self):
		self.processor = ChameleonProcessor.from_pretrained(self.config.model_id)
		self.model = ChameleonForConditionalGeneration.from_pretrained(self.config.model_id, device_map="auto", torch_dtype=torch.bfloat16)
		print(f'Loaded {self.config.model_id}!')
		# return processor, model

	def setup_logger(self):
		# This is for tensorboard, which is not used for this project anymore
		# log_dir = f'./runs/{self.config.exp_name}/{self.config.sks_name}'
		# if os.path.exists(log_dir):
		# shutil.rmtree(log_dir)
		# writer = SummaryWriter(f'./runs/{config.exp_name}/{config.sks_name}')
		self.save_location = f'{self.config.savedir}/{self.config.exp_name}/{self.config.sks_name}'
		os.makedirs(self.save_location, exist_ok=True)
		if not self.config.no_wandb:
			self.wandb = wandb.init(project=self.config.project_name,
				name=self.config.exp_name + '-' + self.config.sks_name,
				entity=self.config.entity,
				config=self.config)
			self.wandb.define_metric("eval")
			# Set all other metrics to use "eval" as the step metric
			self.wandb.define_metric("Recognition/*", step_metric="eval")
			self.wandb.define_metric("Metrics/*", step_metric="eval")
			self.wandb.define_metric("Image", step_metric="eval")
			self.wandb.define_metric("Text", step_metric="eval")
		else:
			self.wandb = None

	def get_optimizer_and_scheduler(self, config):
		try:
			config = Config(config)
		except:
			config = config # check if config is already a Config object
		optimizer_config = Config(config.optimizer)
		scheduler_config = Config(config.scheduler)
		if self.config.whole_model:
			trainable_params = self.model.model.parameters()
			optimizer = torch.optim.AdamW(
				trainable_params,
				lr=float(optimizer_config.lr),
				betas=tuple(optimizer_config.betas),
				weight_decay=float(optimizer_config.weight_decay),
				eps=float(optimizer_config.eps)
			)
		else:
			# train embedding weights and lm only
			trainable_params = [self.model.get_input_embeddings().weight, self.model.lm_head.weight]
			optimizer = torch.optim.AdamW(
				trainable_params,
				lr=float(optimizer_config.lr),
				betas=tuple(optimizer_config.betas),
				weight_decay=float(optimizer_config.weight_decay),
				eps=float(optimizer_config.eps),
			)
		if scheduler_config.type == 'StepLR':
			scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_config.step_size, gamma=scheduler_config.gamma)
		else:
			print('Scheduler not implemented yet')
			scheduler = None
		self.optimizer, self.scheduler, self.optimizer_config, self.scheduler_config = optimizer, scheduler, optimizer_config, scheduler_config
		# return optimizer, scheduler, optimizer_config, scheduler_config

	def save_checkpoint(self, iteration, finetune=False):
		# if type(iteration) == int:
		# 	iteration=iteration+1 # increment iteration to save the correct iteration as python starts from 0
		if finetune:
			save_path_token = os.path.join(self.save_location, f'{iteration}-token-ft.pt')
			save_path_lmhead = os.path.join(self.save_location, f'{iteration}-lmhead-ft.pt')
		else:
			save_path_token = os.path.join(self.save_location, f'{iteration}-token.pt')
			save_path_lmhead = os.path.join(self.save_location, f'{iteration}-lmhead.pt')
		torch.save(self.model.get_input_embeddings().weight.data[self.personalized_token_ids], save_path_token)
		print('Saved token embeddings at: ', save_path_token)

		if self.config.whole_model:
			if iteration > 1: # save whole model only after the first iteration to save space
				torch.save(self.model.model.state_dict(), os.path.join(self.save_location, f'{iteration}-model.pt'))
				print('Saved whole model at: ', os.path.join(self.save_location, f'{iteration}-model.pt'))
		else:
			torch.save(self.model.lm_head.weight.data[self.personalized_token_ids], save_path_lmhead)
			print('Saved lm_head at: ', save_path_lmhead)


	def load_prefix(self, config_resume, exp_name, resume_token_ids):
		lm_head_path = os.path.join(config_resume.savedir, exp_name, self.config.sks_name, f"{config_resume.resume_iteration}-lmhead.pt")
		embedding_path = os.path.join(config_resume.savedir, exp_name, self.config.sks_name, f"{config_resume.resume_iteration}-token.pt")
		# Load language model head
		lm_head = torch.load(lm_head_path, map_location='cuda').to(self.model.lm_head.weight.data.device)
		lm_head = lm_head.to(self.model.dtype)
		self.model.lm_head.weight.data[resume_token_ids] = lm_head

		# Load input embeddings
		embeddings = torch.load(embedding_path).to(self.model.device).to(self.model.dtype)
		self.model.get_input_embeddings().weight.data[resume_token_ids] = embeddings

		print('\n\n\n           ATTENTION -- PLEASE YOU CHECK IF THE RESUME IS CORRECT!\n\n\n')
		print(f'\n\n\n Resume tokens ids: {resume_token_ids} \n From: {exp_name} at epochs {config_resume.resume_iteration}\n\n\n')

	def load_prefix_mixture(self, config_resume, resume_token_ids):
		# import
		
		gen_lm_head_path = os.path.join(config_resume.savedir, config_resume.gen_exp_name, self.config.sks_name, f"{config_resume.resume_iteration}-lmhead.pt")
		gen_embedding_path = os.path.join(config_resume.savedir, config_resume.gen_exp_name, self.config.sks_name, f"{config_resume.resume_iteration}-token.pt")
		gen_lm_head = torch.load(gen_lm_head_path, map_location='cuda').to(self.model.lm_head.weight.data.device)
		gen_lm_head = gen_lm_head.to(self.model.dtype)
		gen_embeddings = torch.load(gen_embedding_path).to(self.model.device).to(self.model.dtype)

		understand_lm_head_path = os.path.join(config_resume.savedir, config_resume.understand_exp_name, self.config.sks_name, f"{config_resume.resume_iteration}-lmhead.pt")
		understand_embedding_path = os.path.join(config_resume.savedir, config_resume.understand_exp_name, self.config.sks_name, f"{config_resume.resume_iteration}-token.pt")
		understand_lm_head = torch.load(understand_lm_head_path, map_location='cuda').to(self.model.lm_head.weight.data.device)
		understand_lm_head = understand_lm_head.to(self.model.dtype)
		understand_embeddings = torch.load(understand_embedding_path).to(self.model.device).to(self.model.dtype)

		#--- Load the understand tokens
		self.model.lm_head.weight.data[self.personalized_token_ids] = understand_lm_head
		self.model.get_input_embeddings().weight.data[self.personalized_token_ids] = understand_embeddings
		# --- Load the generation tokens
		#
		#     Thao: Because for generation tokens, we skip <sks> token, then append generation tokens...
		#           Remind: The prompt format is: <sks> is <generation tokens><understanding tokens>.
		# ----
		self.model.lm_head.weight.data[self.generation_prefix_token_ids] = gen_lm_head[1:len(self.generation_prefix_token_ids)+1]
		self.model.get_input_embeddings().weight.data[self.generation_prefix_token_ids] = gen_embeddings[1:len(self.generation_prefix_token_ids)+1]
		print('\n\n\n           ATTENTION -- PLEASE YOU CHECK IF THE RESUME IS CORRECT!\n\n\n')
		print(f'\n\n\n Resume tokens ids: {resume_token_ids} \n')
		print(f'        Understanding from... : {config_resume.understand_exp_name} at epochs {config_resume.resume_iteration}')
		print(f'        Generation from... : {config_resume.gen_exp_name} at epochs {config_resume.resume_iteration}\n\n\n')

	def resume_training(self):
		try:
			if self.config.resume['resume']:
				print('Resuming training... from iteration:', self.config.resume['resume_iteration'])
				config_resume = Config(self.config.resume)
				# embedding_path = f'{config_resume.savedir}/{config_resume.exp_name}/{self.config.sks_name}/{config_resume.resume_iteration}-token.pt'
				try:
					if self.config.task_disjoin:
						self.load_prefix_mixture(config_resume, self.personalized_tokens)
					else: # no task disjoin -- just load from the saved personalized tokens
						self.load_prefix(config_resume, config.resume.exp_name, self.personalized_token_ids)
				except Exception as e:
					print(e)
					model_path = os.path.join(config_resume.savedir, config_resume.exp_name, self.config.sks_name, str(config_resume.resume_iteration) + '-model.pt')
					state_dict = torch.load(model_path)
					self.model.model.load_state_dict(state_dict)
					print(f'\n\n\n           Resumed model from {model_path} \n\n\n')
				self.iteration = config_resume.resume_iteration
			else:
				print('Starting training from scratch...')
		except Exception as e:
			print(e)
			print('\n\n\n       The config said I should load from the resume, but I could not find the resume config')
			print('       Also, check the above error... \n\n\n')
			exit()

	def configure_model(self):
		if self.config.whole_model:
			self.model.model.requires_grad_(True)
			self.model.model.embed_tokens.weight.requires_grad_(True)
			self.model.model.vqmodel.requires_grad_(False)
			self.index_no_updates = torch.zeros((len(self.processor.tokenizer),), dtype=torch.bool)
		else:
			if self.config.task_disjoin:
				self.model.model.requires_grad_(False)
				self.model.model.embed_tokens.weight.requires_grad_(True)
				self.index_no_updates_understand = torch.ones((len(self.processor.tokenizer),), dtype=torch.bool)
				self.index_no_updates_understand[self.understand_prefix_token_ids] = False

				self.index_no_updates_generation = torch.ones((len(self.processor.tokenizer),), dtype=torch.bool)
				self.index_no_updates_generation[self.generation_prefix_token_ids] = False
			else:
				self.model.model.requires_grad_(False)
				self.model.model.embed_tokens.weight.requires_grad_(True)
				self.index_no_updates = torch.ones((len(self.processor.tokenizer),), dtype=torch.bool)
				self.index_no_updates[self.personalized_token_ids] = False

	def train_epoch(self, dataloader, recognition_data_loader_train=None, recognition_data_loader_test=None):
		if not self.config.no_wandb:
			self.wandb.log({"Dataset/Train_dataset_length": len(dataloader.dataset)})
		if self.config.eval['clip_sim']:
			real_images_path = [x for x in sorted(recognition_data_loader_train.image_paths) if self.sks_name in x]
			real_images = [Image.open(x).convert("RGB") for x in real_images_path]
		for iteration in tqdm(range(self.config.iteration+1)):
			# Save model checkpoints
			eval_list = []
			if iteration % self.config.save_every == 0:
				self.save_checkpoint(iteration)
				if self.config.eval_visualization:
					visual_dict = self.visualize_evaluation()
					eval_list.append(visual_dict)
				if self.config.eval['recognition']:
					train_recog = self.eval_recognition(recognition_data_loader_train, split='train')
					test_recog = self.eval_recognition(recognition_data_loader_test, split='test')
					eval_list.append(train_recog)
					eval_list.append(test_recog)
				if self.config.eval['clip_sim']:
					clip_sim = self.eval_clip_similarity(real_images, number_fake_images=self.config.eval['number_fake_images'])
					eval_list.append(clip_sim)
				avg_score = clip_sim['Metrics/CLIP'] + train_recog['Metrics/train_weighted_accuracy']
				if self.avg_metric <= avg_score:
					self.avg_metric = avg_score
					self.save_checkpoint('best')
				if not self.config.no_wandb:
					log_dict = {**{"eval": iteration}}
					for item in eval_list:
						log_dict.update(item)
					self.wandb.log(log_dict)
			for batch in tqdm(dataloader):
				self.optimizer.zero_grad()
				batch['pixel_values'] = batch['pixel_values'].to(self.model.dtype)

				# Process labels with image tokens
				for i, item in enumerate(batch['labels']):
					if len(torch.nonzero(batch['labels'][i] == self.config.special_tokens["START_OF_IMAGE_INDEX"])) != 0:
						soi_index = torch.nonzero(batch['labels'][i] == self.config.special_tokens["START_OF_IMAGE_INDEX"]).item() + 1
						eot_index = torch.nonzero(batch['labels'][i] == self.config.special_tokens["END_OF_IMAGE_INDEX"]).item()
						image_tokens = self.model.model.get_image_tokens(pixel_values=batch['pixel_values'][None, i])[0]
						batch['labels'][i, soi_index:eot_index] = image_tokens
				for i, item in enumerate(batch['input_ids']):
					if len(torch.nonzero(batch['input_ids'][i] == self.config.special_tokens["START_OF_IMAGE_INDEX"])) != 0:
						soi_index = torch.nonzero(batch['input_ids'][i] == self.config.special_tokens["START_OF_IMAGE_INDEX"]).item() + 1
						eot_index = torch.nonzero(batch['input_ids'][i] == self.config.special_tokens["END_OF_IMAGE_INDEX"]).item()
						image_tokens = self.model.model.get_image_tokens(pixel_values=batch['pixel_values'][None, i])[0]
						batch['input_ids'][i, soi_index:eot_index] = image_tokens
						# print('image tokens added to input_ids')
				batch = {k: v.to(self.model.device) for k, v in batch.items()}

				# Forward pass
				output = self.model(
					input_ids=batch['input_ids'],
					# pixel_values=batch['pixel_values'],
					attention_mask=batch['attention_mask'],
					labels=batch['labels']
				)
				loss = output.loss
				loss.backward()
				self.optimizer.step()
				if self.scheduler is not None:
					self.scheduler.step()

				# Gradient clipping
				if self.optimizer_config.grad_clip > 0:
				    torch.nn.utils.clip_grad_value_(self.model.model.parameters(), clip_value=self.optimizer_config.grad_clip)

				# Revert embeddings if not training the whole model
				if not self.config.whole_model:
					with torch.no_grad():
						self.model.get_input_embeddings().weight[self.index_no_updates] = self.orig_embeds_params[self.index_no_updates]
						self.model.lm_head.weight[self.index_no_updates] = self.orig_lm_params[self.index_no_updates]

				# Log loss to W&B
				if not self.config.no_wandb:
				    self.wandb.log({"loss": loss.item()})
			torch.cuda.empty_cache()
			self.iteration = iteration

	@torch.no_grad()
	def test(self):
		config_test = Config(self.config.test)
		index = 0
		for i in tqdm(range(0, config_test.num_images, config_test.batch_size)):  # Step through by batch size
			prompt_short = config_test.prompt
			full_prompt = f"{self.sks_prompt} {prompt_short}"
			inputs = self.processor([full_prompt] * config_test.batch_size, return_tensors="pt").to(self.model.device)
			generate_ids = self.model.generate(**inputs, multimodal_generation_mode="image-only", max_new_tokens=1026, do_sample=True)
			response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
			pixel_values = self.model.decode_image_tokens(response_ids[:, 1:-1])
			pixel_values = self.processor.postprocess_pixel_values(pixel_values)
			# Save generated images using the helper function
			save_path = os.path.join(str(config_test.save_dir), self.config.exp_name, str(self.iteration))
			index, image = save_generated_images(pixel_values, prompt_short, save_path, self.config.sks_name, index)
	
	@torch.no_grad()
	def eval_recognition(self, recognition_data_loader, split='test'):
		print('\n\n                Recognition Evaluation \n\n')
		ground_truth = []
		predictions = []

		for batch in tqdm(recognition_data_loader):
			# batch['inputs'] = batch['inputs'].to(model.device)
			# reshape tensor to remove batch dimension
			batch['inputs'] = {k: v.squeeze(1).to(self.model.device) for k, v in batch['inputs'].items()}
			batch['inputs']['pixel_values'] = batch['inputs']['pixel_values'].to(self.model.dtype)

			output = self.model.generate(**batch['inputs'], multimodal_generation_mode="text-only", max_new_tokens=10)
			result_with_special_tokens = self.processor.decode(output[0], skip_special_tokens=False)
			answer = chameleon_trim_answer(result_with_special_tokens)
			# breakpoint()
			if ('Yes' in answer) or ('yes' in answer):
				predictions.append('Yes')
			elif ('No' in answer) or ('no' in answer):
				predictions.append('No')
			else:
				predictions.append(answer)
			ground_truth.extend(batch['labels'])

		positive_indices = [i for i, x in enumerate(ground_truth) if x == 'Yes']
		negative_indices = [i for i, x in enumerate(ground_truth) if x == 'No']

		predict_positive = [predictions[i] for i in positive_indices]
		predict_negative = [predictions[i] for i in negative_indices]
		gt_positive = [ground_truth[i] for i in positive_indices]
		gt_negative = [ground_truth[i] for i in negative_indices]

		accuracy = sum([1 for i, j in zip(ground_truth, predictions) if i == j]) / len(ground_truth)
		positive_accuracy = sum([1 for i, j in zip(gt_positive, predict_positive) if i == j]) / len(gt_positive)
		negative_accuracy = sum([1 for i, j in zip(gt_negative, predict_negative) if i == j]) / len(gt_negative)
		print(f'Accuracy: {accuracy}')
		print(f'Positive Accuracy: {positive_accuracy}')
		print(f'Negative Accuracy: {negative_accuracy}')
		weighted_acc = (positive_accuracy + negative_accuracy) / 2
		if split == 'train':
			if self.weighted_acc <= weighted_acc:
				self.weighted_acc = weighted_acc
				self.save_checkpoint('best-recog')
		answer = html.escape(answer)
		recog_dict = {
			f"Recognition/{split}_accuracy": accuracy,
			f"Recognition/{split}_positive_accuracy": positive_accuracy,
			f"Recognition/{split}_negative_accuracy": negative_accuracy,
			f"Metrics/{split}_weighted_accuracy": weighted_acc,
			"Text/Recognition": wandb.Html(f'<p>{answer}</p>')
		}
		return recog_dict

	@torch.no_grad()
	def eval_clip_similarity(self, real_images, number_fake_images=10):
		print('\n\n                CLIP Similarity Evaluation \n\n')
		if self.config.self_prompting:
			prompt = f'{self.sks_prompt} A photo of {self.identifier}.<reserved08706>{self.identifier} is {self.generation_prompt}'
		else:
			prompt = self.sks_prompt + f' A photo of {self.identifier}.<reserved08706>'
		inputs = self.processor(prompt, return_tensors="pt").to(self.model.device)
		fake_images = []
		for index in tqdm(range(number_fake_images)):
			generate_ids = self.model.generate(**inputs, multimodal_generation_mode="image-only", max_new_tokens=1026, do_sample=True)
			response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
			pixel_values = self.model.decode_image_tokens(response_ids[:, 1:-1])
			pixel_values = self.processor.postprocess_pixel_values(pixel_values)
			image = to_pil_image(pixel_values[0].detach().cpu())
			fake_images.append(image)
		clip_score = self.clip_evaluator.compute_similarity(real_images, fake_images)
		mean_clip = np.mean(clip_score)
		if self.mean_clip <= mean_clip:
			self.save_checkpoint('best-gen')
			self.mean_clip = mean_clip
		return {'Metrics/CLIP': mean_clip}
		# if not self.config.no_wandb:
		# 	self.wandb.log({"Metrics/clip": mean_clip})

	@torch.no_grad()
	def visualize_evaluation(self):
		print('Generate evaluation images...')
		if self.config.self_prompting:
			prompt = f'{self.sks_prompt} A photo of {self.identifier}.<reserved08706>{self.identifier} is {self.generation_prompt}'
		else:
			prompt = self.sks_prompt + f' A photo of {self.identifier}.'
		print(prompt)
		inputs = self.processor(prompt, return_tensors="pt").to(self.model.device)
		generate_ids = self.model.generate(**inputs, multimodal_generation_mode="image-only", max_new_tokens=1026, do_sample=True)
		response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
		pixel_values = self.model.decode_image_tokens(response_ids[:, 1:-1])
		pixel_values = self.processor.postprocess_pixel_values(pixel_values)
		image = to_pil_image(pixel_values[0].detach().cpu())

		print('Generate the text response...')
		prompt = self.sks_prompt + f' Can you describe {self.identifier} in details?'
		inputs = self.processor(prompt, return_tensors="pt").to(self.model.device)
		output = self.model.generate(**inputs, max_new_tokens=200)
		result_with_special_tokens = self.processor.decode(output[0], skip_special_tokens=False)
		answer = chameleon_trim_answer(result_with_special_tokens)
		escaped_string = html.escape(result_with_special_tokens)
		print(answer)
		visual_dict = {
			"Image": wandb.Image(image),
			"Text/Describe": wandb.Html(f'<p>{escaped_string}</p>')
			}
		return visual_dict
			# self.wandb.log({"Text/Prediction": wandb.Html(f'<p>{escaped_string}</p>')})

