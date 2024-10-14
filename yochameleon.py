import os
import torch
import wandb

from itertools import cycle
from tqdm import tqdm
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor
from transformers.image_transforms import to_pil_image
from utils import Config

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
		
		self.sks_prompt = f"{self.personalized_tokens[0]} is {''.join(self.personalized_tokens[1:])}."
		self.orig_embeds_params = self.model.get_input_embeddings().weight.data.clone()
		self.orig_lm_params = self.model.lm_head.weight.data.clone()
		self.index_no_updates = None
		self.iteration = 0


	def prepare_personalized_tokens(self):
		self.latent_tokens_start_index = self.config.special_tokens["LATENT_TOKEN_START"]
		self.identifier = self.config.special_tokens["PERSONALITY_TOKEN"]

		prefix_tokens = [f'<reserved{self.latent_tokens_start_index+i}>' for i in range(self.config.prefix_token)]
		personalized_tokens = [self.identifier ]
		personalized_tokens.extend(prefix_tokens)
		self.personalized_token_ids = self.processor.tokenizer.convert_tokens_to_ids(personalized_tokens)
		self.personalized_tokens = personalized_tokens

		print(f'Personalized tokens: {self.personalized_tokens}')
		print(f'Personalized token ids: {self.personalized_token_ids}')

	def get_model(self):
		self.processor = ChameleonProcessor.from_pretrained(self.config.model_id)
		self.model = ChameleonForConditionalGeneration.from_pretrained(self.config.model_id, device_map="auto")#, torch_dtype=torch.float16)
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
				name=self.config.exp_name,
				entity=self.config.entity,
				config=self.config)
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
		if finetune:
			save_path_token = os.path.join(self.save_location, f'{iteration}-token-ft.pt')
			save_path_lmhead = os.path.join(self.save_location, f'{iteration}-lmhead-ft.pt')
		else:
			save_path_token = os.path.join(self.save_location, f'{iteration}-token.pt')
			save_path_lmhead = os.path.join(self.save_location, f'{iteration}-lmhead.pt')
		torch.save(self.model.get_input_embeddings().weight.data[self.personalized_token_ids], save_path_token)
		print('Saved token embeddings at: ', save_path_token)

		if self.config.whole_model:
			torch.save(self.model.model.state_dict(), os.path.join(self.save_location, f'{iteration}-model.pt'))
			print('Saved whole model at: ', os.path.join(self.save_location, f'{iteration}-model.pt'))
		else:
			torch.save(self.model.lm_head.weight.data[self.personalized_token_ids], save_path_lmhead)
			print('Saved lm_head at: ', save_path_lmhead)

	def resume_training(self):
		# Thao: TODO
		if self.config.resume['resume']:
			config_resume = Config(self.config.resume)
			embedding_path = f'{config_resume.savedir}/{config_resume.exp_name}/{config_resume.sks_name}/{config_resume.resume_iteration}-token.pt'
			try:
				lm_head = torch.load(f'{config_resume.savedir}/{config_resume.exp_name}/{config_resume.sks_name}/{config_resume.resume_iteration}-lmhead.pt', map_location='cuda').to(self.model.lm_head.weight.data.device)
				lm_head = lm_head.to(self.model.dtype)
				# For sequential learning
				if config_resume.sequential_learning:
					self.model.lm_head.weight.data[self.personalized_token_ids[:-config_resume.spacing]] = lm_head
					self.model.get_input_embeddings().weight.data[self.personalized_token_ids[:-config_resume.spacing]] = torch.load(embedding_path).to(self.model.device).to(self.model.dtype)
					self.personalized_token_ids = self.personalized_token_ids[:-config_resume.spacing]
				else:
					self.model.lm_head.weight.data[self.personalized_token_ids] = lm_head
					self.model.get_input_embeddings().weight.data[self.personalized_token_ids] = torch.load(embedding_path).to(self.model.device).to(self.model.dtype)
			except:
				state_dict = torch.load(f'{config_resume.savedir}/{config_resume.exp_name}/{config_resume.sks_name}/{config_resume.resume_iteration}-model.pt')
				self.model.model.load_state_dict(state_dict)

	def configure_model(self):
		if self.config.whole_model:
			self.model.model.requires_grad_(True)
			self.model.model.embed_tokens.weight.requires_grad_(True)
			self.model.model.vqmodel.requires_grad_(False)
			self.index_no_updates = torch.zeros((len(self.processor.tokenizer),), dtype=torch.bool)
		else:
			self.model.model.requires_grad_(False)
			self.model.model.embed_tokens.weight.requires_grad_(True)
			self.index_no_updates = torch.ones((len(self.processor.tokenizer),), dtype=torch.bool)
			self.index_no_updates[self.personalized_token_ids] = False

	def train(self, dataloader):
		dataloader_iter = cycle(dataloader)
		if not self.config.no_wandb:
			self.wandb.log({"train_dataset_length": len(dataloader.dataset)})

		for iteration in tqdm(range(self.config.iteration)):
			self.optimizer.zero_grad()
			batch = next(dataloader_iter)
			batch['pixel_values'] = batch['pixel_values'].to(self.model.dtype)

			# Process labels with image tokens
			for i, item in enumerate(batch['labels']):
				if len(torch.nonzero(batch['labels'][i] == self.config.special_tokens["START_OF_IMAGE_INDEX"])) != 0:
					soi_index = torch.nonzero(batch['labels'][i] == self.config.special_tokens["START_OF_IMAGE_INDEX"]).item() + 1
					eot_index = torch.nonzero(batch['labels'][i] == self.config.special_tokens["END_OF_IMAGE_INDEX"]).item()
					image_tokens = self.model.model.get_image_tokens(pixel_values=batch['pixel_values'][None, i])[0]
					batch['labels'][i, soi_index:eot_index] = image_tokens

			batch = {k: v.to(self.model.device) for k, v in batch.items()}

			# Forward pass
			output = self.model(
				input_ids=batch['input_ids'],
				pixel_values=batch['pixel_values'],
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

			# Save model checkpoints
			if iteration % self.config.save_every == 0:
				self.save_checkpoint(iteration)
				if self.config.eval_visualization:
					self.visualize_evaluation()
			torch.cuda.empty_cache()
		self.iteration = iteration

	def finetune(self, dataloader):
		# this code is similar to train, but it is used for finetuning
		self.get_optimizer_and_scheduler(self.config.finetune)
		dataloader_iter = cycle(dataloader)
		if not self.config.no_wandb:
			self.wandb.log({"finetune_dataset_length": len(dataloader.dataset)})

		total_iter = self.iteration + self.config.finetune['finetune_iteration']
		for iteration in tqdm(range(self.iteration, total_iter)):
			self.optimizer.zero_grad()
			batch = next(dataloader_iter)
			batch['pixel_values'] = batch['pixel_values'].to(self.model.dtype)

			# Process labels with image tokens
			for i, item in enumerate(batch['labels']):
				if len(torch.nonzero(batch['labels'][i] == self.config.special_tokens["START_OF_IMAGE_INDEX"])) != 0:
					soi_index = torch.nonzero(batch['labels'][i] == self.config.special_tokens["START_OF_IMAGE_INDEX"]).item() + 1
					eot_index = torch.nonzero(batch['labels'][i] == self.config.special_tokens["END_OF_IMAGE_INDEX"]).item()
					image_tokens = self.model.model.get_image_tokens(pixel_values=batch['pixel_values'][None, i])[0]
					batch['labels'][i, soi_index:eot_index] = image_tokens

			batch = {k: v.to(self.model.device) for k, v in batch.items()}

			# Forward pass
			output = self.model(
				input_ids=batch['input_ids'],
				pixel_values=batch['pixel_values'],
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

			# Save model checkpoints
			if iteration % self.config.save_every == 0:
				self.save_checkpoint(iteration, finetune=True)
				if self.config.eval_visualization:
					self.visualize_evaluation()
			torch.cuda.empty_cache()
		self.iteration = iteration

	# TODO: Add support test with config iteration?
	def test(self):
		config_test = Config(self.config.test)
		with torch.no_grad():
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

	def visualize_evaluation(self):
		with torch.no_grad():
			print('Generate evaluation images...')
			prompt = self.sks_prompt + f' A photo of {self.identifier}.'
			print(prompt)
			inputs = self.processor(prompt, return_tensors="pt").to(self.model.device)
			generate_ids = self.model.generate(**inputs, multimodal_generation_mode="image-only", max_new_tokens=1026, do_sample=True)
			response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
			pixel_values = self.model.decode_image_tokens(response_ids[:, 1:-1])
			pixel_values = self.processor.postprocess_pixel_values(pixel_values)
			image = to_pil_image(pixel_values[0].detach().cpu())

			if not self.config.no_wandb:
			    self.wandb.log({"Generated Image": wandb.Image(image)})