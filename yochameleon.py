import torch
import wandb

import os

from tqdm import tqdm
from transformers.image_transforms import to_pil_image
from utils import Config
from utils import collate_fn
from utils import get_model
from utils import get_optimizer_and_scheduler
from utils import prepare_personalized_tokens
from utils import save_checkpoint
from utils import setup_logger

class YoChameleonTrainer:
    def __init__(self, config):
        self.config = config
        self.processor, self.model = get_model(config)
        self.personalized_token_ids, personalized_tokens = prepare_personalized_tokens(config, self.processor)
        self.sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}."
        self.optimizer, self.scheduler, self.optimizer_config, self.scheduler_config = get_optimizer_and_scheduler(config, self.model)
        self.wandb, self.save_location = setup_logger(config)
        self.orig_embeds_params = self.model.get_input_embeddings().weight.data.clone()
        self.orig_lm_params = self.model.lm_head.weight.data.clone()
        self.index_no_updates = None

    def resume_training(self):
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

    def train(self, dataloader_iter):
        if not self.config.no_wandb:
            wandb.log({"train_dataset_length": len(dataloader_iter)})
        
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
                wandb.log({"loss": loss.item()})

            # Save model checkpoints
            if iteration % self.config.save_every == 0:
                self.save_checkpoint(iteration)

                if self.config.eval_visualization:
                    self.visualize_evaluation()

            torch.cuda.empty_cache()

    def save_checkpoint(self, iteration):
        save_checkpoint(self.model, self.config, iteration, self.personalized_token_ids, self.save_location)

    def visualize_evaluation(self):
        with torch.no_grad():
            print('Generate evaluation images...')
            prompt = self.sks_prompt + ' A photo of <reserved16300>.'
            print(prompt)
            inputs = self.processor(prompt, return_tensors="pt").to(self.model.device)
            generate_ids = self.model.generate(**inputs,
                                               multimodal_generation_mode="image-only",
                                               max_new_tokens=1026,
                                               do_sample=True)
            response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
            pixel_values = self.model.decode_image_tokens(response_ids[:, 1:-1])
            pixel_values = self.processor.postprocess_pixel_values(pixel_values)
            image = to_pil_image(pixel_values[0].detach().cpu())

            if not self.config.no_wandb:
                wandb.log({"Generated Image": wandb.Image(image)})

            # print('Generated images are saved in ', self.save_location)