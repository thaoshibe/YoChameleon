import requests
import torch

import os
import torch
import wandb

from PIL import Image
from transformers import Emu3ForConditionalGeneration
from transformers import Emu3Processor

import re

import numpy as np

import html

from PIL import Image
from evaluation.clip_image_similarity import CLIPEvaluator
from itertools import cycle
from tqdm import tqdm
from transformers.image_transforms import to_pil_image
from utils import Config
# out = model.generate(
#     **inputs,
#     max_new_tokens=50000, # make sure to have enough tokens for one image
#     prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
#     return_dict_in_generate=True,
#     negative_prompt_ids=neg_inputs.input_ids, # indicate for Classifier-Free Guidance
#     negative_prompt_attention_mask=neg_inputs.attention_mask,
# )

# image = model.decode_image_tokens(out.sequences[:, inputs.input_ids.shape[1]: ], height=HEIGHT, width=WIDTH)
# images = processor.postprocess(list(image.float()), return_tensors="PIL.Image.Image") # internally we convert to np but it's not supported in bf16 precision
# for i, image in enumerate(images['pixel_values']):
#     image.save(f"result{i}.png")

def save_generated_images(pixel_values, prompt_short, save_path, sks_name, index):
    """Save generated images to a specified directory."""
    for pixel_value in pixel_values:
        image = to_pil_image(pixel_value.detach().cpu())
        prompt_short = prompt_short.replace('<reserved16200>', sks_name).replace('.', '')
        os.makedirs(save_path, exist_ok=True)
        image.save(f'{save_path}/{prompt_short}_{index}.png')
        index += 1
    return index, image

class YoEmu3Trainer:
    def __init__(self, config):
        self.config = config
        self.get_model()
        self.prepare_personalized_tokens()
        self.get_optimizer_and_scheduler(config) # get optimizer and scheduler for pretraining
        self.setup_logger()
        self.sks_name = config.sks_name
        self.sks_prompt = f"{self.personalized_tokens[0]} is {''.join(self.personalized_tokens[1:])}."
        self.orig_embeds_params = self.model.get_input_embeddings().weight.data.clone()
        self.orig_lm_params = self.model.text_model.lm_head.weight.data.clone()
        self.index_no_updates = None
        self.iteration = 0
        self.clip_evaluator = CLIPEvaluator()
        self.weighted_acc = 0.0
        self.mean_clip = 0.0
        self.avg_metric = 0.0
        self.VISUAL_TOKENS = self.model.vocabulary_mapping.image_tokens

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
            gen_prefix_tokens = [f'<|extra_{self.latent_tokens_start_index+i}|>' for i in range(self.config.prefix_token)]
            # understanding tokens
            understand_prefix_tokens = [f'<|extra_{self.latent_tokens_start_index+self.config.prefix_token+i}|>' for i in range(self.config.prefix_token)]
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

            prefix_tokens = [f'<|extra_{self.latent_tokens_start_index+i}|>' for i in range(self.config.prefix_token)]
            personalized_tokens = [self.identifier]
            personalized_tokens.extend(prefix_tokens)

            self.personalized_tokens = personalized_tokens
            self.personalized_token_ids = self.processor.tokenizer.convert_tokens_to_ids(personalized_tokens)
            print(f'Personalized tokens: {self.personalized_tokens}')
            print(f'Personalized token ids: {self.personalized_token_ids}')
            print(f'There are {len(self.personalized_tokens)} personalized tokens')

    def get_model(self):
        # self.processor = ChameleonProcessor.from_pretrained(self.config.model_id)
        # self.model = ChameleonForConditionalGeneration.from_pretrained(self.config.model_id, device_map="auto", torch_dtype=torch.bfloat16)
        # return processor, model
        self.processor = Emu3Processor.from_pretrained(self.config.model_id)
        self.model = Emu3ForConditionalGeneration.from_pretrained(self.config.model_id, torch_dtype="bfloat16", device_map="auto", attn_implementation="flash_attention_2")
        print(f'Loaded {self.config.model_id}!')

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
            trainable_params = self.model.text_model.parameters()
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=float(optimizer_config.lr),
                betas=tuple(optimizer_config.betas),
                weight_decay=float(optimizer_config.weight_decay),
                eps=float(optimizer_config.eps)
            )
        else:
            # train embedding weights and lm only
            # trainable_params = [self.model.get_input_embeddings().weight, self.model.text_model.lm_head.weight]
            trainable_params = [self.model.get_input_embeddings().weight, self.model.text_model.lm_head.weight]
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
        #   iteration=iteration+1 # increment iteration to save the correct iteration as python starts from 0
        if finetune:
            save_path_token = os.path.join(self.save_location, f'{iteration}-token-ft.pt')
            save_path_lmhead = os.path.join(self.save_location, f'{iteration}-lmhead-ft.pt')
        else:
            save_path_token = os.path.join(self.save_location, f'{iteration}-token.pt')
            save_path_lmhead = os.path.join(self.save_location, f'{iteration}-lmhead.pt')
        torch.save(self.model.get_input_embeddings().weight.data[self.personalized_token_ids], save_path_token)
        print('Saved token embeddings at: ', save_path_token)

        if self.config.whole_model:
            torch.save(self.model.text_model.state_dict(), os.path.join(self.save_location, f'{iteration}-model.pt'))
            print('Saved whole model at: ', os.path.join(self.save_location, f'{iteration}-model.pt'))
        else:
            torch.save(self.model.text_model.lm_head.weight.data[self.personalized_token_ids], save_path_lmhead)
            print('Saved lm_head at: ', save_path_lmhead)


    def load_prefix(self, config_resume, exp_name, resume_token_ids):
        lm_head_path = os.path.join(config_resume.savedir, exp_name, self.config.sks_name, f"{config_resume.resume_iteration}-lmhead.pt")
        embedding_path = os.path.join(config_resume.savedir, exp_name, self.config.sks_name, f"{config_resume.resume_iteration}-token.pt")
        # Load language model head
        lm_head = torch.load(lm_head_path, map_location='cuda').to(self.model.text_model.lm_head.weight.data.device)
        lm_head = lm_head.to(self.model.dtype)
        self.model.text_model.lm_head.weight.data[resume_token_ids] = lm_head

        # Load input embeddings
        embeddings = torch.load(embedding_path).to(self.model.device).to(self.model.dtype)
        self.model.get_input_embeddings().weight.data[resume_token_ids] = embeddings

        print('\n\n\n           ATTENTION -- PLEASE YOU CHECK IF THE RESUME IS CORRECT!\n\n\n')
        print(f'\n\n\n Resume tokens ids: {resume_token_ids} \n From: {exp_name} at epochs {config_resume.resume_iteration}\n\n\n')

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
                    self.model.text_model.load_state_dict(state_dict)
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
            self.model.text_model.requires_grad_(True)
            self.model.text_model.model.embed_tokens.weight.requires_grad_(True)
            self.model.text_model.vqmodel.requires_grad_(False)
            self.index_no_updates = torch.zeros((len(self.processor.tokenizer),), dtype=torch.bool)
        else:

            self.model.text_model.requires_grad_(False)
            self.model.text_model.model.embed_tokens.weight.requires_grad_(True)
            self.index_no_updates = torch.ones((len(self.processor.tokenizer),), dtype=torch.bool)
            self.index_no_updates[self.personalized_token_ids] = False

    def train_epoch(self, dataloader, recognition_data_loader_train=None, recognition_data_loader_test=None):
        if not self.config.no_wandb:
            self.wandb.log({"Dataset/Train_dataset_length": len(dataloader.dataset)})
            self.mean_clip_at_best = 0.0
            self.weighted_acc_at_best = 0.0
            self.weighted_acc = 0.0
            self.mean_clip = 0.0

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
                    if not self.config.no_wandb:
                        self.wandb.log(visual_dict)
            for batch in tqdm(dataloader):
                self.optimizer.zero_grad()
                batch['pixel_values'] = batch['pixel_values'].to(self.model.dtype)

                # Process labels with image tokens
                for i, item in enumerate(batch['labels']):
                    soi_index = torch.nonzero(batch['input_ids'][i] == self.config.special_tokens["START_OF_IMAGE_INDEX"]).item() + 1
                    eot_index = torch.nonzero(batch['input_ids'][i] == self.config.special_tokens["END_OF_IMAGE_INDEX"]).item()
                    image_tokens = self.model.get_image_tokens(pixel_values=batch['pixel_values'][None, i], image_sizes=batch['image_sizes'][None, i])
                    batch['labels'][i, soi_index:eot_index] = image_tokens
                    batch['input_ids'][i, soi_index:eot_index] = image_tokens
                # breakpoint()
                # for i, item in enumerate(batch['input_ids']):
                #     if len(torch.nonzero(batch['input_ids'][i] == self.config.special_tokens["START_OF_IMAGE_INDEX"])) != 0:
                #         soi_index = torch.nonzero(batch['input_ids'][i] == self.config.special_tokens["START_OF_IMAGE_INDEX"]).item() + 1
                #         eot_index = torch.nonzero(batch['input_ids'][i] == self.config.special_tokens["END_OF_IMAGE_INDEX"]).item()
                #         image_tokens = self.model.get_image_tokens(pixel_values=batch['pixel_values'][None, i])[0]
                #         batch['input_ids'][i, soi_index:eot_index] = image_tokens
                #         # print('image tokens added to input_ids')
                # self.model.get_image_tokens(pixel_values=batch['pixel_values'][None, i], image_sizes=batch['image_sizes'])
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
                    torch.nn.utils.clip_grad_value_(self.model.text_model.parameters(), clip_value=self.optimizer_config.grad_clip)

                # Revert embeddings if not training the whole model
                if not self.config.whole_model:
                    with torch.no_grad():
                        self.model.get_input_embeddings().weight[self.index_no_updates] = self.orig_embeds_params[self.index_no_updates]
                        self.model.text_model.lm_head.weight[self.index_no_updates] = self.orig_lm_params[self.index_no_updates]

                # Log loss to W&B
                if not self.config.no_wandb:
                    self.wandb.log({"loss": loss.item()})
            torch.cuda.empty_cache()
            self.iteration = iteration
    @torch.no_grad()
    def prefix_allowed_tokens_fn(self, batch_id, input_ids):
        height, width = self.HEIGHT, self.WIDTH
        visual_tokens = self.VISUAL_TOKENS
        image_wrapper_token_id = self.processor.tokenizer.encode("<|image token|>", return_tensors="pt")[0].to(self.model.device)
        eoi_token_id = self.processor.tokenizer.encode("<|image end|>", return_tensors="pt")[0]
        eos_token_id = self.processor.tokenizer.encode("<|extra_204|>", return_tensors="pt")[0]
        pad_token_id = self.processor.tokenizer.encode("<|endoftext|>", return_tensors="pt")[0]
        eol_token_id = self.processor.tokenizer.encode("<|extra_200|>", return_tensors="pt")[0]
        eof_token_id = self.processor.tokenizer.encode("<|extra_201|>", return_tensors="pt")[0]

        position = torch.nonzero(input_ids == image_wrapper_token_id, as_tuple=True)[0][0]
        offset = input_ids.shape[0] - position
        if offset % (width + 1) == 0:
            return (eol_token_id, )
        elif offset == (width + 1) * height + 1:
            return (eof_token_id, )
        elif offset == (width + 1) * height + 2:
            return (eoi_token_id, )
        elif offset == (width + 1) * height + 3:
            return (eos_token_id, )
        elif offset > (width + 1) * height + 3:
            return (pad_token_id, )
        else:
            return visual_tokens

    @torch.no_grad()
    def visualize_evaluation(self):
        print('Generate evaluation images...')
        if self.config.self_prompting:
            prompt = f'{self.sks_prompt} A photo of {self.identifier}.<reserved08706>{self.generation_prompt}'
        else:
            prompt = self.sks_prompt + f' A photo of {self.identifier}.'
        print(prompt)

        # inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        inputs = self.processor(text=prompt, return_tensors="pt", return_for_image_generation=True).to(self.model.device)
        
        neg_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
        NEGATIVE_PROMPT = self.processor(text=neg_prompt, return_tensors="pt").to(device=self.model.device)

        image_sizes = inputs.pop("image_sizes")
        self.HEIGHT, self.WIDTH = image_sizes[0]

        out = self.model.generate(
            **inputs,
            max_new_tokens=50_000, # make sure to have enough tokens for one image
            prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
            return_dict_in_generate=True,
            negative_prompt_ids=NEGATIVE_PROMPT.input_ids, # indicate for Classifier-Free Guidance
            negative_prompt_attention_mask=NEGATIVE_PROMPT.attention_mask,
        )
        image = self.model.decode_image_tokens(out.sequences[:, inputs.input_ids.shape[1]: ], height=self.HEIGHT, width=self.WIDTH)
        images = self.processor.postprocess(list(image.float()), return_tensors="PIL.Image.Image") # internally we convert to np but it's not supported in bf16 precision
        for i, image in enumerate(images['pixel_values']):
            image.save(f"result{i}.png")
            print(f"Saved result{i}.png")
        image = image.resize((256, 256))
        visual_dict = {
            "Image": wandb.Image(image),
            # "Text/Describe": wandb.Html(f'<p>{escaped_string}</p>')
            }
        return visual_dict
