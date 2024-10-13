import os 
import shutil

import torch

import wandb

from dataset import PersonalizedDataset
from functools import cached_property
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

def prepare_personalized_tokens(config, processor):
    prefix_tokens = [f'<reserved{16301+i}>' for i in range(config.prefix_token)]
    personalized_tokens = [config.special_tokens["PERSONALITY_TOKEN"]]
    personalized_tokens.extend(prefix_tokens)
    personalized_token_ids = processor.tokenizer.convert_tokens_to_ids(personalized_tokens)
    print(f'Personalized tokens: {personalized_tokens}')
    print(f'Personalized token ids: {personalized_token_ids}')
    return personalized_token_ids, personalized_tokens

def get_model(config):
    processor = ChameleonProcessor.from_pretrained(config.model_id)
    model = ChameleonForConditionalGeneration.from_pretrained(config.model_id, device_map="auto")#, torch_dtype=torch.float16)
    print(f'Loaded {config.model_id}!')
    return processor, model
    
def get_optimizer_and_scheduler(config, model):
    optimizer_config = Config(config.optimizer)
    scheduler_config = Config(config.scheduler)
    if config.whole_model:
        trainable_params = model.model.parameters()
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(optimizer_config.lr),
            betas=tuple(optimizer_config.betas),
            weight_decay=float(optimizer_config.weight_decay),
            eps=float(optimizer_config.eps)
        )
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler.step_size, gamma=scheduler.gamma)
    else:
        # train embedding weights and lm only
        trainable_params = [model.get_input_embeddings().weight, model.lm_head.weight]
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
    return optimizer, scheduler, optimizer_config, scheduler_config

def save_checkpoint(model, config, iteration, save_token_ids, save_location):
    save_path_token = os.path.join(save_location, f'{iteration}-token.pt')
    save_path_lmhead = os.path.join(save_location, f'{iteration}-lmhead.pt')
    torch.save(model.get_input_embeddings().weight.data[save_token_ids], save_path_token)
    print('Saved token embeddings at: ', save_path_token)

    if config.whole_model:
        torch.save(model.model.state_dict(), os.path.join(save_location, f'{iteration}-model.pt'))
        print('Saved whole model at: ', os.path.join(save_location, f'{iteration}-model.pt'))
    else:
        torch.save(model.lm_head.weight.data[save_token_ids], save_path_lmhead)
        print('Saved lm_head at: ', save_path_lmhead)

def get_dataloader_iter(config, processor):
    train_dataset = PersonalizedDataset(
            json_file=config.json_file,
            processor=processor,
            tokenizer_max_length=config.tokenizer_max_length,
            END_OF_TURN=config.special_tokens["END_OF_TURN"],
            )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1,
    )
    dataloader_iter = cycle(train_dataloader)
    return dataloader_iter

def setup_logger(config):
    log_dir = f'./runs/{config.exp_name}/{config.sks_name}'
    # This is for tensorboard, which is not used for this project anymore
    # if os.path.exists(log_dir):
    #     shutil.rmtree(log_dir)
    # writer = SummaryWriter(f'./runs/{config.exp_name}/{config.sks_name}')
    save_location = f'{config.savedir}/{config.exp_name}/{config.sks_name}'
    os.makedirs(save_location, exist_ok=True)
    if not config.no_wandb:
        wandb.init(project=config.project_name,
            name=config.exp_name,
            entity=config.entity,
            config=config_dict)
    else:
        wandb = None
    return wandb, save_location

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def collate_fn(batch):
    inputs = [item['input'] for item in batch]
    images = [item['image'] for item in batch]
    img_gen_bools = [item['image_generation'] for item in batch]
    # question = [f'{questions[i]}{answers[i]}' for i in range(len(questions))]
    example = processor(inputs, images, padding=True)
    example['labels'] = example['input_ids'].clone()

    # Find the index of the first occurrence of END_OF_TURN in each sequence
    batch_size, seq_len = example['labels'].shape
    eot_mask = example['labels'] == END_OF_TURN
    eot_indices = torch.argmax(eot_mask.int(), dim=1)

    # Create a mask for the positions to be replaced with -100
    mask = torch.arange(seq_len).expand(batch_size, seq_len) < eot_indices.unsqueeze(1)

    # Apply the mask to the labels
    example['labels'][mask] = -100
    example['img_gen_bools'] = img_gen_bools
    return example