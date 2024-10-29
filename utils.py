import os 
import torch

from dataset import PersonalizedDataset
from dataset import RecognitionData

import re

from torchvision import datasets

def chameleon_trim_answer(long_answer):
    end_of_turn = '<reserved08706>'
    pattern = r"<reserved08706>(.*)"
    short_answer = re.findall(pattern, long_answer)[0] # trim the first end of turn
    short_answer = short_answer.split(end_of_turn)[0] # trim the second end of turn
    return short_answer
    
def get_dataloader_iter(config, processor, only_positive=False, personalized_prompt=None):
    if not hasattr(config, 'task_disjoin'):
        config.task_disjoin = False
    if only_positive:
        train_dataset = PersonalizedDataset(
                json_file=config.json_file,
                processor=processor,
                placeholder_token=config.special_tokens["PERSONALITY_TOKEN"],
                tokenizer_max_length=config.tokenizer_max_length,
                END_OF_TURN=config.special_tokens["END_OF_TURN"],
                only_positive=True,
                personalized_prompt=personalized_prompt,
                task_disjoin=config.task_disjoin
            )
    else:
        train_dataset = PersonalizedDataset(
                json_file=config.json_file,
                processor=processor,
                placeholder_token=config.special_tokens["PERSONALITY_TOKEN"],
                tokenizer_max_length=config.tokenizer_max_length,
                END_OF_TURN=config.special_tokens["END_OF_TURN"],
                personalized_prompt=personalized_prompt,
                task_disjoin=config.task_disjoin
            )
        
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1,
    )
    # dataloader_iter = cycle(train_dataloader)
    return train_dataloader

def get_eval_dataloader(config, processor, image_folder, personalized_prompt=None):
    eval_dataset = RecognitionData(
        sks_name=config.sks_name,
        image_folder=image_folder,
        placeholder_token=config.special_tokens["PERSONALITY_TOKEN"],
        tokenizer_max_length=config.tokenizer_max_length,
        processor=processor,
        personalized_prompt=personalized_prompt,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=config.batch_size, shuffle=False, num_workers=1,
    )
    return eval_dataset

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