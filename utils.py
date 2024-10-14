import os 
import torch

from dataset import PersonalizedDataset

from torchvision import datasets

def get_dataloader_iter(config, processor, only_positive=False):
    if only_positive:
        train_dataset = PersonalizedDataset(
            json_file=config.json_file,
            processor=processor,
            tokenizer_max_length=config.tokenizer_max_length,
            END_OF_TURN=config.special_tokens["END_OF_TURN"],
            only_positive=True,
            )
    else:
        train_dataset = PersonalizedDataset(
                json_file=config.json_file,
                processor=processor,
                tokenizer_max_length=config.tokenizer_max_length,
                END_OF_TURN=config.special_tokens["END_OF_TURN"],
                )
        
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1,
    )
    # dataloader_iter = cycle(train_dataloader)
    return train_dataloader

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