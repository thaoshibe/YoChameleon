import os 
import shutil

from torch.utils.tensorboard import SummaryWriter

START_OF_IMAGE_INDEX = 8197 # <racm3:break>
END_OF_IMAGE_INDEX = 8196 # <eoss>
END_OF_TURN = 8710
PAD_INDEX = 1

def setup(config):
    log_dir = f'./runs/{config.exp_name}/{config.sks_name}'
    
    if os.path.exists(log_dir):
        # Delete the directory and its contents
        shutil.rmtree(log_dir)
    writer = SummaryWriter(f'./runs/{config.exp_name}/{config.sks_name}')
    save_location = f'./{config.savedir}/{config.exp_name}/{config.sks_name}'
    os.makedirs(save_location, exist_ok=True)
    return writer, save_location

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