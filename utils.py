import argparse
import glob
import os 
import re
import torch

import sys

from PIL import Image
from dataset import PersonalizedDataset
from dataset import RecognitionData
from torchvision import datasets
from tqdm import tqdm

# class CLIPEvaluator:
#     def __init__(self, model_id="openai/clip-vit-base-patch32"):
#         self.model = CLIPModel.from_pretrained(model_id)
#         self.preprocessor = CLIPImageProcessor.from_pretrained(model_id)
#         print(f'\n             Hello, CLIPEvaluator is loaded from {model_id}\n')

#     def load_image(selfm, image_path):
#         image = Image.open(image_path)
#         return image

#     def preprocess(self, image):
#         image_pt = self.preprocessor(images=image, return_tensors="pt")["pixel_values"]
#         return image_pt

#     # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
#     #
#     #       This function will take a list of images
#     #           If a list of images are given, then compute the average embeddings
#     #
#     # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
#     @torch.no_grad
#     def get_image_features(self, image_ft):
#         # image_ft = [self.model.get_image_features(image) for image in tqdm(image_ft)]
#         image_ft = self.model.get_image_features(image_ft)
#         return image_ft

#     def compute_similarity(self, real_images, fake_images, average=True):
#         # --- Check if given list are images or paths
#         # breakpoint()
#         if isinstance(real_images[0], str):
#             real_images = [self.load_image(image_path) for image_path in real_images]
#             fake_images = [self.load_image(image_path) for image_path in fake_images]
#         real_images_ft = [self.preprocess(image) for image in real_images]
#         real_images_ft = [self.get_image_features(image_ft) for image_ft in real_images_ft]

#         if average:
#             real_images_ft = torch.concat(real_images_ft, dim=0).mean(dim=0, keepdim=True)

#         # Thao: TODO: Implement the average=False case (?)
#         clip_scores = []

#         print("\n\n\n             compute CLIP similarity score between generated and real images\n\n\n")
#         for fake_image in tqdm(fake_images):
#             fake_images_ft = self.preprocess(fake_image)
#             fake_images_ft = self.get_image_features(fake_images_ft)
#             similarity_score = torch.nn.functional.cosine_similarity(real_images_ft, fake_images_ft).item()
#             clip_scores.append(similarity_score)
#         return clip_scores

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