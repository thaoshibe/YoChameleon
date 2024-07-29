import json
import os

import glob

import numpy as np

import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision import transforms
from transformers import ChameleonProcessor

START_OF_IMAGE_INDEX = 8197 # <racm3:break>
END_OF_IMAGE_INDEX = 8196 # <eoss>
END_OF_TURN = 8710
PAD_INDEX = 1

class PersonalizedDataset(Dataset):
    def __init__(
        self,
        data_root,
        sks_name,
        set="train",
        placeholder_token="<sks>",
        center_crop=False,
        device="cuda",
        config=None,
        flip_p=0.5,
        personalized_prompt = False,
        # get_image_tokens = None,
    ):
        self.data_root = data_root
        self.device = device
        self.config = config
        # self.processor = ChameleonProcessor.from_pretrained(model_id)
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.sks_name = sks_name
        self.questions = []
        self.images_path = []
        self.answers = []
        self.has_image = []
        self.require_image_generation = []
        self.personalized_prompt = personalized_prompt
        # self.get_image_tokens = get_image_tokens
        # --- Load data from json files

        conversation_types = ['recognition_positive', 'recognition_negative-laion', 'recognition_negative-cc12m', 'text-only-conversation']
        for conversation_type in conversation_types:
            f = open(os.path.join(data_root, sks_name, f'{conversation_type}.json'))
            data = json.load(f)
            file_names = [x for x in data.keys()]
            for file_name in file_names:
                questions = []
                answers = []
                for conv in data[file_name]:
                    questions.append(conv['Human'])
                    answers.append(conv['AI'])

                self.questions.extend(questions)
                self.answers.extend(answers)
                
                self.images_path.extend([file_name]*len(answers))
                if conversation_type == 'text-only-conversation':
                    self.has_image.extend([False]*len(answers))
                    # self.require_image_generation.extend([False]*len(answers))
                else:
                    self.has_image.extend([True]*len(answers))
                self.require_image_generation.extend([False]*len(answers))
            print(conversation_type, len(self.questions))
        print('Total: ', len(self.questions), len(self.answers), len(self.images_path), len(self.has_image))
        # # Add data for image generation
        # gt_images = [x for x in self.images_path if f'train/{self.sks_name}' in x]
        # for image_path in gt_images:
        #     self.questions.append('')
        #     self.answers.append('<image>')
        #     self.images_path.append(image_path)
        #     self.has_image.extend([False])
        #     self.require_image_generation.extend([True])

        if set == "train":
            self._length = len(self.questions)
        else:
            self._length = self.num_images
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

        # self.templates = my_query_templates

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        # --- Center crop -- Not sure?
        # if self.center_crop:
        #     crop = min(img.shape[0], img.shape[1])
        #     (
        #         h,
        #         w,
        #     ) = (
        #         img.shape[0],
        #         img.shape[1],
        #     )
        #     img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
        # breakpoint()
        # image_generation = self.require_image_generation[i]
        image_path = self.images_path[i]
        image = Image.open(image_path).convert("RGB")
        image = self.flip_transform(image)

        example['question'] = self.questions[i]
        example['answer'] = self.answers[i]
        example['has_image'] = self.has_image[i]
        example['image_generation'] = self.require_image_generation[i]
        example['image'] = image
        # TODO: clean up this condition
        if example['has_image']:
            example['input'] = f'{self.personalized_prompt}{self.questions[i]}<image><reserved08706>{self.answers[i]}'
        else:
            example['input'] = f'{self.personalized_prompt}{self.questions[i]}<reserved08706>{self.answers[i]}'
        if example['image_generation']:
            example['input'] = f'{self.personalized_prompt}{self.questions[i]}<reserved08706><image>{self.answers[i]}'
        return example

# def collate_fn(batch):
#     # unpack batch
#     breakpoint()
#     text = [_[0] for _ in batch]
#     target = [_[1] for _ in batch]

#     # get the input tokens
#     input_tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

#     # get the target
#     target = torch.tensor(target)

#     return input_tokens, target
def collate_fn(batch):
    images = [item['image'] for item in batch]
    inputs = [item['input'] for item in batch]

    # question = [f'{questions[i]}{answers[i]}' for i in range(len(questions))]
    example = chemeleon_processor(inputs, images, padding=True)
    example['labels'] = example['input_ids'].clone()

    # Find the index of the first occurrence of END_OF_TURN in each sequence
    batch_size, seq_len = example['labels'].shape
    eot_mask = example['labels'] == END_OF_TURN
    eot_indices = torch.argmax(eot_mask.int(), dim=1)+1

    # Create a mask for the positions to be replaced with -100
    mask = torch.arange(seq_len).expand(batch_size, seq_len) < eot_indices.unsqueeze(1)
    # Apply the mask to the labels
    example['labels'][mask] = -100

    eot_index = torch.nonzero(example['labels']==END_OF_TURN).item()
    soi_index = torch.nonzero(example['labels']==START_OF_IMAGE_INDEX).item()
    # input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)

    return example


if __name__ == "__main__":

    model_id = 'leloy/Anole-7b-v0.1-hf'
    chemeleon_processor = ChameleonProcessor.from_pretrained(model_id)

    train_dataset = PersonalizedDataset(
        data_root="./example_training_data/",
        sks_name='mam',
        personalized_prompt="<sks> is a cat."
        )
    print(train_dataset[400])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=3, shuffle=True, num_workers=0, collate_fn=collate_fn,
    )
    for i, batch in enumerate(train_dataloader):
        print(len(batch['query']), i)
    print('Done one loop on dataset')