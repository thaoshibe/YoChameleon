import json
import os

import glob

import numpy as np

import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor

START_OF_IMAGE_INDEX = 8197 # <racm3:break>
END_OF_IMAGE_INDEX = 8196 # <eoss>
END_OF_TURN = 8710
PAD_INDEX = 1

# END-OF-TURN token: <reserved08706>

class PersonalizedDataset(Dataset):
    def __init__(
        self,
        data_root,
        sks_name,
        set="train",
        placeholder_token="<sks>",
        center_crop=False,
        flip_p=0.5,
        personalized_prompt = False,
        repeat=10,
        processor: ChameleonProcessor = None,
        # get_image_tokens = None,
    ):
        self.data_root = data_root
        self.flip_p = flip_p
        self.sks_name = sks_name
        self.questions = []
        self.images_path = []
        self.answers = []
        self.personalized_prompt = personalized_prompt
        self.processor = processor
        gt_images = glob.glob(os.path.join(data_root, self.sks_name, '*.png'))

        with open(f'./preprocess/{self.sks_name}.json') as f:
            captions = json.load(f)

        for image_path in gt_images:
            self.questions.append(captions[image_path])
            self.answers.append('<image>')
            self.images_path.extend([image_path])

        # repeat for more data
        if set == "train":
            self.questions = self.questions*repeat
            self.answers = self.answers*repeat
            self.images_path = self.images_path*repeat

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

        image_path = self.images_path[i]
        image = Image.open(image_path).convert("RGB")
        image = self.flip_transform(image)

        # example['question'] = self.questions[i]
        # example['answer'] = self.answers[i]
        # example['image'] = image
        example = self.processor(
            self.questions[i],
            images=image,
            padding="max_length",
            max_length=4096,
            )
        # example['labels'] = example['labels'][0]
        example['input_ids'] = example['input_ids'][0]
        example['attention_mask'] = example['attention_mask'][0]
        example['pixel_values'] = example['pixel_values'][0]
        example['labels'] = example['input_ids'].clone()
        return example

if __name__ == "__main__":

    model_id = 'leloy/Anole-7b-v0.1-hf'
    processor = ChameleonProcessor.from_pretrained(model_id)
    
    print(f'Loaded {model_id}!')
    train_dataset = PersonalizedDataset(
        data_root="./yollava-data/train/",
        sks_name='bo',
        personalized_prompt="<sks> is a cat.",
        processor=processor,
        )
    print(train_dataset[0])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=3, shuffle=True, num_workers=0,
    )
    for i, batch in enumerate(train_dataloader):
        print(len(batch['input_ids']), i)
    print('Done one loop on dataset')