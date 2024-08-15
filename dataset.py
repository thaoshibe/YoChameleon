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
from transformers import ChameleonVQVAE
from transformers import ChameleonVQVAEConfig
from transformers.image_transforms import to_pil_image
# from utils import ChameleonImageVocabularyMapping
from utils import ChameleonVQVAEPreprocessor

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
        json_file=None,
        placeholder_token="<sks>",
        center_crop=False,
        personalized_prompt = False,
        repeat=10,
        processor: ChameleonProcessor = None,
        vqvae: ChameleonVQVAE = None
        # get_image_tokens: ChameleonForConditionalGeneration = None,
        # get_image_tokens = None,
    ):
        self.data_root = data_root
        self.sks_name = sks_name
        self.questions = []
        self.images_path = []
        self.answers = []
        self.personalized_prompt = personalized_prompt
        self.processor = processor
        self.vqvae = vqvae
        # self.get_image_tokens = get_image_tokens
        with open(json_file) as f:
            data = json.load(f)
        self.data = data
        self.flip_transform = transforms.RandomHorizontalFlip()

        # self.templates = my_query_templates

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image_paths = [os.path.join(self.data_root, self.sks_name, item) for item in self.data[i]['image']]
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        images = [self.flip_transform(image) for image in images]

        conv = self.data[i]['conversations']
        chat_template = "{% for message in messages %}{% if not (loop.first and message['from'] != 'human') %}{{ message['value'] }}{% if not loop.last %}<reserved08706>{% endif %}{% endif %}{% endfor %}"
        conversations = self.processor.apply_chat_template(conv, chat_template=chat_template)
        full_text = f'{self.personalized_prompt}\n{conversations}'

        
        example = self.processor(
            full_text,
            images=images,
            padding="max_length",
            max_length=2048,
            )
        example['input_ids'] = example['input_ids'][0]
        example['attention_mask'] = example['attention_mask'][0]
        example['pixel_values'] = example['pixel_values'][0]

        clone_inputs = example['input_ids'].clone()
        eot_indices = (clone_inputs == END_OF_TURN).nonzero()[:]
        
        # Initialize a mask with the same shape as the tensor, filled with -100 (mask out question)
        labels = torch.full(clone_inputs.shape, -100)
        for start_idx, end_idx in zip(eot_indices[0::2]+1, eot_indices[1::2]):
            cur_labels = clone_inputs[start_idx:end_idx+1] 
            # check if there is any image token in the current conversation
            # check = torch.nonzero(cur_labels==START_OF_IMAGE_INDEX).shape[0]
            # if check > 0:
            #     soi_index = torch.nonzero(cur_labels==START_OF_IMAGE_INDEX).item()+1
            #     eot_index = torch.nonzero(cur_labels==END_OF_IMAGE_INDEX).item()
            #     #----
            #     image_tokens = self.vqvae.get_image_tokens(pixel_values=example['pixel_values'][None])[0]
            #     breakpoint()
            #     pixel_values = self.vqvae.decode(image_tokens[None])
            #     images = self.processor.postprocess_pixel_values(pixel_values)
            #     image = to_pil_image(images[0].detach().cpu())
            #     image.save("test.png")

            #     cur_labels[soi_index:eot_index] = image_tokens
            # replace <image> to real vq-vae tokens
            labels[start_idx:end_idx+1] = cur_labels
        example['labels'] = labels
        return example

if __name__ == "__main__":

    model_id = 'leloy/Anole-7b-v0.1-hf'
    processor = ChameleonProcessor.from_pretrained(model_id)
    model = ChameleonForConditionalGeneration.from_pretrained(model_id, device_map="auto")
    pretrained_vqvae = ChameleonVQVAEPreprocessor.from_pretrained(model_id)
    print(f'Loaded {model_id}!')
    train_dataset = PersonalizedDataset(
        data_root="./yollava-data/train/",
        json_file='./example_training_data/v1/bo.json',
        sks_name='bo',
        personalized_prompt="<bo> is a cat.",
        processor=processor,
        vqvae=pretrained_vqvae,
        )
    print(train_dataset[0])
    #-- test labels
    labels = train_dataset[0]['labels']
    pixel_values = model.decode_image_tokens(labels[1022:-2][None])
    images = processor.postprocess_pixel_values(pixel_values)
    image = to_pil_image(images[0].detach().cpu())
    # image.save("test.png")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=3, shuffle=True, num_workers=0,
    )
    for i, batch in enumerate(train_dataloader):
        print(len(batch['input_ids']), i)
    print('Done one loop on dataset')