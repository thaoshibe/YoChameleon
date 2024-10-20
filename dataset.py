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

# END-OF-TURN token: <reserved08706>

class PersonalizedDataset(Dataset):
    def __init__(
        self,
        json_file=None,
        placeholder_token="<reserved16200>",
        center_crop=False,
        repeat=10,
        tokenizer_max_length=2048, 
        processor: ChameleonProcessor = None,
        END_OF_TURN: int = 8710,
        only_positive: bool = False,
        personalized_prompt: str = None,
    ):
        self.processor = processor
        self.placeholder_token = placeholder_token
        self.max_length = tokenizer_max_length
        self.personalized_prompt = personalized_prompt
        self.END_OF_TURN = END_OF_TURN
        data = []
        for file in json_file:
            print(f"Loading {file}")
            with open(file) as f:
                info = json.load(f)
                data.extend(info)
        self.data = data
        if only_positive:
            # If only train with positive images, then filter out all the negative_example in the image path
            self.data = [d for d in self.data if 'negative_example' not in d['image'][0]]
        self.flip_transform = transforms.RandomHorizontalFlip()
        # self.templates = my_query_templates


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image_paths = self.data[i]['image']
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        images = [self.flip_transform(image) for image in images]

        conv = self.data[i]['conversations']
        # Manually added personalized prompt for text-only generation and image understanding
        if conv[-1]['value'] != "<image>":
            conv[0]['value'] = f'{self.personalized_prompt} {conv[0]["value"]}'
            # print(conv)
        chat_template = "{% for message in messages %}{% if not (loop.first and message['from'] != 'human') %}{{ message['value'] }}{% if not loop.last %}<reserved08706>{% endif %}{% endif %}{% endfor %}"
        conversations = self.processor.apply_chat_template(conv, chat_template=chat_template)

        full_text = conversations.replace("<sks>", self.placeholder_token)
        example = self.processor(
            full_text,
            images=images,
            padding="max_length",
            max_length=self.max_length,
            )
        example['input_ids'] = example['input_ids'][0]
        example['attention_mask'] = example['attention_mask'][0]
        example['pixel_values'] = example['pixel_values'][0]

        clone_inputs = example['input_ids'].clone()
        eot_indices = (clone_inputs == self.END_OF_TURN).nonzero()[:]
        
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
    # model = ChameleonForConditionalGeneration.from_pretrained(model_id, device_map="auto")
    # pretrained_vqvae = ChameleonVQVAEPreprocessor.from_pretrained(model_id)
    # print(f'Loaded {model_id}!')
    # train_dataset = PersonalizedDataset(
    #     data_root="/mnt/localssd/code/data/minibo/",
    #     json_file=[
    #         '/mnt/localssd/code/data/minibo/text-conversation.json',
    #         '/mnt/localssd/code/data/minibo/recognition.json'
    #         ],
    #     sks_name='bo',
    #     personalized_prompt="<bo> is a cat.",
    #     processor=processor,
    #     # vqvae=pretrained_vqvae,
    #     )
    # print(train_dataset[0])
    # #-- test labels
    # labels = train_dataset[0]['labels']
    # pixel_values = model.decode_image_tokens(labels[1022:-2][None])
    # images = processor.postprocess_pixel_values(pixel_values)
    # image = to_pil_image(images[0].detach().cpu())
    # # image.save("test.png")
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=3, shuffle=True, num_workers=0,
    # )
    # for i, batch in enumerate(train_dataloader):
    #     print(len(batch['input_ids']), i)
    # print('Done one loop on dataset')