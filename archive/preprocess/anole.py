# ---- HuggingFace code: https://github.com/leloykun/transformers/blob/fc--anole/docs/source/en/model_doc/chameleon.md
import numpy as np

import glob
import os
import torch

import argparse

from PIL import Image
from tqdm import tqdm
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor

# model_id = '../chameleon-7b'
model_id = 'leloy/Anole-7b-v0.1-hf'
processor = ChameleonProcessor.from_pretrained(model_id)
model = ChameleonForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
)
def get_args():
    parser = argparse.ArgumentParser(description='Anole')
    # personalized token related
    parser.add_argument('--sks_name', type=str, default='sks', help='Name of the personalized token')
    parser.add_argument('--prompt', type=str, default='What can you see in this photo? <image>', help='user prompt')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    prompt = args.prompt
    print(prompt)
    image = Image.open('../yollava-data/train/bo/9.png')
    inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device)
    breakpoint()
    image_tokens = model.model.get_image_tokens(inputs['pixel_values'])
    image_tokens = image_tokens.to(image_tokens.device, image_tokens.dtype)

    recon = model.decode_image_tokens(image_tokens)
    recon_img = processor.postprocess_pixel_values(recon.detach().cpu().numpy())
    recon_img[0].save('recon.png')

    # mask out image
    seg_mask = torch.load('/mnt/localssd/code/YoChameleon/preprocess/bo-body.pt')
    seg_mask = ~seg_mask.bool().to(image_tokens.device)
    fake_img_tokens = torch.ones(image_tokens.shape)
    fake_img_tokens = fake_img_tokens*7348
    fake_img_tokens = fake_img_tokens.to(image_tokens.device, image_tokens.dtype)
    
    fake = image_tokens.masked_scatter(seg_mask, fake_img_tokens)
    fake = model.decode_image_tokens(fake)
    fake_img = processor.postprocess_pixel_values(fake.detach().cpu().numpy())
    fake_img[0].save('fake.png')

    bg = Image.open('bg.png')
    bg_inputs = processor(prompt, bg, return_tensors="pt", padding=True).to(model.device)
    bg_tokens = model.model.get_image_tokens(bg_inputs['pixel_values'])
    
    seg_mask = torch.load('/mnt/localssd/code/YoChameleon/preprocess/bo-body.pt')
    seg_mask = ~seg_mask.bool().to(image_tokens.device)
    fake = image_tokens.masked_scatter(seg_mask, bg_tokens)
    fake = model.decode_image_tokens(fake)
    fake_img = processor.postprocess_pixel_values(fake.detach().cpu().numpy())
    fake_img[0].save('fake.png')


    #---- Generate a single image
    outputs = model.generate(
        **inputs,
        # multimodal_generation_mode="image-only",
        # Note: We need to set `max_new_tokens` to 1026 since the model generates the `image_start_token` marker token first, then 1024 image tokens, and finally the `image_end_token` marker token.
        max_new_tokens=100,
        # This is important because most of the image tokens during training were for "empty" patches, so greedy decoding of image tokens will likely result in a blank image.
        # do_sample=True,
    )
    answer = processor.decode(outputs[0])
    print(answer)
    # response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
    # pixel_values = model.decode_image_tokens(response_ids[:, 1:-1].cpu())
    # images = processor.postprocess_pixel_values(pixel_values.detach().cpu().numpy())
    # images[0].save("test.png")