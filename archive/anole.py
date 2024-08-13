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
    parser.add_argument('--prompt', type=str, default='Here is a photo of my cat. Can you generate another photo of him?<image>', help='user prompt')
    return parser.parse_args()

if __name__ == '__main__':
    # Prepare a prompt
    # prompt = "A photo of a Shiba Inu."
    args = get_args()
    prompt = args.prompt
    print(prompt)
    # image = Image.open("./yollava-data/train/mam/3.png")
    # # Preprocess the prompt
    prompt = '''<<mam> is a cute gray cat with large, round eyes and a curious expression.
        Generate a photo of <mam>.
        '''
    inputs = processor(prompt, return_tensors="pt", padding=True).to(model.device)

    #---- Generate a single image
    generate_ids = model.generate(
        **inputs,
        multimodal_generation_mode="image-only",
        # Note: We need to set `max_new_tokens` to 1026 since the model generates the `image_start_token` marker token first, then 1024 image tokens, and finally the `image_end_token` marker token.
        max_new_tokens=1026,
        # This is important because most of the image tokens during training were for "empty" patches, so greedy decoding of image tokens will likely result in a blank image.
        do_sample=True,
    )
    response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
    pixel_values = model.decode_image_tokens(response_ids[:, 1:-1].cpu())
    images = processor.postprocess_pixel_values(pixel_values.detach())
    images[0].save("test.png")
    # images[1].save("test1.png")
    # images[2].save("test2.png")

    #---- Generate a bunch of image
    # list_imgs = glob.glob(f"./yollava-data/train/{args.sks_name}/*.png")
    # print('Found:', len(list_imgs), 'images')
    # for img_path in list_imgs:
    #     num_img = 10
    #     img_index = img_path.split('/')[-1].split('.')[0]
    #     image = Image.open(img_path)
    #     # Preprocess the prompt
    #     inputs = processor(prompt, image, return_tensors="pt", padding=True).to(model.device)
    #     concat_img = []
    #     for index in tqdm(range(num_img)):
    #         try:
    #             generate_ids = model.generate(
    #                 **inputs,
    #                 multimodal_generation_mode="image-only",
    #                 # Note: We need to set `max_new_tokens` to 1026 since the model generates the `image_start_token` marker token first, then 1024 image tokens, and finally the `image_end_token` marker token.
    #                 max_new_tokens=1026,
    #                 # This is important because most of the image tokens during training were for "empty" patches, so greedy decoding of image tokens will likely result in a blank image.
    #                 do_sample=True,
    #             )
    #             response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
    #             pixel_values = model.decode_image_tokens(response_ids[:, 1:-1].cpu())
    #             images = processor.postprocess_pixel_values(pixel_values.detach().cpu().numpy())
    #             concat_img.append(images[0])
    #             torch.cuda.empty_cache()
    #         except Exception as e:
    #             print(e)
    #             continue
    #     Image.fromarray(np.concatenate(concat_img, axis=1)).save(f"./generated_images/nam_{args.sks_name}_{img_index}.png")
        # images[0].save("test.png")