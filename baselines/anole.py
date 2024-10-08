import torch

import argparse

import os

import glob
import random

from PIL import Image
from tqdm import tqdm
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor
from transformers.image_transforms import to_pil_image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_prompt", type=bool, default=True)
    parser.add_argument("--text_prompt", type=str, default="Generater a photo of a person with long, dark hair, often seen wearing stylish, comfortable outfits.")
    # parser.add_argument("--fake_folder", type=str, default=True)
    parser.add_argument("--input_folder", type=str, default='/mnt/localssd/code/data/yollava-data/train/thao/')
    parser.add_argument("--save_folder", type=str, default='/sensei-fs/users/thaon/generated_images/chameleon/1')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    # Load the model and tokenizer
    processor = ChameleonProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
    model = ChameleonForConditionalGeneration.from_pretrained(
        "leloy/Anole-7b-v0.1-hf",
        device_map="cuda",
        # torch_dtype=torch.bfloat16,
    )
    if args.image_prompt:
        list_inputs = []
        prompt = 'Generate another photo of this person <image>.'
        list_images = glob.glob(os.path.join(args.input_folder, '*.png'))[:4]
        for image_path in list_images:
            image = Image.open(image_path)
            inputs = processor(prompt, image, return_tensors="pt").to(model.device)
            list_inputs.append(inputs)
    else:
        prompt = args.text_prompt
        inputs = processor(prompt,
            padding=True,
            return_tensors="pt"
            ).to(model.device, dtype=model.dtype)
    os.makedirs(args.save_folder, exist_ok=True)

    for index in tqdm(range(args.start,args.end)):
        # Generate discrete image tokens
        inputs = random.choice(list_inputs) if args.image_prompt else inputs
        generate_ids = model.generate(
            **inputs,
            multimodal_generation_mode="image-only",
            # Note: We need to set `max_new_tokens` to 1026 since the model generates the `image_start_token` marker token first, then 1024 image tokens, and finally the `image_end_token` marker token.
            max_new_tokens=1026,
            # This is important because most of the image tokens during training were for "empty" patches, so greedy decoding of image tokens will likely result in a blank image.
            do_sample=True,
        )

        # Only keep the tokens from the response
        response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]

        # Decode the generated image tokens
        pixel_values = model.decode_image_tokens(response_ids[:, 1:-1])
        images = processor.postprocess_pixel_values(pixel_values)
        image = to_pil_image(images[0].detach().cpu())
        # Save the image
        save_location = os.path.join(args.save_folder, f"{index}.png")
        image.save(save_location)