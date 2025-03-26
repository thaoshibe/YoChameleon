import argparse
import glob
import os
import random

import torch
import wandb
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import ChameleonForConditionalGeneration, ChameleonProcessor
from transformers.image_transforms import to_pil_image

SUBJECT_NAMES = ["bo", "duck-banana", "marie-cat", "pusheen-cup", "brown-duck", "dug", "mydieu", "shiba-black",
    "tokyo-keyboard", "butin", "elephant", "neurips-cup",
    "shiba-gray", "toodles-galore", "cat-cup", "fire",
    "nha-tho-hanoi", "shiba-sleep", "viruss", "chua-thien-mu",
    "henry", "nha-tho-hcm", "shiba-yellow", "water",
    "ciin", "khanhvy", "oong", "thao",
    "willinvietnam", "denisdang", "lamb", "phuc-map",
    "thap-but", "yellow-duck", "dragon", "mam",
    "pig-cup", "thap-cham", "yuheng", "thuytien",
]

# SUBJECT_NAMES = ["chua-thien-mu",
# ]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_prompt", type=bool, default=True)
    parser.add_argument("--text_prompt", type=str, default="Generater a photo of a person with long, dark hair, often seen wearing stylish, comfortable outfits.")
    # parser.add_argument("--fake_folder", type=str, default=True)
    parser.add_argument("--input_folder", type=str, default='/mnt/localssd/code/data/yochameleon-data/train/')
    parser.add_argument("--save_folder", type=str, default='/sensei-fs/users/thaon/generated_images/chameleon/image_prompt/')
    parser.add_argument("--number_of_image", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    # Load the model and tokenizer
    processor = ChameleonProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
    processor = ChameleonProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
    model = ChameleonForConditionalGeneration.from_pretrained(
        "leloy/Anole-7b-v0.1-hf", 
        torch_dtype=torch.bfloat16, 
        # low_cpu_mem_usage=True,
        # attn_implementation="flash_attention_2" #Thao: Don't know why transformers 4.46.1 doesnt support Chameleon with this option
    ).to('cuda')
    subjectnames = SUBJECT_NAMES[args.start:args.end]
    if args.image_prompt:
        image_folders = [os.path.join(args.input_folder, subjectname) for subjectname in subjectnames]
    else:
        caption_file = 'subject-detailed-captions.json'
        with open(caption_file, 'r') as f:
            captions = json.load(f)
    os.makedirs(args.save_folder, exist_ok=True)
    saving_index = 0
    for index, subjectname in enumerate(tqdm(subjectnames)):
        image_folder = image_folders[index]
        image_chunk = '<image>'
        image_chunk = image_chunk* args.number_of_image
        full_prompt = f"This is a subject {image_chunk} Generate photo of this subject with cherry blooms"
        image_paths = glob.glob(os.path.join(image_folder, "*.png"))[:args.number_of_image]
        image = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        inputs = processor(text=[full_prompt]*10, images=[image]*10, return_tensors="pt").to(model.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype)
        save_path = os.path.join(args.save_folder, str(args.number_of_image), subjectname)
        os.makedirs(save_path, exist_ok=True)
        for i in tqdm(range(0, 20, 10)):
            generate_ids = model.generate(**inputs, multimodal_generation_mode="image-only", max_new_tokens=1026, do_sample=True)
            response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
            pixel_values = model.decode_image_tokens(response_ids[:, 1:-1])
            pixel_values = processor.postprocess_pixel_values(pixel_values)

            for pixel_value in pixel_values:
                image = to_pil_image(pixel_value.detach().cpu())
                image = image.resize((512, 512))
                image.save(os.path.join("./anole", f"{saving_index}.png"))
                saving_index += 1
                print('Saved image:', os.path.join("./anole", f"{saving_index}.png"))
            #     os.makedirs(save_path, exist_ok=True)
            #     image.save(f'{save_path}/{prompt_short}_{index}.png')
            #     print(f"Saved image {index} to {save_path}/{prompt_short}_{index}.png")
            #     index += 1
            # save_location = os.path.join(args.save_folder, f"{index}.png")
            # image.save(save_location)
