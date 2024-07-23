import argparse
import os
# from transformers import ChameleonForCausalLM
import shutil

import requests
import torch

from PIL import Image
from dataset import PersonalizedDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor


def get_args():
    parser = argparse.ArgumentParser(description='Chameleon')
    # model related
    parser.add_argument('--image', type=str, default='./yollava-data/train/bo/0.png', help='Path to image')
    parser.add_argument('--prompt', type=str, default="What is the color of the dog? <image>", help='Prompt')
    parser.add_argument('--model_id', type=str, default='chameleon_ckpt/chameleon-7b', help='Model ID')
    parser.add_argument('--exp_name', type=str, default='anole', help='Number of epochs')
    # personalized token related
    parser.add_argument('--sks_name', type=str, default='sks', help='Name of the personalized token')
    parser.add_argument('--prefix_token', type=int, default=16, help='Number of prefix tokens')

    # hyperparameters
    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--savedir', type=str, default='./ckpt/', help='Directory to save the model')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    log_dir = f'./runs/{args.sks_name}'
    
    if os.path.exists(log_dir):
        # Delete the directory and its contents
        shutil.rmtree(log_dir)
    writer = SummaryWriter(f'./runs/{args.sks_name}')
    save_location = f'./{args.savedir}/{args.sks_name}'
    os.makedirs(save_location, exist_ok=True)

    model_id = args.model_id
    processor = ChameleonProcessor.from_pretrained(model_id)
    model = ChameleonForConditionalGeneration.from_pretrained(model_id, device_map="auto")
    print(f'Loaded {model_id}!')

    # --- Add personalized tokens
    prefix_tokens = [f'<reserved{16301+i}>' for i in range(args.prefix_token)]
    personalized_tokens = [f'<reserved16300>']
    personalized_tokens.extend(prefix_tokens)
    sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}."
    personalized_token_ids = processor.tokenizer.convert_tokens_to_ids(personalized_tokens)

    # breakpoint()
    model.get_input_embeddings().weight.data[personalized_token_ids] = torch.load(f'./ckpt/{args.exp_name}/{args.sks_name}/{args.epoch}-token.pt').to(model.device)
    model.lm_head.weight.data[personalized_token_ids] = torch.load(f'./ckpt/{args.exp_name}/{args.sks_name}/{args.epoch}-lmhead.pt').to(model.lm_head.weight.data.device)

    image = Image.open(args.image)

    prompt = f"{sks_prompt} Can you try to describe <reserved16300> in details?"
    inputs = processor(prompt, images=None, return_tensors="pt").to(model.device)
    # breakpoint()
    # prompt = f"{sks_prompt} What is the similarity between <reserved16300> and this dog? <image>."
    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=20)
    print(processor.decode(output[0], skip_special_tokens=False))
    print('-------------------------')
    print(processor.decode(output[0], skip_special_tokens=True))
    # breakpoint()
    # for index in range(0,10):
    #     try:
    #         prompt = f"{sks_prompt}"
    #         # prompt = "<reserved16300>"
    #         inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    #         generate_ids = model.generate(**inputs, multimodal_generation_mode="image-only", max_new_tokens=1026, do_sample=True,)
    #         breakpo
    #         response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
    #         pixel_values = model.decode_image_tokens(response_ids[:, 1:-1].cpu())
    #         images = processor.postprocess_pixel_values(pixel_values.detach().cpu().numpy())
    #         images[0].save(f"./generated_images/{args.sks_name}_{index}.png")
    #         print('Done')
    #     except Exception as e:
    #         print(e)









