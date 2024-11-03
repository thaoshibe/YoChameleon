import argparse
import os
import torch

import wandb
import yaml

from PIL import Image
from tqdm import tqdm
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor
from transformers.image_transforms import to_pil_image
from utils import Config
from utils import chameleon_trim_answer

def save_generated_images(pixel_values, prompt_short, save_path, sks_name, index, img_size=256):
    """Save generated images to a specified directory."""
    for pixel_value in pixel_values:
        image = to_pil_image(pixel_value.detach().cpu())
        image = image.resize((img_size, img_size))
        prompt_short = prompt_short.replace('<reserved16200>', sks_name).replace('.', '')
        os.makedirs(save_path, exist_ok=True)
        image.save(f'{save_path}/{prompt_short}_{index}.png')
        print(f"Saved image {index} to {save_path}/{prompt_short}_{index}.png")
        index += 1
    return index, image

def get_args():
    parser = argparse.ArgumentParser(description='Your Chameleon model')
    # model related
    parser.add_argument('--config', type=str, default='./config/basic.yml')
    parser.add_argument('--image_path', type=str, default='../data/yochameleon-data/train/thao/0.png')
    parser.add_argument('--iteration', type=int, default=-100)
    parser.add_argument('--ckpt', action='store_true', help='Use fine-tuned model')
    parser.add_argument('--sks_name', type=str, default=None)
    parser.add_argument('--img_size', type=int, default=256)
    # parser.add_argument('--no_wandb', action='store_true', help='Turn off log to WanDB for debug reason')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    config_dict = yaml.safe_load(open(args.config, 'r'))
    config = Config(config_dict)

    sks_token = config.special_tokens['SKS_TOKEN']

    # Initialize processor and model
    processor = ChameleonProcessor.from_pretrained(config.model_id)
    model = ChameleonForConditionalGeneration.from_pretrained(
        config.model_id, 
        torch_dtype=torch.bfloat16, 
        # low_cpu_mem_usage=True,
        # attn_implementation="flash_attention_2"
    ).to('cuda')

    # Create personalized tokens
    prefix_tokens = [f'<reserved{16201+i}>' for i in range(config.prefix_token)]
    personalized_tokens = [sks_token] + prefix_tokens
    sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}."
    personalized_token_ids = processor.tokenizer.convert_tokens_to_ids(personalized_tokens)
    # model.resize_token_embeddings(len(processor.tokenizer))

    # Load pre-trained model parameters
    # try:
    if args.ckpt:
        lm_head_path = os.path.join(args.ckpt, f'{args.iteration}-lmhead.pt')
        lm_head = torch.load(lm_head_path, map_location='cuda')
        model.lm_head.weight.data[personalized_token_ids] = lm_head.to(model.lm_head.weight.data.device).to(model.dtype)
        embedding_path = f'{config.savedir}/{config.exp_name}/{config.sks_name}/{args.iteration}-token.pt'
        model.get_input_embeddings().weight.data[personalized_token_ids] = torch.load(embedding_path).to(model.device).to(model.dtype)

    image = Image.open(args.image_path).convert("RGB")
    question = f'{sks_prompt} Is {sks_token} in this photo? Answer "Yes" or "No" <image>'
    question = 'Can you identify a dog in this photo?<image>'

    # move to model's device and dtype
    inputs = processor(images=image, text=question, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

    # Generate and process output
    output = model.generate(**inputs, max_new_tokens=200)
    result_with_special_tokens = processor.decode(output[0], skip_special_tokens=False)
    answer = chameleon_trim_answer(result_with_special_tokens)