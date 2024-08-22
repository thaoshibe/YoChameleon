import argparse
import os, glob
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
from transformers.image_transforms import to_pil_image

def get_args():
    parser = argparse.ArgumentParser(description='Chameleon')
    # model related
    parser.add_argument('--image', type=str, default='./yollava-data/train/mam/3.png', help='Path to image')
    parser.add_argument('--prompt', type=str, default="What is the color of the dog?<image>", help='Prompt')
    parser.add_argument('--model_id', type=str, default='leloy/Anole-7b-v0.1-hf', help='Model ID')
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
    # Uncomment this for learned token
    try:
        model.lm_head.weight.data[personalized_token_ids] = torch.load(f'./ckpt/{args.exp_name}/{args.sks_name}/{args.epoch}-lmhead.pt').to(model.lm_head.weight.data.device)
    except:
        state_dict = torch.load(f'./ckpt/{args.exp_name}/{args.sks_name}/{args.epoch}-model.pt')
        model.model.load_state_dict(state_dict)
    model.get_input_embeddings().weight.data[personalized_token_ids] = torch.load(f'./ckpt/{args.exp_name}/{args.sks_name}/{args.epoch}-token.pt').to(model.device)
    # Uncomment this for original prompting
    # sks_prompt = "<reserved16300> is a Shiba Inu with white and orange coat."
    
    prompt = f"{sks_prompt}\nIs <reserved16300> in this photo? Answer with Yes or No.<image><reserved08706>"
    
    # image = Image.open(args.image)
    list_positive = glob.glob('/mnt/localssd/code/YoChameleon/yollava-data/test/bo/*.png')
    count_yes = 0
    for image_path in list_positive:
        # breakpoint()
        image = Image.open(image_path)
        inputs = processor(prompt, images=image, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=20)
        result_with_special_tokens = processor.decode(output[0], skip_special_tokens=False)
        result_with_special_tokens = result_with_special_tokens.split('<reserved08706>')[-1]
        answer = result_with_special_tokens.split(' ')[0]
        print(image_path, result_with_special_tokens)
        if ('yes' in answer) or ("Yes" in answer):
            count_yes += 1
    print('Accuracy positive:', count_yes/len(list_positive))
    list_negative = glob.glob('/mnt/localssd/code/YoChameleon/yollava-data/test/mam/*.png')
    count_no = 0
    for image_path in list_negative:
        # breakpoint()
        image = Image.open(image_path)
        inputs = processor(prompt, images=image, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=20)
        result_with_special_tokens = processor.decode(output[0], skip_special_tokens=False)
        # print(result_with_special_tokens)
        result_with_special_tokens = result_with_special_tokens.split('<reserved08706>')[-1]
        answer = result_with_special_tokens.split(' ')[0]
        print(image_path, result_with_special_tokens)
        if ('no' in result_with_special_tokens) or ("No" in result_with_special_tokens) or ("not" in result_with_special_tokens):
            # print(image_path, result_with_special_tokens)
            count_no += 1
    print('Accuracy negative:', count_no/len(list_negative))
