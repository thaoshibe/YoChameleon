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
from transformers.image_transforms import to_pil_image

def get_args():
    parser = argparse.ArgumentParser(description='Chameleon')
    # model related
    parser.add_argument('--prompt', type=str, default="What is the color of the dog?<image>", help='Prompt')
    parser.add_argument('--model_id', type=str, default='leloy/Anole-7b-v0.1-hf', help='Model ID')
    parser.add_argument('--exp_name', type=str, default='anole', help='Number of epochs')
    parser.add_argument('--sks_name', type=str, default='sks', help='Name of the personalized token')
    parser.add_argument('--prefix_token', type=int, default=16, help='Number of prefix tokens')
    parser.add_argument('--token_len', type=int, default=16, help='Number of used tokens')

    # hyperparameters
    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--savedir', type=str, default='/sensei-fs/users/thaon/ckpt', help='Directory to save the model')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # save_location = f'./{args.savedir}/{args.sks_name}'
    # os.makedirs(save_location, exist_ok=True)

    model_id = args.model_id
    processor = ChameleonProcessor.from_pretrained(model_id)
    model = ChameleonForConditionalGeneration.from_pretrained(model_id, device_map="cuda")#, torch_dtype=torch.float16)
    print(f'Loaded {model_id}!')

    # --- Add personalized tokens
    prefix_tokens = [f'<reserved{16301+i}>' for i in range(args.prefix_token)]
    personalized_tokens = [f'<reserved16300>']
    personalized_tokens.extend(prefix_tokens)
    sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:args.token_len])}."
    personalized_token_ids = processor.tokenizer.convert_tokens_to_ids(personalized_tokens)
    model.model.resize_token_embeddings(len(processor.tokenizer))
    try:
        lm_head = torch.load(f'{args.savedir}/{args.exp_name}/{args.sks_name}/{args.epoch}-lmhead.pt', map_location='cuda').to(model.lm_head.weight.data.device)
        lm_head = lm_head.to(model.dtype)
        model.lm_head.weight.data[personalized_token_ids] = lm_head
    except:
        state_dict = torch.load(f'{args.savedir}/{args.exp_name}/{args.sks_name}/{args.epoch}-model.pt')
        model.model.load_state_dict(state_dict)
        
    # image = Image.open(args.image)
    model.get_input_embeddings().weight.data[personalized_token_ids] = torch.load(f'{args.savedir}/{args.exp_name}/{args.sks_name}/{args.epoch}-token.pt').to(model.device).to(model.dtype)
    prompt = f"{sks_prompt}\nCan you describe <reserved16300>? Answer in details."
    # prompt = f"{sks_prompt} Is <reserved16300> in this photo?"
    inputs = processor(prompt, images=None, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=200)
    result_with_special_tokens = processor.decode(output[0], skip_special_tokens=False)
    print(processor.decode(output[0], skip_special_tokens=False))

    print('-------------------------')
    os.makedirs(f"/sensei-fs/users/thaon/generated_images/{args.exp_name}", exist_ok=True)
    print(processor.decode(output[0], skip_special_tokens=True))
    with open(f'/sensei-fs/users/thaon/generated_images/{args.exp_name}/output.txt', 'w') as file:
        file.write(result_with_special_tokens + '\n')
        file.write('-------------------------\n')
        # file.write(result_without_special_tokens + '\n')

    for index in tqdm(range(0, 100)):
        # prompt_short = 'A photo of <reserved16300> in a sunflower field'
        prompt_short = args.prompt
        prompt = f"{sks_prompt}\n{prompt_short}"
        inputs = processor(prompt, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**inputs,
            multimodal_generation_mode="image-only",
            max_new_tokens=1026,
            do_sample=True,)
        response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
        pixel_values = model.decode_image_tokens(response_ids[:, 1:-1])
        pixel_values = processor.postprocess_pixel_values(pixel_values)
        image = to_pil_image(pixel_values[0].detach().cpu())
        image.save('test.png')
        prompt_short = prompt_short.replace('<reserved16300>', f'{args.sks_name}')
        os.makedirs(f"/sensei-fs/users/thaon/generated_images/{args.exp_name}/{args.epoch}/{args.token_len-1}", exist_ok=True)
        image.save(f"/sensei-fs/users/thaon/generated_images/{args.exp_name}/{args.epoch}/{args.token_len-1}/{prompt_short}_{index}.png")
        print('Done')
        # except Exception as e:
        #     print(e)
        # torch.cuda.empty_cache()








