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

def save_generated_images(pixel_values, prompt_short, save_path, sks_name, index):
    """Save generated images to a specified directory."""
    for pixel_value in pixel_values:
        image = to_pil_image(pixel_value.detach().cpu())
        prompt_short = prompt_short.replace('<reserved16300>', sks_name).replace('.', '')
        os.makedirs(save_path, exist_ok=True)
        image.save(f'{save_path}/{prompt_short}_{index}.png')
        index += 1
    return index, image

def get_args():
    parser = argparse.ArgumentParser(description='Your Chameleon model')
    # model related
    parser.add_argument('--config', type=str, default='./config/basic.yml')
    parser.add_argument('--wandb', action='store_true', help='Turn off log to WanDB for debug reason')
    parser.add_argument('--iteration', type=int, default=-100)
    parser.add_argument('--wandb_id', type=str, default='1eyixddq')
    parser.add_argument('--exp_name', type=str, default=None)
    # parser.add_argument('--no_wandb', action='store_true', help='Turn off log to WanDB for debug reason')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    config_dict = yaml.safe_load(open(args.config, 'r'))
    config = Config(config_dict)
    config_test = Config(config.test)
    if args.iteration != -100:
        config_test.iteration = args.iteration
    if args.exp_name is not None:
        config.exp_name = args.exp_name
    # Initialize processor and model
    processor = ChameleonProcessor.from_pretrained(config.model_id)
    model = ChameleonForConditionalGeneration.from_pretrained(
        config.model_id, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    ).to('cuda')

    if args.wandb:
        wandb.init(project=config.project_name,
            name=config.exp_name,
            entity="thaoshibe-university-of-wisconsin-madison",
            config=config_dict,
            resume="must", id=args.wandb_id)
    # Create personalized tokens
    prefix_tokens = [f'<reserved{16301+i}>' for i in range(config.prefix_token)]
    personalized_tokens = [f'<reserved16300>'] + prefix_tokens
    sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:config_test.token_len])}."
    personalized_token_ids = processor.tokenizer.convert_tokens_to_ids(personalized_tokens)

    model.resize_token_embeddings(len(processor.tokenizer))

    # Load pre-trained model parameters
    try:
        lm_head = torch.load(f'{config.savedir}/{config.exp_name}/{config.sks_name}/{config_test.iteration}-lmhead.pt', map_location='cuda')
        model.lm_head.weight.data[personalized_token_ids] = lm_head.to(model.lm_head.weight.data.device).to(model.dtype)
    except:
        state_dict = torch.load(f'{config.savedir}/{config.exp_name}/{config.sks_name}/{config_test.iteration}-model.pt', map_location='cuda')#.to(model.dtype)
        model.model.load_state_dict(state_dict)

    # Update token embeddings
    embedding_path = f'{config.savedir}/{config.exp_name}/{config.sks_name}/{config_test.iteration}-token.pt'
    model.get_input_embeddings().weight.data[personalized_token_ids] = torch.load(embedding_path).to(model.device).to(model.dtype)

    # Define prompt and inputs
    # prompt = f"{sks_prompt}\nCan you describe <reserved16300>? Answer in detail."
    # inputs = processor(prompt, return_tensors="pt").to(model.device)

    # # Generate and process output
    # output = model.generate(**inputs, max_new_tokens=200)
    # result_with_special_tokens = processor.decode(output[0], skip_special_tokens=False)
    
    # # Save the results
    # output_dir = os.path.join(config_test.save_dir, config.exp_name)
    # os.makedirs(output_dir, exist_ok=True)

    # with open(f'{output_dir}/output.txt', 'w') as file:
    #     file.write(result_with_special_tokens + '\n')
    #     file.write('-------------------------\n')

    # Generate images based on prompt
    config = Config(config_dict)
    config_test = Config(config.test)
    index = 0
    for i in tqdm(range(0, config_test.num_images, config_test.batch_size)):  # Step through by batch size
        prompt_short = config_test.prompt
        full_prompt = f"{sks_prompt} {prompt_short}"
        # full_prompt = f"{prompt_short}"
        inputs = processor([full_prompt] * config_test.batch_size, return_tensors="pt").to(model.device)
        
        generate_ids = model.generate(**inputs, multimodal_generation_mode="image-only", max_new_tokens=1026, do_sample=True)
        response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
        pixel_values = model.decode_image_tokens(response_ids[:, 1:-1])
        pixel_values = processor.postprocess_pixel_values(pixel_values)

        # Save generated images using the helper function
        save_path = os.path.join(str(config_test.save_dir), config.exp_name, str(config_test.iteration))
        index, image = save_generated_images(pixel_values, prompt_short, save_path, config.sks_name, index)
        # if not args.no_wandb:
        #     wandb.log({"test_generated_images": image}, step=i)
