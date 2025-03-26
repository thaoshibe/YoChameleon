import argparse
import os

import torch
import wandb
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import ChameleonForConditionalGeneration, ChameleonProcessor
from transformers.image_transforms import to_pil_image
from utils import Config


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
    parser.add_argument('--iteration', type=str, default='10')
    parser.add_argument('--finetune', action='store_true', help='Use fine-tuned model')
    parser.add_argument('--savedir', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--sks_name', type=str, default=None)
    parser.add_argument('--img_size', type=int, default=512)
    # parser.add_argument('--no_wandb', action='store_true', help='Turn off log to WanDB for debug reason')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    config_dict = yaml.safe_load(open(args.config, 'r'))
    config = Config(config_dict)
    config_test = Config(config.test)
    if args.iteration != -100:
        config_test.iteration = str(args.iteration)
    if args.exp_name is not None:
        config.exp_name = args.exp_name
    if args.sks_name is not None:
        config.sks_name = args.sks_name
    if args.savedir is not None:
        config.savedir = args.savedir

    # Initialize processor and model
    processor = ChameleonProcessor.from_pretrained(config.model_id)
    model = ChameleonForConditionalGeneration.from_pretrained(
        config.model_id, 
        torch_dtype=torch.bfloat16, 
        # low_cpu_mem_usage=True,
        # attn_implementation="flash_attention_2" #Thao: Don't know why transformers 4.46.1 doesnt support Chameleon with this option
    ).to('cuda')

    # Create personalized tokens
    latent_tokens_start_index = config.special_tokens['LATENT_TOKEN_START']
    if config.self_prompting:
        config.prefix_token = config.prefix_token*2
    prefix_tokens = [f'<reserved{latent_tokens_start_index+i}>' for i in range(config.prefix_token)]
    personalized_tokens = [f'<reserved16200>'] + prefix_tokens
    sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}."
    personalized_token_ids = processor.tokenizer.convert_tokens_to_ids(personalized_tokens)

    # Load pre-trained model parameters
    try:
        if args.finetune:
            lm_head_path = os.path.join(config.savedir, config.exp_name, config.sks_name, f'{config_test.iteration}-lmhead-ft.pt')
            lm_head = torch.load(lm_head_path, map_location='cuda')
        else:
            lm_head_path = os.path.join(config.savedir, config.exp_name, config.sks_name, f'{config_test.iteration}-lmhead.pt')
            lm_head = torch.load(lm_head_path, map_location='cuda')
        model.lm_head.weight.data[personalized_token_ids[:1]] = lm_head.to(model.lm_head.weight.data.device).to(model.dtype)[:1]
        # model.lm_head.weight.data[personalized_token_ids] = lm_head.to(model.lm_head.weight.data.device).to(model.dtype)
        # Update token embeddings
        if args.finetune:
            embedding_path = f'{config.savedir}/{config.exp_name}/{config.sks_name}/{config_test.iteration}-token-ft.pt'
        else:
            embedding_path = f'{config.savedir}/{config.exp_name}/{config.sks_name}/{config_test.iteration}-token.pt'
        
        model.get_input_embeddings().weight.data[personalized_token_ids] = torch.load(embedding_path).to(model.device).to(model.dtype)

    except:
        model_path = os.path.join(config.savedir, config.exp_name, config.sks_name, f'{config_test.iteration}-model.pt')
        state_dict = torch.load(model_path, map_location='cuda')#.to(model.dtype)
        model.model.load_state_dict(state_dict)
        print(model_path)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #
    #         Text-Only response
    #
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    prompt = f"{sks_prompt} Can you describe <reserved16200>? Answer in detail."
    inputs = processor(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=200)
    result_with_special_tokens = processor.decode(output[0], skip_special_tokens=False)
    print(result_with_special_tokens)
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #
    #         VQA response
    #
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    image = Image.open('../data/yochameleon-data/test/thao/0.png')
    prompt = f"{sks_prompt} Can you see <reserved16200> in this photo?<image><reserved08706><reserved16217><reserved16218><reserved16219><reserved16220><reserved16221><reserved16222><reserved16223><reserved16224><reserved16225><reserved16226><reserved16227><reserved16228><reserved16229><reserved16230><reserved16231><reserved16232>."

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype)
    output = model.generate(**inputs, max_new_tokens=200)
    result_with_special_tokens = processor.decode(output[0], skip_special_tokens=False)
    print(result_with_special_tokens)
    # # Save the results
    # output_dir = os.path.join(config_test.save_dir, config.exp_name)
    # os.makedirs(output_dir, exist_ok=True)

    # with open(f'{output_dir}/output.txt', 'w') as file:
    #     file.write(result_with_special_tokens + '\n')
    #     file.write('-------------------------\n')

    index = 0
    for i in tqdm(range(0, config_test.num_images, config_test.batch_size)):  # Step through by batch size
        prompt_short = config_test.prompt
        # breakpoint()
        full_prompt = f"{sks_prompt} {prompt_short}. <reserved08706><reserved16201><reserved16202><reserved16203><reserved16204><reserved16205><reserved16206><reserved16207><reserved16208><reserved16209><reserved16210><reserved16211><reserved16212><reserved16213><reserved16214><reserved16215><reserved16216>."
        full_prompt = f"{sks_prompt} {prompt_short}."
        inputs = processor([full_prompt] * config_test.batch_size, return_tensors="pt").to(model.device)
        generate_ids = model.generate(**inputs, multimodal_generation_mode="image-only", max_new_tokens=1026, do_sample=True)
        response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
        pixel_values = model.decode_image_tokens(response_ids[:, 1:-1])
        pixel_values = processor.postprocess_pixel_values(pixel_values)

        # Save generated images using the helper function
        if args.finetune:
            save_path = os.path.join(str(config_test.save_dir), config.exp_name, str(config_test.iteration)+'ft', config.sks_name)
        else:
            save_path = os.path.join(str(config_test.save_dir), config.exp_name, str(config_test.iteration), config.sks_name)

        index, image = save_generated_images(pixel_values, prompt_short, f'./generated_images/{config.exp_name}/{config.iteration}', config.sks_name, index, img_size=args.img_size)
