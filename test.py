import argparse
import os
import torch

from PIL import Image
from tqdm import tqdm
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor
from transformers.image_transforms import to_pil_image

def get_args():
    parser = argparse.ArgumentParser(description='Chameleon Model for Image Captioning')
    
    # Model-related arguments
    parser.add_argument('--prompt', type=str, default="What is the color of the dog?<image>", help='Prompt')
    parser.add_argument('--model_id', type=str, default='leloy/Anole-7b-v0.1-hf', help='Model ID')
    parser.add_argument('--exp_name', type=str, default='anole', help='Experiment name')
    parser.add_argument('--sks_name', type=str, default='sks', help='Personalized token name')
    parser.add_argument('--prefix_token', type=int, default=16, help='Number of prefix tokens')
    parser.add_argument('--token_len', type=int, default=16, help='Number of used tokens')
    
    # Hyperparameters
    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of batchsize')
    parser.add_argument('--num_images', type=int, default=100, help='Number of batchsize')
    parser.add_argument('--savedir', type=str, default='/sensei-fs/users/thaon/ckpt', help='Directory to save the model')
    
    return parser.parse_args()

def save_generated_images(pixel_values, prompt_short, save_path, sks_name, index):
    """Save generated images to a specified directory."""
    for pixel_value in pixel_values:
        image = to_pil_image(pixel_value.detach().cpu())
        prompt_short = prompt_short.replace('<reserved16300>', sks_name).replace('.', '')
        os.makedirs(save_path, exist_ok=True)
        image.save(f'{save_path}/{prompt_short}_{index}.png')
        index += 1
    return index

def main():
    args = get_args()
    
    # Initialize processor and model
    processor = ChameleonProcessor.from_pretrained(args.model_id)
    model = ChameleonForConditionalGeneration.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    ).to('cuda')

    # Create personalized tokens
    prefix_tokens = [f'<reserved{16301+i}>' for i in range(args.prefix_token)]
    personalized_tokens = [f'<reserved16300>'] + prefix_tokens
    sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:args.token_len])}."
    personalized_token_ids = processor.tokenizer.convert_tokens_to_ids(personalized_tokens)

    model.resize_token_embeddings(len(processor.tokenizer))

    # Load pre-trained model parameters
    try:
        lm_head = torch.load(f'{args.savedir}/{args.exp_name}/{args.sks_name}/{args.epoch}-lmhead.pt', map_location='cuda')
        model.lm_head.weight.data[personalized_token_ids] = lm_head.to(model.lm_head.weight.data.device).to(model.dtype)
    except:
        state_dict = torch.load(f'{args.savedir}/{args.exp_name}/{args.sks_name}/{args.epoch}-model.pt')#.to(model.dtype)
        model.load_state_dict(state_dict)

    # Update token embeddings
    embedding_path = f'{args.savedir}/{args.exp_name}/{args.sks_name}/{args.epoch}-token.pt'
    model.get_input_embeddings().weight.data[personalized_token_ids] = torch.load(embedding_path).to(model.device).to(model.dtype)

    # Define prompt and inputs
    prompt = f"{sks_prompt}\nCan you describe <reserved16300>? Answer in detail."
    inputs = processor(prompt, return_tensors="pt").to(model.device)

    # Generate and process output
    output = model.generate(**inputs, max_new_tokens=200)
    result_with_special_tokens = processor.decode(output[0], skip_special_tokens=False)
    
    # Save the results
    output_dir = f"/sensei-fs/users/thaon/generated_images/{args.exp_name}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f'{output_dir}/output.txt', 'w') as file:
        file.write(result_with_special_tokens + '\n')
        file.write('-------------------------\n')

    # Generate images based on prompt
    index = 0
    for i in tqdm(range(0, args.num_images, args.batch_size)):  # Step through by batch size
        prompt_short = args.prompt
        full_prompt = f"{sks_prompt} {prompt_short}"
        inputs = processor([full_prompt] * args.batch_size, return_tensors="pt").to(model.device)
        
        generate_ids = model.generate(**inputs, multimodal_generation_mode="image-only", max_new_tokens=1026, do_sample=True)
        response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
        pixel_values = model.decode_image_tokens(response_ids[:, 1:-1])
        pixel_values = processor.postprocess_pixel_values(pixel_values)

        # Save generated images using the helper function
        save_path = f"{output_dir}/{args.epoch}/{args.token_len-1}"
        index = save_generated_images(pixel_values, prompt_short, save_path, args.sks_name, index)


if __name__ == '__main__':
    main()
