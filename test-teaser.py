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

def save_generated_images(pixel_values, prompt_short, save_path, sks_name, index, img_size=512):
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
        print(f'Loading token embeddings from {config.savedir}/{config.exp_name}/{config.sks_name}/{config_test.iteration}-token.pt')
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
    import json
    with open('baselines/text_qa_soft_2.json') as f:
        data = json.load(f)

    total_count = 0
    count_correct = 0
    for data_key in tqdm(data.keys()):
        if data_key == args.sks_name:
            info = data[data_key]

            for index in info.keys():
                prompt = info[index]['question'].replace('sks', '<reserved16200>')
                options = info[index]['option']
                option_A = options['A']
                option_B = options['B']
                # question = caption + ' ' + prompt + f'{option_A} or {option_B}?'
                question = prompt + f' {option_A} or {option_B}?'
                correct_answer = options[info[index]['correct_answer']]
                inputs = processor(text=sks_prompt + question, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
                output = model.generate(**inputs, max_new_tokens=10)
                answer = processor.decode(output[0], skip_special_tokens=False)
                # breakpoint()
                answer = answer.replace(sks_prompt, '')
                answer = answer.replace(question, '')
                print(prompt)
                print(correct_answer)
                print(answer)
                if correct_answer.lower() in answer.lower():
                    count_correct += 1
                total_count += 1
            print('Current accuracy:', count_correct/total_count)

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
    # # image = Image.open('../data/yochameleon-data/test/thao/0.png')
    # image = Image.open('./next.jpg')
    # prompt = f"{sks_prompt} Describe this image in details<image><reserved08706><reserved16217><reserved16218><reserved16219><reserved16220><reserved16221><reserved16222><reserved16223><reserved16224><reserved16225><reserved16226><reserved16227><reserved16228><reserved16229><reserved16230><reserved16231><reserved16232>."
    # prompt = f"{sks_prompt} Describe this image in details<image>"
    # inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    # inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype)
    # output = model.generate(**inputs, max_new_tokens=200)
    # result_with_special_tokens = processor.decode(output[0], skip_special_tokens=False)
    # print(result_with_special_tokens)
    # breakpoint()
    # image = Image.open('./image.jpg')
    # prompt = f"{sks_prompt} Describe what is next to <reserved16200> in this photo?<image><reserved08706><reserved16217><reserved16218><reserved16219><reserved16220><reserved16221><reserved16222><reserved16223><reserved16224><reserved16225><reserved16226><reserved16227><reserved16228><reserved16229><reserved16230><reserved16231><reserved16232>."
    
    # inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    # inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype)
    # output = model.generate(**inputs, max_new_tokens=200)
    # result_with_special_tokens = processor.decode(output[0], skip_special_tokens=False)
    # print(result_with_special_tokens)

    # # Save the results
    # output_dir = os.path.join(config_test.save_dir, config.exp_name)
    # os.makedirs(output_dir, exist_ok=True)

    # with open(f'{output_dir}/output.txt', 'w') as file:
    #     file.write(result_with_special_tokens + '\n')
    #     file.write('-------------------------\n')
    # templates = [
    #     "A photo of sks surrounded by a vibrant field of sunflowers during the day.",
    #     "A photo of sks illuminated by the glow of the moon during the night.",
    #     "A photo of sks during a beautiful sunset with rich, warm colors in the sky.",
    #     "A photo of sks at dawn with mist rolling in from the nearby forest.",
    #     "A photo of sks surrounded by snow-covered trees on a cold winter's morning.",
    #     "A photo of sks framed by autumn leaves with hues of orange, red, and yellow.",
    #     "A photo of sks at high noon with clear blue skies and no clouds.",
    #     "A photo of sks during a thunderstorm with dark storm clouds in the background.",
    #     "A photo of sks with bustling city life around it in the early evening.",
    #     "A photo of sks partially covered in fog with distant mountains in the background.",
    #     "A photo of sks under a starry night sky with the Milky Way visible.",
    #     "A photo of sks in the middle of a bustling festival, with people celebrating.",
    #     "A photo of sks surrounded by a wildflower meadow during spring.",
    #     "A photo of sks reflected in a calm lake during golden hour.",
    #     "A photo of sks with a rainbow in the sky after a rainstorm.",
    #     "A photo of sks viewed from a hot air balloon during sunrise.",
    #     "A photo of sks during a full moon with its silhouette against the glowing orb.",
    #     "A photo of sks with mist rising from the surrounding hills at dawn.",
    #     "A photo of sks set against a deep blue sky with a few scattered clouds.",
    #     "A photo of sks during a clear, crisp morning with no haze in the air.",
    #     "A photo of sks illuminated by bright city lights at night.",
    #     "A photo of sks covered in colorful lights during a major celebration or event.",
    #     "A photo of sks surrounded by dense jungle vegetation with a few sun rays shining through.",
    #     "A photo of sks under a dramatic sky with intense clouds during a storm.",
    #     "A photo of sks at twilight, with the sky transitioning from orange to purple.",
    #     "A photo of sks perched on a cliff, overlooking the vast ocean in the distance.",
    #     "A photo of sks with a snowy mountain backdrop on a bright, clear day.",
    #     "A photo of sks during a rainy day, with raindrops splashing on the ground.",
    #     "A photo of sks with a colorful sunrise painting the sky with soft pastels.",
    #     "A photo of sks at night with the lights of the city skyline illuminating the horizon.",
    #     "A photo of sks surrounded by dense mist with only part of it visible.",
    #     "A photo of sks as the last light of the day fades into a dark, tranquil night."
    # ]
    templates = [
    "A photo of sks with a bouquet of sunflowers in background smiling brightly.",
    ]
    # templates = [
    #     "A portrait of sks standing on a mountain peak, with a vast landscape stretching in the background during sunset.",
    #     "A candid photo of sks walking through a field of wildflowers on a sunny afternoon.",
    #     "A black and white photo of sks in the middle of a bustling city street, with the crowd blurred around them.",
    #     "A photo of sks sitting on a park bench under a cherry blossom tree during springtime.",
    #     "A photo of sks with a bright smile, surrounded by golden autumn leaves on a breezy day.",
    #     "A photo of sks gazing up at the night sky filled with stars, standing on a quiet hilltop.",
    #     "A photo of sks riding a bicycle through a rain-soaked street, with reflections from the puddles.",
    #     "A silhouette of sks against the backdrop of a fiery sunset at the beach.",
    #     "A photo of sks walking through a snow-covered forest, with soft snowflakes falling around.",
    #     "A photo of sks enjoying a peaceful moment on a dock by the lake, with the calm water reflecting the sky.",
    #     "A candid photo of sks laughing with friends, sitting on a grassy hill during a summer afternoon.",
    #     "A photo of sks standing on a rooftop, looking out over the city skyline at night.",
    #     "A close-up photo of sks reading a book in a cozy café with soft lighting in the evening.",
    #     "A photo of sks during a morning jog along the beach with the sun rising behind them.",
    #     "A photo of sks standing alone in a dense fog in the early morning, with only their silhouette visible.",
    #     "A photo of sks during a peaceful meditation session in a serene forest with dappled sunlight.",
    #     "A photo of sks sitting at the edge of a cliff, overlooking a vast valley at sunrise.",
    #     "A photo of sks standing confidently in front of an ancient monument, with the sky painted in hues of purple and orange.",
    #     "A photo of sks standing on a bridge, gazing out over the flowing river with autumn foliage around.",
    #     "A photo of sks walking through a foggy street at dawn, with streetlights casting a soft glow.",
    #     "A portrait of sks on a quiet mountain trail, surrounded by dense pine trees during a misty morning.",
    #     "A photo of sks in a busy marketplace, with vibrant colors and people bustling around.",
    #     "A photo of sks wearing a raincoat, standing under an umbrella in the middle of a rainstorm.",
    #     "A photo of sks standing tall in front of a modern building, with clean lines and glass windows.",
    #     "A photo of sks practicing yoga at sunrise on a secluded beach, with the waves gently crashing in the background.",
    #     "A photo of sks walking on a cobblestone street, with old architecture lining the sides during the golden hour.",
    #     "A candid photo of sks and a dog playing together in a sunlit park.",
    #     "A photo of sks sitting on a wooden bench by the lake, watching the ripples in the water as dusk falls.",
    #     "A photo of sks dancing under the bright lights of a lively street festival at night.",
    #     "A photo of sks in a field of tall grass, the sun casting a warm, golden glow around them.",
    #     "A photo of sks with their arms raised in triumph, standing on a rocky outcrop overlooking the ocean during sunset.",
    #     "A photo of sks surrounded by friends around a campfire, with the night sky full of stars above them."
    # ]

    for prompt_short in tqdm(templates):
        index = 0
        prompt_short = prompt_short.replace('sks', '<reserved16200>')
        for i in range(0, 50, 16):  # Step through by batch size
            # prompt_short = config_test.prompt
            full_prompt = f"{sks_prompt} {prompt_short}<reserved08706><reserved16201><reserved16202><reserved16203><reserved16204><reserved16205><reserved16206><reserved16207><reserved16208><reserved16209><reserved16210><reserved16211><reserved16212><reserved16213><reserved16214><reserved16215><reserved16216>."
            # full_prompt = f"{sks_prompt} {prompt_short}."
            inputs = processor([full_prompt] * 16, return_tensors="pt").to(model.device)
            generate_ids = model.generate(**inputs, multimodal_generation_mode="image-only", max_new_tokens=1026, do_sample=True)
            response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
            pixel_values = model.decode_image_tokens(response_ids[:, 1:-1])
            pixel_values = processor.postprocess_pixel_values(pixel_values)

            # Save generated images using the helper function
            if args.finetune:
                save_path = os.path.join(str(config_test.save_dir), config.exp_name, str(config_test.iteration)+'ft', config.sks_name)
            else:
                save_path = os.path.join(str(config_test.save_dir), config.exp_name, str(config_test.iteration), config.sks_name)

            index, image = save_generated_images(pixel_values, prompt_short, './generated_images_bo', config.sks_name, index, img_size=args.img_size)