import argparse
import json
import os

import torch

from PIL import Image
from tqdm import tqdm
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor

def get_args():
    parser = argparse.ArgumentParser(description='Chameleon')
    parser.add_argument('--image_folder', type=str, default='../yollava-data/train/bo', help='Path to the image folder')
    parser.add_argument('--prompt', type=str, default="You are seeing a photo of a dog. Describe this image in details <image>", help='Prompt')
    parser.add_argument('--model_id', type=str, default='leloy/Anole-7b-v0.1-hf', help='Model ID')
    parser.add_argument('--output_file', type=str, default='output.json', help='Path to save the output JSON file')
    return parser.parse_args()

def generate_responses_and_create_json(image_folder, prompt, model_id, output_file):
    # Load model and processor
    processor = ChameleonProcessor.from_pretrained(model_id)
    model = ChameleonForConditionalGeneration.from_pretrained(model_id, device_map="auto")
    print(f'Loaded {model_id}!')

    # Initialize the list to store the JSON structure
    data = []
    # if not os.path.exists(output_file):
    #     os.makedirs(output_file, exist_ok=True)
    # Iterate through each image in the folder
    for image_name in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_name)
        if image_path.endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(image_path)
            inputs = processor(prompt, image, return_tensors="pt").to(model.device)

            # Autoregressively complete prompt
            output = model.generate(**inputs, max_new_tokens=200)
            prompt_idx = inputs['input_ids'].shape[-1]
            answer = processor.decode(output[0, prompt_idx:], skip_special_tokens=True)
            answer = answer.replace('The dog', '<bo>')
            answer = answer.replace('a dog', '<bo>')
            print(f"Processed {image_path}: {answer}")
            # Create the JSON structure for the current image
            qa_list = [
                {
                    "from": "human",
                    "value": answer
                },
                {
                    "from": "bot",
                    "value": '<image>'
                },
            ]

            data.append({
                "image": [image_name],
                "conversations": qa_list
            })
            # breakpoint()

        # Save the result to a JSON file
        with open(output_file, 'w') as f:
            json.dump(data, f)

    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    args = get_args()
    generate_responses_and_create_json(args.image_folder, args.prompt, args.model_id, args.output_file)
