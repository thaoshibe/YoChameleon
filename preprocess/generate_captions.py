import argparse
import json
import os

import torch

from PIL import Image
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor

def get_args():
    parser = argparse.ArgumentParser(description='Chameleon')
    parser.add_argument('--image_folder', type=str, default='../yollava-data/train/mam', help='Path to the image folder')
    parser.add_argument('--prompt', type=str, default="Describe this image in details <image>", help='Prompt')
    parser.add_argument('--model_id', type=str, default='leloy/Anole-7b-v0.1-hf', help='Model ID')
    parser.add_argument('--output_file', type=str, default='output.json', help='Path to save the output JSON file')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    model_id = args.model_id
    image_folder = args.image_folder
    output_file = args.output_file

    processor = ChameleonProcessor.from_pretrained(model_id)
    model = ChameleonForConditionalGeneration.from_pretrained(model_id, device_map="auto")
    prompt = args.prompt
    print(f'Loaded {model_id}!')
    results = {}

    # Iterate through each image in the folder
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if image_path.endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(image_path)
            inputs = processor(prompt, image, return_tensors="pt").to(model.device)

            # Autoregressively complete prompt
            output = model.generate(**inputs, max_new_tokens=200)
            answer = processor.decode(output[0], skip_special_tokens=True)

            # Save the result
            results[image_path] = answer
            print(f"Processed {image_path}: {answer}")

    # Save results to a JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")
