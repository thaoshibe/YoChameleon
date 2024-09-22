import argparse
import glob
import json
import numpy as np
import os
import requests

from tqdm import tqdm

def process_conversation(image_path, conversations):
    # Randomly assign a conversation for each images
    image_paths = glob.glob(os.path.join(image_path, "*.*"))
    data = []
    questions = conversations['questions']
    answers = conversations['answers']

    for image_path in tqdm(image_paths):
        question = np.random.choice(questions, 1)[0]
        answer = np.random.choice(answers, 1)[0]
        conv = [
            {
            "from": "human",
            "value": f"{question}<image>"
            },
            {
            "from": "bot",
            "value": f"{answer}"
            },
        ]
        data.append({
            "image": [image_path],
            "conversations": conv
        })
    return data


def main():
    parser = argparse.ArgumentParser(description='Process images in a folder and prompt for GPT-4 API.')
    parser.add_argument('--positive_image_folder', type=str, required=True, help='Path to the folder positive containing input images.')
    parser.add_argument('--negative_image_folder', type=str, required=True, help='Path to the folder positive containing input images.')
    parser.add_argument('--prompt_file_path', type=str, default='./template-answer/recognition.json')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file.')
    args = parser.parse_args()

    # Read the prompt from the .txt file
    with open(args.prompt_file_path, 'r') as file:
        template = json.load(file)  
    # print(f"Prompt: {prompt}")
    for conv in template:
        if conv['type'] == 'positive':
            positive_conv = conv['conversations']
        elif conv['type'] == 'negative':
            negative_conv = conv['conversations']

    total_conv = []
    negative_conv = process_conversation(args.negative_image_folder, negative_conv)
    positive_conv = process_conversation(args.positive_image_folder, positive_conv)
    total_conv.extend(negative_conv)
    total_conv.extend(positive_conv)
    
    with open(args.output_file, 'w') as file:
        json.dump(total_conv, file)
    print('JSON file created successfully! at', args.output_file)

if __name__ == "__main__":
    main()

