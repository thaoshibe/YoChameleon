import argparse
import glob
import json
import numpy as np
import os
import requests

from tqdm import tqdm

def process_conversation(image_path, conversations, limit=None, match_number=None):
    # Randomly assign a conversation for each images
    image_paths = glob.glob(os.path.join(image_path, "*.*"))
    if limit is not None:
        image_paths = image_paths[:limit]
    if match_number is not None:
        # duplicate the number of positive images to match the number of negative
        image_paths = image_paths * (match_number // len(image_paths) + 1)
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
    parser.add_argument('--positive_image_folder', type=str, required=True, help='Path to the positive image folder')
    parser.add_argument('--negative_image_folder', type=str, required=True, help='Path to the negative image folder (hard negative)')
    parser.add_argument('--random_negative_image_folder', type=str, required=True, help='Path to the random negative image folder (easy negative)')
    parser.add_argument('--prompt_file_path', type=str, default='./create_training_data/conversation_data/template-answer/recognition.json')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file.')
    parser.add_argument('--limit_negative', type=int, default=100, help='Path to the output JSON file.')
    parser.add_argument('--limit_positive', type=int, default=10, help='Path to the output JSON file.')
    args = parser.parse_args()

    # Read the prompt from the .txt file
    with open(args.prompt_file_path, 'r') as file:
        template = json.load(file)
        print("Loaded template file from ", args.prompt_file_path)

    for conv in template:
        if conv['type'] == 'positive':
            positive_conv = conv['conversations']
        elif conv['type'] == 'negative':
            negative_conv = conv['conversations']

    total_conv = []
    hard_negative_conv = process_conversation(args.negative_image_folder, negative_conv, limit=args.limit_negative)
    rd_negative_conv = process_conversation(args.random_negative_image_folder, negative_conv, limit=args.limit_negative)
    positive_conv = process_conversation(args.positive_image_folder, positive_conv, limit=args.limit_positive, match_number=args.limit_negative*2)
    total_conv.extend(hard_negative_conv)
    total_conv.extend(positive_conv)
    total_conv.extend(rd_negative_conv)
    
    output_file = os.path.join(args.output_file, 'recognition.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        json.dump(total_conv, file)
    print('JSON file created successfully! at', output_file)

if __name__ == "__main__":
    main()

