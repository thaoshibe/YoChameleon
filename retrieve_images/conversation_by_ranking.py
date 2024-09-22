import argparse
import glob
import json
import numpy as np
import os
import requests

from tqdm import tqdm

# Function to read prompt from a text file
def read_prompt_from_file(file_path):
    with open(file_path, "r") as file:
        prompt = file.readlines()
    prompt = [p.strip() for p in prompt]
    return prompt

def process_conversation(image_paths):
    # Randomly assign a conversation for each images
    # image_paths = glob.glob(os.path.join(image_path, "*.*"))
    data = []
    # questions = conversations['questions']
    # answers = conversations['answers']
    for index, image_path in tqdm(enumerate(image_paths)):
        if index < 10:
            prefix_tokens = [f'<reserved{16301+i}>' for i in range(14)]
            personalized_tokens = [f'<reserved16300>']
            personalized_tokens.extend(prefix_tokens)
            sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}."
            conv = [
                {
                "from": "human",
                "value": f"A photo of {sks_prompt}"
                },
                {
                "from": "bot",
                "value": f"<image>"
                },
            ]
        elif (index >= 10) & (index < 20):
            prefix_tokens = [f'<reserved{16301+i}>' for i in range(12)]
            personalized_tokens = [f'<reserved16300>']
            personalized_tokens.extend(prefix_tokens)
            sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}."
            conv = [
                {
                "from": "human",
                "value": f"A photo of {sks_prompt}"
                },
                {
                "from": "bot",
                "value": f"<image>"
                },
            ]
        elif (index >= 20) & (index < 35):
            prefix_tokens = [f'<reserved{16301+i}>' for i in range(10)]
            personalized_tokens = [f'<reserved16300>']
            personalized_tokens.extend(prefix_tokens)
            sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}."
            conv = [
                {
                "from": "human",
                "value": f"A photo of {sks_prompt}"
                },
                {
                "from": "bot",
                "value": f"<image>"
                },
            ]
        elif (index >= 35) & (index < 50):
            prefix_tokens = [f'<reserved{16301+i}>' for i in range(8)]
            personalized_tokens = [f'<reserved16300>']
            personalized_tokens.extend(prefix_tokens)
            sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}."
            conv = [
                {
                "from": "human",
                "value": f"A photo of {sks_prompt}"
                },
                {
                "from": "bot",
                "value": f"<image>"
                },
            ]
        elif (index >= 50) & (index < 60):
            prefix_tokens = [f'<reserved{16301+i}>' for i in range(6)]
            personalized_tokens = [f'<reserved16300>']
            personalized_tokens.extend(prefix_tokens)
            sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}."
            conv = [
                {
                "from": "human",
                "value": f"A photo of {sks_prompt}"
                },
                {
                "from": "bot",
                "value": f"<image>"
                },
            ]
        elif (index >= 60) & (index < 70):
            prefix_tokens = [f'<reserved{16301+i}>' for i in range(4)]
            personalized_tokens = [f'<reserved16300>']
            personalized_tokens.extend(prefix_tokens)
            sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}."
            conv = [
                {
                "from": "human",
                "value": f"A photo of {sks_prompt}"
                },
                {
                "from": "bot",
                "value": f"<image>"
                },
            ]
        elif (index >= 70) & (index < 80):
            prefix_tokens = [f'<reserved{16301+i}>' for i in range(2)]
            personalized_tokens = [f'<reserved16300>']
            personalized_tokens.extend(prefix_tokens)
            sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}."
            conv = [
                {
                "from": "human",
                "value": f"A photo of {sks_prompt}"
                },
                {
                "from": "bot",
                "value": f"<image>"
                },
            ]
        else:
            prefix_tokens = [f'<reserved{16301+i}>' for i in range(1)]
            personalized_tokens = [f'<reserved16300>']
            personalized_tokens.extend(prefix_tokens)
            sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}."
            conv = [
                {
                "from": "human",
                "value": f"A photo of {sks_prompt}"
                },
                {
                "from": "bot",
                "value": f"<image>"
                },
            ]
        data.append({
            "image": [image_path],
            "conversations": conv
        })
    list_real_bo = ['/mnt/localssd/code/data/minibo/6.png', '/mnt/localssd/code/data/minibo/7.png', '/mnt/localssd/code/data/minibo/8.png', '/mnt/localssd/code/data/minibo/9.png']
    list_real_bo=list_real_bo*3
    for index, image_path in tqdm(enumerate(list_real_bo)):
        prefix_tokens = [f'<reserved{16301+i}>' for i in range(16)]
        personalized_tokens = [f'<reserved16300>']
        personalized_tokens.extend(prefix_tokens)
        sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}."
        conv = [
            {
            "from": "human",
            "value": f"A photo of {sks_prompt}"
            },
            {
            "from": "bot",
            "value": f"<image>"
            },
        ]
        data.append({
            "image": [image_path],
            "conversations": conv
        })
    return data


def main():
    parser = argparse.ArgumentParser(description='Process images in a folder and prompt for GPT-4 API.')
    # parser.add_argument('--positive_image_folder', type=str, , help='Path to the folder positive containing input images.')
    # parser.add_argument('--negative_image_folder', type=str, , help='Path to the folder positive containing input images.')
    parser.add_argument('--prompt_file_path', type=str, default='./template-answer/recognition.json')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file.')
    args = parser.parse_args()

    # Read the prompt from the .txt file
    image_rankings = read_prompt_from_file('ranking.txt')
    # breakpoint()
    # total_conv.extend(positive_conv)
    total_conv = process_conversation(image_rankings)
    with open(args.output_file, 'w') as file:
        json.dump(total_conv, file)
    print('JSON file created successfully! at', args.output_file)

if __name__ == "__main__":
    main()

