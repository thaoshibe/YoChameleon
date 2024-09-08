import argparse
import base64
import glob
import json
import os
import requests

import random

from openai import AzureOpenAI
from tqdm import tqdm

parameters = {}
parameters['azure'] = {}
parameters['azure']['api_version'] = '2023-07-01-preview'

####### GPT-4o instance 1
parameters['azure']['api_key'] = 'bd475283dc23429e89e9ac97445fb912'
parameters['azure']['azure_endpoint'] = 'https://vietgpt.openai.azure.com/'
parameters['azure']['model'] = 'gpt-4o-2024-05-13'

client = AzureOpenAI(
    api_key=parameters['azure']['api_key'],
    azure_endpoint=parameters['azure']['azure_endpoint'],
    api_version=parameters['azure']['api_version'],
)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to read prompt from a text file
def read_prompt_from_file(file_path):
    with open(file_path, "r") as file:
        prompt = file.read()
    return prompt

# Function to call GPT-4 API
def call_gpt4_api(image_path, current_reference_image, text_input, client):
    base64_image = encode_image(image_path)
    base64_image_reference = encode_image(current_reference_image)
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": f"{text_input}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_reference}"}}
        ]}
    ]

    chat_completion = client.chat.completions.create(messages=messages, model=parameters['azure']['model'], temperature=0.6, max_tokens=512)
    # breakpoint()
    try:
        content = chat_completion.choices[0].message.content
        content = content.replace("Charlie", "<sks>")
        return content
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def get_image_caption(reference_folder, target_folder, prompt_file_path, client):
    # Read the prompt from the .txt file
    prompt = read_prompt_from_file(prompt_file_path)
    print(f"Prompt: {prompt}")
    
    data = []
    # Loop over all images in the folder
    reference_images = glob.glob(os.path.join(reference_folder, "*.png"))
    target_images = glob.glob(os.path.join(target_folder, "*.png"))
    
    for image_path in tqdm(target_images):
        print(f"Processing image: {image_path}")
        current_reference_image = random.choice(reference_images)
        response = call_gpt4_api(image_path, current_reference_image, prompt, client)
        if response:
            print("Response:")
            print(response)
            conv = [
                    {
                    "from": "human",
                    "value": f"{response}"
                    },
                    {
                    "from": "bot",
                    "value": "<image>"
                    },
                    # {"question": None, "answer": None},
                ]
            # Add the key-value pair to the main dictionary
            # data[filename] = qa_list
            data.append({
                "image": [image_path],
                "reference_image": [current_reference_image],
                "conversations":
                conv
                })
        else:
            print("Failed to get a response for image:", image_path)
    return data

def main():
    parser = argparse.ArgumentParser(description='Process images in a folder and prompt for GPT-4 API.')
    parser.add_argument('--reference_folder', type=str, required=True, help='Path to the folder containing input images.')
    parser.add_argument('--target_folder', type=str, required=True, help='Path to the folder containing target images.')
    parser.add_argument('--prompt_file_path', type=str, default='./system-prompts/describe-base-on-reference.txt')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file.')
    parser.add_argument('--text_conversation', default=False, action='store_true')

    args = parser.parse_args()
    data = get_image_caption(args.reference_folder, args.target_folder, args.prompt_file_path, client)
    with open(args.output_file, 'w') as file:
        json.dump(data, file)
    print('JSON file created successfully! at', args.output_file)

if __name__ == "__main__":
    main()