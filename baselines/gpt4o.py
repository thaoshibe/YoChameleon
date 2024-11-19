import argparse
import base64
import glob
import json
import os
import requests

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
def call_gpt4_api(image_path, text_input, client):
    base64_image = encode_image(image_path)

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": f"{text_input}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}
    ]

    chat_completion = client.chat.completions.create(messages=messages, model=parameters['azure']['model'], temperature=0.6, max_tokens=512)

    try:
        content = chat_completion.choices[0].message.content
        content = content.replace("Charlie", "<sks>")
        return content
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def get_image_caption(input_image_folder, prompt_file_path, client):
    # Read the prompt from the .txt file
    prompt = read_prompt_from_file(prompt_file_path)
    print(f"Prompt: {prompt}")
    data = []
    # Loop over all images in the folder
    image_paths = glob.glob(os.path.join(input_image_folder, "*.*"))
    
    for image_path in tqdm(image_paths):
        print(f"Processing image: {image_path}")
        response = call_gpt4_api(image_path, prompt, client)
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
                "conversations":
                conv
                })
        else:
            print("Failed to get a response for image:", image_path)
    return data

def image_generation(image_path, prompt, client):
    response = client.images.generate(
      model="dall-e-3",
      prompt="a white siamese cat",
      size="1024x1024",
      quality="standard",
      n=1,
    )

    image_url = response.data[0].url
    breakpoint()
    return image_url

def main():
    parser = argparse.ArgumentParser(description='Process images in a folder and prompt for GPT-4 API.')
    parser.add_argument('--input_image_folder', type=str, help='Path to the folder containing input images.')
    parser.add_argument('--prompt_file_path', type=str, help='Path to the text file containing the prompt.')
    parser.add_argument('--output_file', type=str, help='Path to the output JSON file.')
    parser.add_argument('--human', default=False, action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    breakpoint()
    # data = get_image_caption(args.input_image_folder, args.prompt_file_path, client)

    # output_file = os.path.join(args.output_file, 'text_conversation.json')
    # os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    # with open(output_file, 'w') as file:
    #     json.dump(data, file)
    # print('JSON file created successfully! at', output_file)

if __name__ == "__main__":
    main()

