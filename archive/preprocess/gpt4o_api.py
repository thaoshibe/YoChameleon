import base64
import json
import os
import random
import requests

from openai import AzureOpenAI

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
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def generate_prompt(image_path):
    prompt = f"""
    Given the image {image_path}, please describe this image in detail.
     """
    return prompt

def main():
    input_image_path = "../yollava-data/train/bo/0.png"#"image.jpg"
    prompt = generate_prompt(input_image_path)
    response = call_gpt4_api(input_image_path, prompt, client)
    
    if response:
        print("Response:")
        print(response)
    else:
        print("Failed to get a response.")

if __name__ == "__main__":
    main()