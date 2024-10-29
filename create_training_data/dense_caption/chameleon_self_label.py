import argparse
import base64
import glob
import json
import os
import requests

import re
import torch

from PIL import Image
from tqdm import tqdm
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor

def chameleon_trim_answer(long_answer):
    end_of_turn = '<reserved08706>'
    pattern = r"<reserved08706>(.*)"
    short_answer = re.findall(pattern, long_answer)[0] # trim the first end of turn
    short_answer = short_answer.split(end_of_turn)[0] # trim the second end of turn
    return short_answer

def get_text_conversation(input_image_folder, model, processor, human=True, limit=5):
    if human:
        questions = [
        "What is this person hair color?",
        "What is this person height?",
        "What is this person skin tone?",
        "How would you describe this person hairstyle?",
        "Does this person wear glasses or any accessories?",
        "How would you describe this person outfits?",
        "Does this person have any distinctive facial features?",
        "What is this person overall build or physique?",
        "What is this person general expression or demeanor?",
        "How would you describe this person overall appearance?",
    ]
    else:
        questions = [
        "What color is this subject?",
        "What shape does this subject have?",
        "What is the overall vibe of this subject?",
        "What material is this subject made of?",
        "What size is this subject?",
        "Does this subject have any patterns or markings?",
        "What type of object is this subject?",
        "Does this subject have any distinctive features or details?",
        "What’s this subject's general texture like?",
        "How would you describe this subject's overall appearance?"
        ]
    data = []
    filter_keywords = ['the person', 'The person in the image',
        'The person', 'the subject', 'The subject', 'this person', 'this subject']

    image_paths = glob.glob(os.path.join(input_image_folder, "*.png"))
    if limit is not None:
        image_paths = sorted(image_paths)[:limit]
    for image_path in tqdm(image_paths):
        print(f"Processing image: {image_path}")
        for question in questions:
            current_prompt = f"{question} <image>"
            image = Image.open(image_path)
            inputs = processor(current_prompt, image, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].to(model.device).to(model.dtype)
            inputs['attention_mask'] = inputs['attention_mask'].to(model.device)
            inputs['input_ids'] = inputs['input_ids'].to(model.device)
            outputs = model.generate(**inputs, multimodal_generation_mode="text-only", max_new_tokens=200)
            response = processor.decode(outputs[0], skip_special_tokens=False)
            response = chameleon_trim_answer(response)
            for keyword in filter_keywords:
                response = response.replace(keyword, '<sks>')
            if response:
                print("Response:")
                print(question, response)
                conv = [
                        {
                        "from": "human",
                        "value": f"{question}"
                        },
                        {
                        "from": "bot",
                        "value": f"{response}"
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

def main():
    parser = argparse.ArgumentParser(description='Process images in a folder and prompt for GPT-4 API.')
    parser.add_argument('--input_image_folder', type=str, required=True, help='Path to the folder containing input images.')
    parser.add_argument('--prompt_file_path', type=str, required=True, help='Path to the text file containing the prompt.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file.')
    parser.add_argument('--text_conversation', default=False, action='store_true')
    parser.add_argument('--human', default=False, action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    processor = ChameleonProcessor.from_pretrained('leloy/Anole-7b-v0.1-hf')
    model = ChameleonForConditionalGeneration.from_pretrained(
        "leloy/Anole-7b-v0.1-hf", 
        torch_dtype=torch.bfloat16, 
        # low_cpu_mem_usage=True,
        # attn_implementation="flash_attention_2" #Thao: Don't know why transformers 4.46.1 doesnt support Chameleon with this option
    ).to('cuda')
    data = get_text_conversation(args.input_image_folder, model, processor, human=args.human, limit=args.limit)

    output_file = os.path.join(args.output_file, 'text_conversation_chameleon.json')
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        json.dump(data, file)
    print('JSON file created successfully! at', output_file)

if __name__ == "__main__":
    main()

