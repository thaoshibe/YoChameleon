"Borrowed from https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_loader.py"
import argparse
import json
import os
import shortuuid
import torch

from tqdm import tqdm


import math
import re

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from PIL import Image

import sys, datetime, glob, importlib, csv

from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, processor=None):
        self.questions = questions
        self.image_folder = image_folder
        self.processor = processor

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        question = "<image>\n" + line["text"] + '<reserved08706>'
        image = [Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')]
        inputs = self.processor(
            question,
            image,
            return_tensors="pt",
            padding="max_length",
            max_length=1500,
            )
        data_dict = {
            'inputs': inputs,
            'question': question,
            # 'image': image,
            'question_id': line['question_id']
        }
        return data_dict
        # return question, [image], line['question_id']

    def __len__(self):
        return len(self.questions)

def collate_fn(batch):
    inputs = [item['inputs'] for item in batch]
    images = [item['image'] for item in batch]
    img_gen_bools = [item['image_generation'] for item in batch]
    # question = [f'{questions[i]}{answers[i]}' for i in range(len(questions))]
    example = processor(inputs, images, padding=True)
    return example

def eval_model(args):
    # Model
    pattern = r"<reserved08706>(.*)<reserved08706>" # Find the answer

    model = ChameleonForConditionalGeneration.from_pretrained(
        "leloy/Anole-7b-v0.1-hf",
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    ).to('cuda')
    print('Loaded model from leloy/Anole-7b-v0.1-hf')

    processor = ChameleonProcessor.from_pretrained('leloy/Anole-7b-v0.1-hf')

    if args.model_path is not None:
        print(f"A new model path are given, thus, loading model from {args.model_path}")
        state_dict = torch.load(args.model_path, map_location='cuda')#.to(model.dtype)
        model.model.load_state_dict(state_dict)
    else:
        args.model_path = "leloy/Anole-7b-v0.1-hf"

    # Create dataset 
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.save_location)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    dataset = CustomDataset(questions, args.image_folder, processor=processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    for batch in tqdm(dataloader):
        batch['inputs'] = batch['inputs'].to(model.device)
        # reshape tensor to remove batch dimension
        batch['inputs'] = {k: v.squeeze(1) for k, v in batch['inputs'].items()}
        batch['inputs']['pixel_values'] = batch['inputs']['pixel_values'].to(model.dtype)

        output = model.generate(
            **batch['inputs'], 
            multimodal_generation_mode="text-only", 
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            )
        answers = processor.batch_decode(output, skip_special_tokens=False)
        question_ids = batch['question_id']
        questions = batch['question']
        for question_id, question, answer in zip(question_ids, questions, answers):
            # print(question_id.item(), answer)
            answer = re.findall(pattern, answer)[0]
            answer = answer.replace('<reserved08706>', '')
            answer = answer.replace('.', '')
            
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": question_id.item(),
                                       "prompt": question,
                                       "text": answer,
                                       "answer_id": ans_id,
                                       "model_id": args.model_path,
                                       "metadata": {}}) + "\n")
        torch.cuda.empty_cache()

    ans_file.close()
    print('Results saved at:', answers_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--question_file", type=str, default="tables/question.jsonl")
    parser.add_argument("--save_location", type=str, default="answer.jsonl")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()


    print(' ')
    print(' ')
    print(' ')
    print("     WARNING: If max_new_tokens is small, it may affect eval result.")
    print("     WARNING: If max_new_tokens is small, it may affect eval result.")
    print("     WARNING: If max_new_tokens is small, it may affect eval result.")
    print(' ')
    print(' ')
    print(' ')

    eval_model(args)