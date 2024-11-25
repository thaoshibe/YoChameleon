import argparse
import json
import os
import pandas as pd
import shortuuid
import torch

from tqdm import tqdm

import sys, datetime, glob, importlib, csv

from omegaconf import OmegaConf
sys.path.append(os.getcwd())

from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor


import base64
import math

from PIL import Image
from io import BytesIO

all_options = ['A', 'B', 'C', 'D']

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))

def eval_model(args):
    processor = ChameleonProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
    model = ChameleonForConditionalGeneration.from_pretrained(
        "leloy/Anole-7b-v0.1-hf",
        device_map="cuda",
    )
    if args.model_path is not None:
        if '-model' in args.model_path:
            print(f"A new model path are given, thus, loading model from {args.model_path}")
            state_dict = torch.load(args.model_path, map_location='cuda')#.to(model.dtype)
            model.model.load_state_dict(state_dict)

        elif '-token' in args.model_path:
            print(f"A new token path are given, thus, loading model from {args.model_path}")
            prefix_tokens = [f'<reserved{16201+i}>' for i in range(args.prefix_token)]
            personalized_tokens = [f'<reserved16200>'] + prefix_tokens
            personalized_token_ids = processor.tokenizer.convert_tokens_to_ids(personalized_tokens)
            lm_head_path = args.model_path.replace('token', 'lmhead')
            lm_head = torch.load(lm_head_path, map_location='cuda')
            model.lm_head.weight.data[personalized_token_ids] = lm_head.to(model.lm_head.weight.data.device).to(model.dtype)
            model.get_input_embeddings().weight.data[personalized_token_ids] = torch.load(args.model_path).to(model.device).to(model.dtype)

        else:
            print('What are you giving me? It is not either fullmodel or tokens path, what do you want aaa')

    else:
        args.model_path = "leloy/Anole-7b-v0.1-hf"

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    

    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            
            


            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            
            qs = '<image>\n ' + qs + '<reserved08706>'
            


            inputs = processor(qs, image, return_tensors="pt").to(model.device)

            output = model.generate(
                **inputs, 
                multimodal_generation_mode="text-only", 
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                )

            output = processor.decode(output[0], skip_special_tokens=False)

            output = output.split('<reserved08706>')[1:3] # remove predicted end-of-turn toekn
            output = ''.join(output).strip()
            print(output)

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": output,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": args.model_path,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model_path", type=str, default="/sensei-fs/users/yuhli/CHECKPOINTS/chameleon1.5/log/Chamight-dev-test-vqa/checkpoints/ckpt_step=050000.ckpt")
    parser.add_argument("--model_path", type=str, default="leloy/Anole-7b-v0.1-hf")
    parser.add_argument("--config_path", type=str, default="/sensei-fs/users/yuhli/CHECKPOINTS/chameleon1.5/log/Chamight-dev-test-vqa/configs/2024-08-16T16-32-07-project.yaml")
    # parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    # parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--prefix_token", type=int, default=16)
    args = parser.parse_args()

    eval_model(args)