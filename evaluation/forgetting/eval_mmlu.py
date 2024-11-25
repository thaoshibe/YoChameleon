import argparse
import json
import os
import pandas as pd
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
    
    splits = {'test': 'abstract_algebra/test-00000-of-00001.parquet', 'validation': 'abstract_algebra/validation-00000-of-00001.parquet', 'dev': 'abstract_algebra/dev-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])
    correct = total = 0
    for idx, row in tqdm(df.iterrows()):
        
        question = row['question']
        choice = row['choices']
        answer = choice[row['answer']]
        choice = ", ".join(choice[:-1]) + " or " + choice[-1] + "?"
        # formatted_choices = "\n".join([f"{opt}. {val}" for opt, val in zip(options, choice)])
        # formatted_choices = formatted_choices + ' Answer with A, B, C or D.'
        inputs = processor(question + ' ' + choice, return_tensors="pt").to(model.device)


        output = model.generate(
            **inputs, 
            multimodal_generation_mode="text-only", 
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            )
        # breakpoint()
        output = processor.decode(output[0], skip_special_tokens=False)

        # breakpoint()
        output = output.split('<reserved08706>')[1:3] # remove predicted end-of-turn token
        output = ''.join(output).strip()
        if answer in output:
            correct+=1
        total+=1
        print(correct/total)
        # ans_id = shortuuid.uuid()
        # ans_file.write(json.dumps({"question_id": idx,
        #                         "round_id": round_idx,
        #                         "prompt": cur_prompt,
        #                         "text": output,
        #                         "options": options,
        #                         "option_char": cur_option_char,
        #                         "answer_id": ans_id,
        #                         "model_id": args.model_path,
        #                         "metadata": {}}) + "\n")
        # ans_file.flush()

        # # rotate options
        # options = options[1:] + options[:1]
        # cur_option_char = cur_option_char[1:] + cur_option_char[:1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model_path", type=str, default="/sensei-fs/users/yuhli/CHECKPOINTS/chameleon1.5/log/Chamight-dev-test-vqa/checkpoints/ckpt_step=050000.ckpt")
    parser.add_argument("--model_path", type=str, default="leloy/Anole-7b-v0.1-hf")
    parser.add_argument("--config_path", type=str, default="/sensei-fs/users/yuhli/CHECKPOINTS/chameleon1.5/log/Chamight-dev-test-vqa/configs/2024-08-16T16-32-07-project.yaml")
    # parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--prefix_token", type=int, default=16)
    args = parser.parse_args()

    eval_model(args)