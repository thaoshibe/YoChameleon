import argparse
import json
import os
import re

import sys
sys.path.append(os.getcwd())
from m4c_evaluator import TextVQAAccuracyEvaluator
# from chamight.utils import instantiate_from_config

# from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    return parser.parse_args()


def prompt_processor(prompt):
    # breakpoint()
    if 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 4:
        # This part is based on format in model_vqa_loader.py
        # For the first test data, it should return: 'What is the brand of this camera?'
        question = prompt.split('\n')[1]

        # question =  '<image>\n'+prompt+'<reserved08706>'
    else:
        assert False
    return question.lower()


def eval_single(annotation_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)

    try:
        annotations = json.load(open(annotation_file))['data']
    except:
        with open(annotation_file, 'r') as file:
            annotations = []
            for line in file:
                try:
                    json_object = json.loads(line)
                    annotations.append(json_object)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
    annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    for result in results:
        # breakpoint()
        annotation = annotations[(result['question_id'], prompt_processor(result['prompt']))]
        pred_list.append({
            "pred_answer": result['text'].lower(),
            "gt_answers": annotation['answers'],
        })
    evaluator = TextVQAAccuracyEvaluator()
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)

    if args.result_dir is not None:
        for result_file in sorted(os.listdir(args.result_dir)):
            if not result_file.endswith('.jsonl'):
                print(f'Skipping {result_file}')
                continue
            eval_single(args.annotation_file, os.path.join(args.result_dir, result_file))