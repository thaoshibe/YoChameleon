import argparse
import os
import torch

import glob
import yaml

import re

import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from tqdm import tqdm
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor
from transformers.image_transforms import to_pil_image
def chameleon_trim_answer(long_answer):
    end_of_turn = '<reserved08706>'
    pattern = r"<reserved08706>(.*)"
    short_answer = re.findall(pattern, long_answer)[0] # trim the first end of turn
    short_answer = short_answer.split(end_of_turn)[0] # trim the second end of turn
    return short_answer

class RecognitionData(Dataset):
    def __init__(
        self,
        sks_name,
        placeholder_token="<reserved16200>",
        image_folder=None,
        tokenizer_max_length=1500, 
        processor: ChameleonProcessor = None,
        only_positive: bool = False,
        personalized_prompt: str = None,
    ):
        self.processor = processor
        self.sks_name = sks_name
        self.placeholder_token = placeholder_token
        self.max_length = tokenizer_max_length
        self.personalized_prompt = personalized_prompt
        self.image_paths = glob.glob(os.path.join(image_folder, "*/*.png"))
        self.image_paths = [x for x in self.image_paths if 'negative_example' not in x]
        
        print(f'Found {len(self.image_paths)} images in {image_folder}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        #-- This might be useful, as later can be trained for multiple images (interleaved data)
        # images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        # print(f'Loading {image_path}')
        images = [Image.open(image_path).convert("RGB")]
        #
        #
        #
        # --- FOR Ground Truth Experiments
        #
        #
        #
        # caption_file = './baselines/subject-detailed-captions.json'
        # with open(caption_file, 'r') as f:
        #     captions = json.load(f)
        # question = f"{captions[self.sks_name]} Is this subject in this photo?<image><reserved08706>"
        #
        #
        #
        # --- For Grounth Truth with Image
        #
        #
        #
        question = [f"<image><image><image><image>You are seeing photos of a subject. Is the same subject in this photo?<image><reserved08706>"]
        # breakpoint()
        train_dir = image_path.replace('/test/', '/train/')
        dir_name = os.path.dirname(train_dir)
        list_train_images = glob.glob(f'{dir_name}/*.png')
        rd_train = list_train_images[0]
        images = [Image.open(list_train_images[0]).convert("RGB"),
            Image.open(list_train_images[1]).convert("RGB"),
            Image.open(list_train_images[2]).convert("RGB"),
            Image.open(list_train_images[3]).convert("RGB"),
            Image.open(image_path).convert("RGB")]
        # print(question)
        # # For normally load the data
        # question = f'{self.personalized_prompt} Can you see {self.placeholder_token} in this photo?<image>'
        example = self.processor(
            text=question,
            images=images,
            # padding="max_length",
            # max_length=self.max_length,
            )
        example['inputs'] = {
            'input_ids': example['input_ids'],
            'attention_mask': example['attention_mask'],
            'pixel_values': example['pixel_values'],
        }

        if f'/{self.sks_name}/' in image_path:
            example['labels'] = ['Yes']
        else:
            example['labels'] = ['No']
        return example

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def get_args():
    parser = argparse.ArgumentParser(description='Your Chameleon model')
    # model related
    parser.add_argument('--config', type=str, default='./config/basic.yml')
    parser.add_argument('--iteration', type=str, default=None)
    parser.add_argument('--finetune', action='store_true', help='Use fine-tuned model')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--sks_name', type=str, default=None)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_dir', type=str, default=None)
    # parser.add_argument('--no_wandb', action='store_true', help='Turn off log to WanDB for debug reason')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    config_dict = yaml.safe_load(open(args.config, 'r'))
    config = Config(config_dict)
    config_test = Config(config.test)
    if args.exp_name is not None:
        config.exp_name = args.exp_name
    if args.sks_name is not None:
        config.sks_name = args.sks_name
    if args.iteration is not None:
        config_test.iteration = args.iteration
    if args.save_dir is not None:
        config.savedir = args.save_dir

    # Initialize processor and model
    processor = ChameleonProcessor.from_pretrained(config.model_id)
    model = ChameleonForConditionalGeneration.from_pretrained(
        config.model_id, 
        torch_dtype=torch.bfloat16, 
        # low_cpu_mem_usage=True,
        # attn_implementation="flash_attention_2" #Thao: Don't know why transformers 4.46.1 doesnt support Chameleon with this option
    ).to('cuda')

    # Create personalized tokens
    prefix_tokens = [f'<reserved{16201+i}>' for i in range(config.prefix_token)]
    personalized_tokens = [f'<reserved16200>'] + prefix_tokens
    personalized_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}."
    personalized_token_ids = processor.tokenizer.convert_tokens_to_ids(personalized_tokens)

    model.resize_token_embeddings(len(processor.tokenizer))

    # try:
    #     lm_head_path = os.path.join(config.savedir, config.exp_name, config.sks_name, f'{config_test.iteration}-lmhead.pt')
    #     lm_head = torch.load(lm_head_path, map_location='cuda')
    #     model.lm_head.weight.data[personalized_token_ids] = lm_head.to(model.lm_head.weight.data.device).to(model.dtype)
    #     embedding_path = os.path.join(config.savedir, config.exp_name, config.sks_name, f"{config_test.iteration}-token.pt")
    #     model.get_input_embeddings().weight.data[personalized_token_ids] = torch.load(embedding_path).to(model.device).to(model.dtype)
    # except:
    #     model_path = os.path.join(config.savedir, config.exp_name, config.sks_name, f'{config_test.iteration}-model.pt')
    #     print(model_path)
    #     state_dict = torch.load(model_path, map_location='cuda')#.to(model.dtype)
    #     model.model.load_state_dict(state_dict)

    recognition_data = RecognitionData(
        config.sks_name,
        placeholder_token="<reserved16200>",
        image_folder=config.eval['recognition_path_test'],
        tokenizer_max_length=1500, 
        processor = processor,
        personalized_prompt = personalized_prompt,
    )
    recognition_data_loader = torch.utils.data.DataLoader(recognition_data, batch_size=1, shuffle=False)

    ground_truth = []
    predictions = []
    for batch in tqdm(recognition_data_loader):
        try:
        # batch['inputs'] = batch['inputs'].to(model.device)
        # reshape tensor to remove batch dimension
            batch['inputs'] = {k: v.squeeze(1).to(model.device) for k, v in batch['inputs'].items()}
            batch['inputs']['pixel_values'] = batch['inputs']['pixel_values'].to(model.dtype)[0]
            
            output = model.generate(**batch['inputs'], multimodal_generation_mode="text-only", max_new_tokens=10)
            result_with_special_tokens = processor.batch_decode(output, skip_special_tokens=True)

            # answers = [chameleon_trim_answer(x) for x in result_with_special_tokens]
            
            answers = [x for x in result_with_special_tokens]
            # print(result_with_special_tokens)
            # print(result_with_special_tokens)
            for answer in answers:
                if ('Yes' in answer) or ('yes' in answer):
                    predictions.append('Yes')
                elif ('No' in answer) or ('no' in answer):
                    predictions.append('No')
                else:
                    predictions.append(answer)
            ground_truth.extend(batch['labels'][0])
        except:
            print('Error in batch')

    positive_indices = [i for i, x in enumerate(ground_truth) if x == 'Yes']
    negative_indices = [i for i, x in enumerate(ground_truth) if x == 'No']

    predict_positive = [predictions[i] for i in positive_indices]
    predict_negative = [predictions[i] for i in negative_indices]
    gt_positive = [ground_truth[i] for i in positive_indices]
    gt_negative = [ground_truth[i] for i in negative_indices]
    # accuracy:
    accuracy = sum([1 for i, j in zip(ground_truth, predictions) if i == j]) / len(ground_truth)
    positive_accuracy = sum([1 for i, j in zip(gt_positive, predict_positive) if i == j]) / len(gt_positive)
    negative_accuracy = sum([1 for i, j in zip(gt_negative, predict_negative) if i == j]) / len(gt_negative)

    print('\n\n\n          Evaluation Results          \n\n\n')
    print(f'                   Overall Accuracy: {accuracy}')
    print(f'                   Positive Accuracy: {positive_accuracy}')
    print(f'                   Negative Accuracy: {negative_accuracy}')

    weighted_acc = (positive_accuracy + negative_accuracy) / 2
    print(f'       ╰┈➤ Weighted Accuracy: {weighted_acc}')