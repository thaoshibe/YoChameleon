import json
import os

import glob

import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import ChameleonProcessor

START_OF_IMAGE_INDEX = 8197 # <racm3:break>
END_OF_IMAGE_INDEX = 8196 # <eoss>
PAD_INDEX = 1

class PersonalizedDataset_Anole(Dataset):
    def __init__(
        self,
        data_root,
        sks_name,
        # tokenizer,
        set="train",
        placeholder_token="<sks>",
        center_crop=False,
        device="cuda",
        config=None,
        model_id=None,
        flip_p=0.5,
        train_lm_head=False,
        extreme_negative=False,
        recog_only=False,
        random_image=False,
        text_only=False,
        personalized_prompt = False,
        repeat=10,
    ):
        self.data_root = data_root
        self.device = device
        self.config = config
        self.processor = ChameleonProcessor.from_pretrained(model_id)
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.sks_name = sks_name
        
        self.personalized_prompt = personalized_prompt
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        # --- Load data from json files
        self.image_paths = glob.glob(os.path.join(self.data_root, self.sks_name, '*.png'))
        self.image_paths = self.image_paths*repeat
        self._length = len(self.image_paths)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        # --- Center crop -- Not sure?
        # if self.center_crop:
        #     crop = min(img.shape[0], img.shape[1])
        #     (
        #         h,
        #         w,
        #     ) = (
        #         img.shape[0],
        #         img.shape[1],
        #     )
        #     img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
        image_path = self.image_paths[i]
        image = Image.open(image_path).convert("RGB")

        # --- Maybe flip the image... (? TODO)
        image = self.flip_transform(image)
        # question = self.questions[i].replace(f'<{self.sks_name}>', '<reserved16300>')
        # answer = self.answers[i].replace(f'<{self.sks_name}>', '<reserved16300>')
        question = ''
        answer = '<image>'
        prompt = f'{self.personalized_prompt}{question}{answer}'

        prompt_question = f'{self.personalized_prompt}{question}'
        index_question = len(self.processor(prompt_question)['input_ids'][0])
        example = self.processor(prompt, image, return_tensors="pt")
        # print(question, answer, prompt)
        # -- compute loss on image content
        example['labels'] = example['input_ids'].clone()
        example['labels'][:, :index_question]=-100
        # doubt check this -- should we compute loss on the <eoss> token?
        # example['labels'][:, -1:]=-100
        example['labels'][:, -2:]=-100

        example['query'] = prompt_question
        example['answer'] = answer
        example['image_path'] = image_path
        return example

class PersonalizedDataset(Dataset):
    def __init__(
        self,
        data_root,
        sks_name,
        # tokenizer,
        set="train",
        placeholder_token="<sks>",
        center_crop=False,
        device="cuda",
        config=None,
        model_id=None,
        flip_p=0.5,
        train_lm_head=False,
        extreme_negative=False,
        recog_only=False,
        random_image=False,
        text_only=False,
        personalized_prompt = False,
    ):
        self.data_root = data_root
        self.device = device
        self.config = config
        self.processor = ChameleonProcessor.from_pretrained(model_id)
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.sks_name = sks_name
        self.questions = []
        self.images_path = []
        self.answers = []
        self.has_image = []
        self.personalized_prompt = personalized_prompt
        # --- Load data from json files

        conversation_types = ['recognition_positive', 'recognition_negative-laion', 'recognition_negative-cc12m', 'text-only-conversation']
        for conversation_type in conversation_types:
            f = open(os.path.join(data_root, sks_name, f'{conversation_type}.json'))
            data = json.load(f)
            file_names = [x for x in data.keys()]
            for file_name in file_names:
                questions = []
                answers = []
                for conv in data[file_name]:
                    questions.append(conv['Human'])
                    answers.append(conv['AI'])

                self.questions.extend(questions)
                self.answers.extend(answers)
                
                self.images_path.extend([file_name]*len(answers))
                if conversation_type == 'text-only-conversation':
                    self.has_image.extend([False]*len(answers))
                else:
                    self.has_image.extend([True]*len(answers))
            print(conversation_type, len(self.questions))

        print('Total: ', len(self.questions), len(self.answers), len(self.images_path), len(self.has_image))
        if set == "train":
            self._length = len(self.questions)
        else:
            self._length = self.num_images
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

        # self.templates = my_query_templates

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        # --- Center crop -- Not sure?
        # if self.center_crop:
        #     crop = min(img.shape[0], img.shape[1])
        #     (
        #         h,
        #         w,
        #     ) = (
        #         img.shape[0],
        #         img.shape[1],
        #     )
        #     img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
        image_path = self.images_path[i]
        image = Image.open(image_path).convert("RGB")

        # --- Maybe flip the image... (? TODO)
        # image = self.flip_transform(image)
        question = self.questions[i].replace(f'<{self.sks_name}>', '<reserved16300>')
        answer = self.answers[i].replace(f'<{self.sks_name}>', '<reserved16300>')

        prompt = f'{self.personalized_prompt} {question} <image> {answer}'
        question = f'{self.personalized_prompt} {question} <image>'
        index_question = len(self.processor(question)['input_ids'][0])
        example = self.processor(prompt, image, return_tensors="pt")
        # print(question, answer, prompt)
        # -- compute loss on answer only
        example['labels'] = example['input_ids'].clone()
        example['labels'][:, :index_question]=-100

        example['query'] = question
        example['answer'] = answer
        example['image_path'] = image_path
        return example


class PersonalizedDataset_Mixture(Dataset):
    def __init__(
        self,
        data_root,
        sks_name,
        # tokenizer,
        set="train",
        placeholder_token="<sks>",
        center_crop=False,
        device="cuda",
        config=None,
        model_id=None,
        flip_p=0.5,
        train_lm_head=False,
        extreme_negative=False,
        recog_only=False,
        random_image=False,
        text_only=False,
        personalized_prompt = False,
        get_image_tokens = None,
    ):
        self.data_root = data_root
        self.device = device
        self.config = config
        self.processor = ChameleonProcessor.from_pretrained(model_id)
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.sks_name = sks_name
        self.questions = []
        self.images_path = []
        self.answers = []
        self.has_image = []
        self.require_image_generation = []
        self.personalized_prompt = personalized_prompt
        self.get_image_tokens = get_image_tokens
        # --- Load data from json files

        conversation_types = ['recognition_positive', 'recognition_negative-laion', 'recognition_negative-cc12m', 'text-only-conversation']
        for conversation_type in conversation_types:
            f = open(os.path.join(data_root, sks_name, f'{conversation_type}.json'))
            data = json.load(f)
            file_names = [x for x in data.keys()]
            for file_name in file_names:
                questions = []
                answers = []
                for conv in data[file_name]:
                    questions.append(conv['Human'])
                    answers.append(conv['AI'])

                self.questions.extend(questions)
                self.answers.extend(answers)
                
                self.images_path.extend([file_name]*len(answers))
                if conversation_type == 'text-only-conversation':
                    self.has_image.extend([False]*len(answers))
                    self.require_image_generation.extend([False]*len(answers))
                else:
                    self.has_image.extend([True]*len(answers))
                    self.require_image_generation.extend([False]*len(answers))
            print(conversation_type, len(self.questions))
        print('Total: ', len(self.questions), len(self.answers), len(self.images_path), len(self.has_image))
        # Add data for image generation
        gt_images = [x for x in self.images_path if f'train/{self.sks_name}' in x]
        for image_path in gt_images:
            self.questions.append('')
            self.answers.append('<image>')
            self.images_path.append(image_path)
            self.has_image.append([True])
            self.require_image_generation.extend([True])

        if set == "train":
            self._length = len(self.questions)
        else:
            self._length = self.num_images
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

        # self.templates = my_query_templates

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        # --- Center crop -- Not sure?
        # if self.center_crop:
        #     crop = min(img.shape[0], img.shape[1])
        #     (
        #         h,
        #         w,
        #     ) = (
        #         img.shape[0],
        #         img.shape[1],
        #     )
        #     img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
        # breakpoint()
        image_generation = self.require_image_generation[i]
        image_path = self.images_path[i]
        image = Image.open(image_path).convert("RGB")
        image = self.flip_transform(image)

        if image_generation:
            # question = self.questions[i].replace(f'<{self.sks_name}>', '<reserved16300>')
            # answer = self.answers[i].replace(f'<{self.sks_name}>', '<reserved16300>')
            question = ''
            answer = '<image>'
            prompt = f'{self.personalized_prompt}{question}{answer}'

            prompt_question = f'{self.personalized_prompt}{question}'
            # index_answer = len(self.processor(answer)['input_ids'][0])
            example = self.processor(prompt, image,
                return_tensors="pt",
                padding='max_length', max_length=2048)
            # print(question, answer, prompt)
            # -- compute loss on image content
            # example['labels'] = example['input_ids'].clone()
            vqgan_ids = self.get_image_tokens(pixel_values=example['pixel_values'])
            for key in example:
                example[key] = example[key][0]

            example['labels'] = example['input_ids'].clone()
            soi_index = torch.nonzero(example['labels']==START_OF_IMAGE_INDEX).item()
            eoi_index = torch.nonzero(example['labels']==END_OF_IMAGE_INDEX).item()
            example['labels'][soi_index+1:eoi_index]= vqgan_ids
            example['labels'][:soi_index+1]=-100
            example['labels'][eoi_index:]=-100

        else:
            question = self.questions[i].replace(f'<{self.sks_name}>', '<reserved16300>')
            answer = self.answers[i].replace(f'<{self.sks_name}>', '<reserved16300>')

            prompt = f'{self.personalized_prompt} {question}<image><reserved08706>{answer}'
            # question = f'{self.personalized_prompt} {question} <image>'
            index_answer = len(self.processor(answer)['input_ids'][0])
            
            example = self.processor(prompt,
                image,
                return_tensors="pt",
                padding='max_length', max_length=2048)

            for key in example:
                example[key] = example[key][0]
            # -- compute loss on answer only
            example['labels'] = example['input_ids'].clone()
            example['labels'][:-index_answer+1]=-100
        example['query'] = question
        example['answer'] = answer
        example['image_path'] = image_path
        example['has_image'] = self.has_image[i]
        example['image_generation'] = image_generation
        return example

if __name__ == "__main__":
    # ---- TEST for Chameleon-text-conversation data
    # print('Hi Shibe')
    # train_dataset = PersonalizedDataset(data_root="./example_training_data/", sks_name='mam')
    # print('Dataset loaded! -- But be careful, it is not yet processed!')

    # model_id = './chameleon-hf/chameleon-7b'
    # chemeleon_processor = ChameleonProcessor.from_pretrained(model_id)

    # train_dataset = PersonalizedDataset(
    #     data_root="./example_training_data/",
    #     sks_name='mam',
    #     model_id=model_id
    #     )
    # print(train_dataset[0])

    # ---- TEST for Anole data
    # model_id = 'leloy/Anole-7b-v0.1-hf'
    # chemeleon_processor = ChameleonProcessor.from_pretrained(model_id)

    # train_dataset = PersonalizedDataset_Anole(
    #     data_root="./yollava-data/train/",
    #     sks_name='mam',
    #     model_id=model_id,
    #     personalized_prompt="<sks> is a cat."
    #     )
    # print(train_dataset[0])

    # ---- TEST for mixture data
    model_id = 'leloy/Anole-7b-v0.1-hf'
    chemeleon_processor = ChameleonProcessor.from_pretrained(model_id)

    train_dataset = PersonalizedDataset_Mixture(
        data_root="./example_training_data/",
        sks_name='mam',
        model_id=model_id,
        personalized_prompt="<sks> is a cat."
        )
    print(train_dataset[400])