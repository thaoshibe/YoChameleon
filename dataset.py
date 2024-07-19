import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import ChameleonProcessor


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

        prompt = f'{self.personalized_prompt} {question} <image>. {answer}'
        question = f'{self.personalized_prompt} {question} <image>.'
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

if __name__ == "__main__":
    # print('Hi Shibe')
    # train_dataset = PersonalizedDataset(data_root="./example_training_data/", sks_name='mam')
    # print('Dataset loaded! -- But be careful, it is not yet processed!')

    model_id = './chameleon-hf/chameleon-7b'
    chemeleon_processor = ChameleonProcessor.from_pretrained(model_id)

    train_dataset = PersonalizedDataset(
        data_root="./example_training_data/",
        sks_name='mam',
        model_id=model_id
        )
    print(train_dataset[0])
