import os

import json

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
        # --- Center crop 
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

        # --- Maybe flip the image...
        image = self.flip_transform(image)
        # breakpoint()
        # image_sizes = [x.size for x in images]
        # breakpoint()
        # image = Image.open(image_path).convert("RGB")
        # prompt = self.questions[i]
        # inputs = self.processor(prompt, image, return_tensors="pt")
        # images_tensor = process_images(
        #     images,
        #     self.image_processor,
        #     self.config
        # )
        # example["images"] = images_tensor
        # example['query'] = self.questions[i]#.replace('<sks>', f'<{self.sks_name}>')
        # example['answer'] = self.answers[i]#.replace('<sks>', f'<{self.sks_name}>')
        # example['image'] = images[0]
        example = self.processor(self.questions[i] + self.answers[i], image, return_tensors="pt")
        # breakpoint()
        example['query'] = self.questions[i]
        example['answer'] = self.answers[i]
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
