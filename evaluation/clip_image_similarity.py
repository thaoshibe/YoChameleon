import argparse
import glob
import os
import torch

from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor
from transformers import CLIPModel
from transformers import CLIPTokenizer

def get_args():
    parser = argparse.ArgumentParser(description="Face verification using InsightFace")
    parser.add_argument("--fake_folder", type=str, default="/mnt/localssd/code/data/yochameleon-data/train/thao",
                        help="Path to the folder containing fake images")
    parser.add_argument("--real_folder", type=str, default="/mnt/localssd/code/data/yochameleon-data/train/thao",
                        help="Path to the folder containing real images")
    parser.add_argument("--output_file", type=str, default="clip_similarity.json",
                        help="Path to the output JSON file")
    return parser.parse_args()

class CLIPEvaluator:
    def __init__(self, model_id="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_id)
        self.preprocessor = CLIPImageProcessor.from_pretrained(model_id)
        print(f'\n  Hello, this is CLIPEvaluator, model_id: {model_id}\n')

    def load_image(selfm, image_path):
        image = Image.open(image_path)
        return image

    def preprocess(self, image):
        image_pt = self.preprocessor(images=image, return_tensors="pt")["pixel_values"]
        return image_pt

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
    #
    #       This function will take a list of images
    #           If a list of images are given, then compute the average embeddings
    #
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
    @torch.no_grad
    def get_image_features(self, image_ft):
        # image_ft = [self.model.get_image_features(image) for image in tqdm(image_ft)]
        image_ft = self.model.get_image_features(image_ft)
        return image_ft

    def compute_similarity(self, real_images, fake_images, average=True):
        # --- Check if given list are images or paths
        # breakpoint()
        if isinstance(real_images[0], str):
            real_images = [self.load_image(image_path) for image_path in real_images]
            fake_images = [self.load_image(image_path) for image_path in fake_images]
        real_images_ft = [self.preprocess(image) for image in real_images]
        real_images_ft = [self.get_image_features(image_ft) for image_ft in real_images_ft]

        if average:
            real_images_ft = torch.concat(real_images_ft, dim=0).mean(dim=0, keepdim=True)

        # Thao: TODO: Implement the average=False case (?)
        clip_scores = []

        print("\n\n\n             compute CLIP similarity score between generated and real images\n\n\n")
        for fake_image in tqdm(fake_images):
            fake_images_ft = self.preprocess(fake_image)
            fake_images_ft = self.get_image_features(fake_images_ft)
            similarity_score = torch.nn.functional.cosine_similarity(real_images_ft, fake_images_ft).item()
            clip_scores.append(similarity_score)
        return clip_scores

if __name__ == '__main__':
    args = get_args()
    evaluator = CLIPEvaluator()
    list_real_images = glob.glob(args.real_folder + '/*.png')[:1]
    list_fake_images = glob.glob(args.fake_folder + '/*.png')

    similarity = evaluator.compute_similarity(list_real_images, list_fake_images, average=True)
    print('Similarity score:', similarity)