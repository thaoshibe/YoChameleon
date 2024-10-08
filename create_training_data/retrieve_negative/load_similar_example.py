#!/usr/bin/env python

import cv2
import numpy as np
import piat  # https://git.corp.adobe.com/sniklaus/piat
import traceback

import  glob
import argparse
import json
import os

from PIL import Image
from piat_utils import search_similar_multiple_images
from tqdm import tqdm

def load_base_image(image_path):
    """Load the base image using PIL and convert to a numpy array."""
    return np.array(Image.open(image_path))

def search_similar_images(base_image, limit=100):
    """Search for similar images using PIAT."""
    return piat.search_similar(base_image, intLimit=limit)

def get_image_from_piat(objSample):
    """Retrieve image from PIAT using various sources."""
    try:
        npyImage = piat.get_image({'strSource': '256-pil-antialias'}, objSample['images_raw.strImagehash'])
    except:
        npyImage = piat.get_image({'strSource': 'raw'}, objSample['images_raw.strImagehash'])
    # try:
    #     npyImage = piat.get_image({'strSource': '256-pil-antialias'}, objSample['images_raw.strImagehash'])
    # except:
    #     try:
    #         npyImage = piat.get_image({'strSource': 'raw'}, objSample['images_raw.strImagehash'])
    #     except:
    #         npyImage = None

    return npyImage

def convert_to_cv2_format(image):
    """Convert RGB image (from PIAT) to BGR format (used by OpenCV)."""
    return np.ascontiguousarray(image[:, :, [2, 1, 0]])

def save_image(image, filename):
    """Save an image to a file."""
    cv2.imwrite(filename, image)
    print(f"Image saved as {filename}")

def retrieve_for_one_image(image_path, output_folder = 'piat_retrieved', limit=100, collage_creation=True):
    # Search for similar images
    data_dict = []
    image = np.array(Image.open(image_path))
    for objSample in search_similar_images(image, limit=limit):
        npyImage = get_image_from_piat(objSample)
        if npyImage is None:
            print(f"Could not retrieve image {objSample['images_raw.strImagehash']}")
            continue
        npyImage = convert_to_cv2_format(npyImage)
        save_location = f"{output_folder}/{objSample['images_raw.strImagehash']}.png"
        # Add text to the image
        # text = str(objSample['fltDistance']) + ': ' + ([objText['strText'] for objText in objSample['objTexts'] if objText['strTetype'] == '  '] + [''])[0]
        image_hash = objSample['images_raw.strImagehash']
        clip_score = objSample['fltDistance']
        data_dict.append(
            {
            "image": image_hash,
            "image_path": save_location,
            "clip_score": clip_score
            }
        )
        cv2.imwrite(save_location, npyImage)
    # index = image_path.split('/')[-1].split('.')[0]
    with open(f"{output_folder}/scores.json", 'w') as f:
        json.dump(data_dict, f)
    print(f"Saved scores.json")
    return data_dict

def retrieve_multiple_images(image_paths, output_folder = 'piat_retrieved', limit=100, collage_creation=True):
    # Search for similar images
    data_dict = []
    # image = np.array(Image.open(image_path))
    images = [np.array(Image.open(image_path)) for image_path in image_paths]
    
    # search_similar_multiple_images(images, intLimit=limit)
    for objSample in tqdm(search_similar_multiple_images(images, intLimit=limit)):
        npyImage = get_image_from_piat(objSample)
        npyImage = convert_to_cv2_format(npyImage)
        save_location = f"{output_folder}/{objSample['images_raw.strImagehash']}.png"
        # Add text to the image
        # text = str(objSample['fltDistance']) + ': ' + ([objText['strText'] for objText in objSample['objTexts'] if objText['strTetype'] == '  '] + [''])[0]
        image_hash = objSample['images_raw.strImagehash']
        clip_score = objSample['fltDistance']
        data_dict.append(
            {
            "image": image_hash,
            "image_path": save_location,
            "clip_score": clip_score
            }
        )
        cv2.imwrite(save_location, npyImage)
        print(f"Saved {save_location}")
    # index = image_paths[0].split('/')[-2]#.split('.')[0]
    with open(f"{output_folder}/scores.json", 'w') as f:
        json.dump(data_dict, f)
    # print(f"Saved {index}.json")
    return data_dict

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Retrieve similar images and create a collage.")
    parser.add_argument("--input_folder", type=str, default='/mnt/localssd/code/data/minimam/', help="Path to the base image")
    parser.add_argument("--save_folder", type=str, default='/mnt/localssd/code/data/minimam/piat_retrieved', help="Path to the base image")
    parser.add_argument("--limit", type=int, default=150, help="Number of similar images to retrieve")
    return parser.parse_args()

if __name__ == "__main__":
    # Load the base image
    args = get_args()
    os.makedirs(args.save_folder, exist_ok=True)
    print(f"Images will be saved in {args.save_folder}")
    input_images = glob.glob(args.input_folder + '/*.png')[:4]
    data_dict = retrieve_multiple_images(input_images,
                output_folder=args.save_folder,
                limit=args.limit)

    # This is for retrieve each images by images
    # for image_path in tqdm(input_images):
    #     try:
    #         data_dict = retrieve_for_one_image(
    #             image_paths,
    #             output_folder=args.save_folder,
    #             limit=args.limit)
    #     except Exception as e:
    #         print(f"Error processing {image_path}: {e}")
    #         traceback.print_exc()
