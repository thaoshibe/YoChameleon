#!/usr/bin/env python

import cv2
import numpy as np
import piat  # https://git.corp.adobe.com/sniklaus/piat
import traceback

import  glob
import argparse
import json
import os

from tqdm import tqdm

from PIL import Image

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
    return npyImage

def convert_to_cv2_format(image):
    """Convert RGB image (from PIAT) to BGR format (used by OpenCV)."""
    return np.ascontiguousarray(image[:, :, [2, 1, 0]])

def add_text_to_image(image, text):
    """Add text to the image."""
    cv2.putText(img=image, text=text, org=(8, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.4, color=(0, 0, 0), thickness=3)
    cv2.putText(img=image, text=text, org=(8, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.4, color=(255, 255, 255), thickness=1)
    return image

def resize_image(image, size=(200, 200)):
    """Resize the image to a given size."""
    return cv2.resize(image, size)

def create_collage(images, collage_size, img_size):
    """Create a collage from the list of images."""
    if len(images) != collage_size[0] * collage_size[1]:
        print(f"Not enough images for a {collage_size[0]}x{collage_size[1]} collage.")
        return None

    collage = np.zeros((img_size[1] * collage_size[1], img_size[0] * collage_size[0], 3), dtype=np.uint8)

    for idx, image in enumerate(images):
        row = idx // collage_size[1]
        col = idx % collage_size[0]
        collage[row * img_size[1]:(row + 1) * img_size[1], col * img_size[0]:(col + 1) * img_size[0]] = image

    return collage

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
    index = image_path.split('/')[-1].split('.')[0]
    with open(f"{output_folder}/{index}.json", 'w') as f:
        json.dump(data_dict, f)
    print(f"Saved {index}.json")
    return data_dict

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Retrieve similar images and create a collage.")
    parser.add_argument("--input_folder", type=str, default='/mnt/localssd/code/data/minimam/', help="Path to the base image")
    parser.add_argument("--save_folder", type=str, default='/mnt/localssd/code/data/minimam/piat_retrieved', help="Path to the base image")
    parser.add_argument("--limit", type=int, default=150, help="Number of similar images to retrieve")
    return parser.parse_args()

def main():
    """Main function to handle the overall process."""
    # images = []
    # collage_size = (10, 10)  # 10x10 collage
    # img_size = (200, 200)    # Resize all images to 200x200 for uniformity

    # Load the base image
    args = get_args()
    os.makedirs(args.save_folder, exist_ok=True)

    input_images = glob.glob(args.input_folder + '/*.png')
    for image_path in tqdm(input_images):
        try:
            data_dict = retrieve_for_one_image(
                image_path,
                output_folder=args.save_folder,
                limit=args.limit)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            traceback.print_exc()
    # bo_image = load_base_image('/mnt/localssd/code/data/minibo/6.png')


if __name__ == "__main__":
    main()
