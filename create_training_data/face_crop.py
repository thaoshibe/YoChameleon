import random

import glob
import os

from PIL import Image
from autocrop import Cropper
from tqdm import tqdm as tqdm

image_folder = '../../data/dathao'

images_paths = []
for img_extension in ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
	images_paths.extend(glob.glob(os.path.join(image_folder, img_extension)))
cropper = Cropper(face_percent=10)
print(images_paths)
for image_path in tqdm(images_paths):
	try:
		# rd_percent = random.uniform(1, 50)
		
		cropped_array = cropper.crop(
			image_path
			)
		# breakpoint()
		filename = image_path.split('/')[-1]
		cropped_image = Image.fromarray(cropped_array)
		save_location = os.path.join('/mnt/localssd/code/data/dathao_algined', f"{filename}")
		cropped_image.save(save_location)
		print(f'Face detected and saved as {save_location}')
		# print(f'Face detected and saved as {save_location} with {rd_percent}')
	except Exception as e:
		# print(image_path)
		# print(f'No face detected in {image_path}')
		pass
