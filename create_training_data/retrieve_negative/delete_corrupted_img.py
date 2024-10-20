import os

from PIL import Image
from tqdm import tqdm
def check_and_delete_invalid_images(folder_path):
    # Walk through all files and directories in the folder recursively
    for root, dirs, files in tqdm(os.walk(folder_path)):
        print('Checking', root)
        for filename in files:
            file_path = os.path.join(root, filename)

            # Only process if the file is an image
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                try:
                    # Try to open the image
                    with Image.open(file_path) as img:
                        # Verify the image (this will trigger an exception if the image is invalid)
                        img.verify()

                except (IOError, SyntaxError) as e:
                    # If an error occurs, it's an invalid image
                    print(f"Deleting invalid image: {file_path} - Error: {e}")
                    os.remove(file_path)

if __name__ == "__main__":
    folder_path = "/mnt/localssd/code/data/yochameleon-data/train/"  # Replace with your folder path
    check_and_delete_invalid_images(folder_path)
