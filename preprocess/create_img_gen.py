import argparse
import json
import os

def create_json_for_images(folder_path, output_file):
    # Initialize the dictionary to store the JSON structure
    data = []

    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        print(filename)
        # Check if the file is an image (you can add more image file extensions if needed)
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            
            # Create the key for the current image file
            # Create a list of dictionaries for the questions and answers
            qa_list = [
                {
                "from": "human",
                "value": ""
                },
                {
                "from": "bot",
                "value": "<image>"
                },
                # {"question": None, "answer": None},
            ]
            
            # Add the key-value pair to the main dictionary
            # data[filename] = qa_list
            data.append({
                "image": [filename],
                "conversations":
                qa_list
                })
    
    # Write the dictionary to a JSON file
    with open(output_file, 'w') as file:
        json.dump(data, file)
    print('JSON file created successfully! at', output_file)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Generate a JSON file from image files in a folder.')
    parser.add_argument('--image_folder',
        type=str, help='Path to the folder containing the images.',
        default='/mnt/localssd/code/YoChameleon/yollava-data/train')
    parser.add_argument('--output_dir', type=str, default='/mnt/localssd/code/YoChameleon/example_training_data/image_gen_only')
    parser.add_argument('--sks_name', type=str, default='bo')
    args = parser.parse_args()
    print(args)
    # Run the function with the provided arguments
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_file = os.path.join(args.output_dir, args.sks_name + '.json')

    image_folder = os.path.join(args.image_folder, args.sks_name)
    create_json_for_images(image_folder, output_file)
