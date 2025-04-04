# --- Data Augmentation methdos --- #
# Train process with total dataset
# --- Create inpainting data
# python create_augmentation_mask.py \
#   --image_folder /mnt/localssd/code/data/minibo \
#   --output_folder /mnt/localssd/code/data/minibo/augmented \
#   --num_inpaintings_per_image 100

# python sdxl-inpainting.py \
#   --image_folder /mnt/localssd/code/data/minibo/foreground \
#   --output_folder /mnt/localssd/code/data/minibo/inpainted \
#   --mask_folder /mnt/localssd/code/data/minibo/mask

# #--- Create image caption data
# python gpt4o-api.py \
#   --input_image_folder /mnt/localssd/code/data/minibo/inpainted \
#   --prompt_file_path ./system-prompts/image_caption_prompt.txt \
#   --output_file /mnt/localssd/code/data/minibo/inpainted.json


# python train.py --config config/basic.yml #--no_wandb
# python evaluation/face_verification.py --real_folder $INPUT_FOLDER \
#     --fake_folder $INPUT_FOLDER

# python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder $INPUT_FOLDER \
#     --save_folder $SAVE_FOLDER \
#     --limit $LIMIT

# CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --config config/${NAME}-v1.yaml

# Create training data with 64 tokens
# python create_training_data/retrieve_negative/create_conversation_by_ranking.py --input_folder /mnt/localssd/code/data/yollava-data/train/thao/ --save_folder /sensei-fs/users/thaon/data/json/ --limit 500 --token_length 16 --spacing 4


# Create training data with real image only
# python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder /mnt/localssd/code/data/yollava-data/train/thao/ \
#     --save_folder /sensei-fs/users/thaon/data/json/ \
#     --version real-4 \
#     --limit 500 \
#     --token_length 64 \
#     --spacing 64 \
#     --num_of_real_images 500

# python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder /mnt/localssd/code/data/yollava-data/train/thao/ \
#     --save_folder /sensei-fs/users/thaon/data/json/ \
#     --version thao-8 \
#     --limit 500 \
#     --token_length 8 \
#     --spacing 8 \
#     --negative_image True \
#     --num_of_real_images 0

# python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder /mnt/localssd/code/data/yollava-data/train/thao/ \
#     --save_folder /sensei-fs/users/thaon/data/json/ \
#     --version thao-16 \
#     --limit 500 \
#     --token_length 16 \
#     --spacing 8 \
#     --negative_image True \
#     --num_of_real_images 0

# python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder /mnt/localssd/code/data/yollava-data/train/thao/ \
#     --save_folder ./json \
#     --version "16-4" \
#     --token_length 16 \
#     --spacing 4 \
#     --negative_image True \
#     --num_of_real_images -100

# python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder /mnt/localssd/code/data/yollava-data/train/thao/ \
#     --save_folder ./json \
#     --version "64-8" \
#     --token_length 64 \
#     --spacing 8 \
#     --negative_image True \
#     --num_of_real_images -100

# python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder /mnt/localssd/code/data/yollava-data/train/thao/ \
#     --save_folder ./json \
#     --version "128-16" \
#     --token_length 128 \
#     --spacing 16 \
#     --negative_image True \
#     --num_of_real_images -100
# ### YJ's idea
# python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder /mnt/localssd/code/data/yollava-data/train/thao/ \
#     --save_folder ./json \
#     --version "identifier-64-64" \
#     --token_length 64 \
#     --spacing 64 \
#     --negative_image True \
#     --num_of_real_images -100

# python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder /mnt/localssd/code/data/yollava-data/train/thao/ \
#     --save_folder ./json \
#     --version "identifier-64-8" \
#     --token_length 64 \
#     --spacing 8 \
#     --negative_image True \
#     --num_of_real_images -100

# python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder /mnt/localssd/code/data/yollava-data/train/thao/ \
#     --save_folder ./json \
#     --version "identifier-16-4" \
#     --token_length 16 \
#     --spacing 4 \
#     --negative_image True \
#     --num_of_real_images -100

# python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder /mnt/localssd/code/data/yollava-data/train/thao/ \
#     --save_folder ./json \
#     --version "identifier-128-16" \
#     --token_length 128 \
#     --spacing 16 \
#     --negative_image True \
#     --num_of_real_images -100

# python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder /mnt/localssd/code/data/yollava-data/train/thao/ \
#     --save_folder ./json \
#     --version "500" \
#     --token_length 64 \
#     --spacing 1 \
#     --negative_image True \
#     --num_of_real_images -100 \
#     --limit_negative 500

# python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder /mnt/localssd/code/data/yollava-data/train/thao/ \
#     --save_folder ./json \
#     --version "1000" \
#     --token_length 64 \
#     --spacing 1 \
#     --negative_image True \
#     --num_of_real_images -100 \
#     --limit_negative 1000

# python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder /mnt/localssd/code/data/yollava-data/train/thao/ \
#     --save_folder ./json \
#     --version "2000" \
#     --token_length 64 \
#     --spacing 1 \
#     --negative_image True \
#     --num_of_real_images -100 \
#     --limit_negative 2000

# python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder /mnt/localssd/code/data/yollava-data/train/thao/ \
#     --save_folder ./json \
#     --version "5000" \
#     --token_length 64 \
#     --spacing 1 \
#     --negative_image True \
#     --num_of_real_images -100 \
#     --limit_negative 5000