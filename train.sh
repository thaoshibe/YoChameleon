# Train process with total dataset
#--- Create inpainting data
# python create_augmentation_mask.py \
# 	--image_folder /mnt/localssd/code/data/minibo \
# 	--output_folder /mnt/localssd/code/data/minibo/augmented \
# 	--num_inpaintings_per_image 100

# python sdxl-inpainting.py \
# 	--image_folder /mnt/localssd/code/data/minibo/foreground \
# 	--output_folder /mnt/localssd/code/data/minibo/inpainted \
# 	--mask_folder /mnt/localssd/code/data/minibo/mask

# #--- Create recognition data
# python create_conversation.py \
# 	--positive_image_folder /mnt/localssd/code/data/minibo/foreground \
# 	--negative_image_folder /mnt/localssd/code/YoChameleon/example_training_data/bo/laion \
# 	--output_file /mnt/localssd/code/data/minibo/recognition.json

# #--- Create image caption data
# python gpt4o-api.py \
# 	--input_image_folder /mnt/localssd/code/data/minibo/inpainted \
# 	--prompt_file_path ./system-prompts/image_caption_prompt.txt \
# 	--output_file /mnt/localssd/code/data/minibo/inpainted.json

# #--- Create simple conversation data
# python gpt4o-api.py --text_conversation \
# 	--input_image_folder /mnt/localssd/code/data/minibo/ \
# 	--prompt_file_path ./system-prompts/text-conversation.txt \
# 	--output_file /mnt/localssd/code/data/minibo/text-conversation.json

# python train.py --config ./config/bo-sdxl.yaml

#----------------------------------------------

NAME="thao"

INPUT_FOLDER="/mnt/localssd/code/data/yollava-data/train/${NAME}"
SAVE_FOLDER="${INPUT_FOLDER}/negative_example"
LIMIT=500

# python create_training_data/retrieve_negative/load_similar_example.py \
#     --input_folder $INPUT_FOLDER \
#     --save_folder $SAVE_FOLDER \
#     --limit $LIMIT

# python evaluation/face_verification.py --real_folder $INPUT_FOLDER \
#     --fake_folder $INPUT_FOLDER

# python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder $INPUT_FOLDER \
#     --save_folder $SAVE_FOLDER \
#     --limit $LIMIT

# CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --config config/${NAME}-v1.yaml

# Create training data with 64 tokens
# python create_training_data/retrieve_negative/create_conversation_by_ranking.py --input_folder /mnt/localssd/code/data/yollava-data/train/thao/ --save_folder /mnt/localssd/code/data/yollava-data/train/thao/negative_example --limit 500 --token_length 16 --spacing 4


# Create training data with real image only
python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
    --input_folder /mnt/localssd/code/data/yollava-data/train/thao/ \
    --save_folder /mnt/localssd/code/data/yollava-data/train/thao/negative_example \
    --version real-4 \
    --limit 500 \
    --token_length 64 \
    --spacing 64 \
    --num_of_real_images 500

# python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder /mnt/localssd/code/data/yollava-data/train/thao/ \
#     --save_folder /mnt/localssd/code/data/yollava-data/train/thao/negative_example \
#     --version caption-only \
#     --limit 500 \
#     --token_length 64 \
#     --spacing 8 \
#     --negative_image True \
#     --num_of_real_images 100