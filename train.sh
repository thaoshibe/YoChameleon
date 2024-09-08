# CUDA_VISIBLE_DEVICES=2,3 python train.py --sks_name mam --batch_size 4 --data_root ./yollava-data/train/ --exp_name bz4-caption-v1 --model_id leloy/Anole-7b-v0.1-hf --image_gen_only
# CUDA_VISIBLE_DEVICES=0,1 python train.py --sks_name bo --batch_size 4 --data_root ./yollava-data/train/ --exp_name bz4-caption-v1 --model_id leloy/Anole-7b-v0.1-hf --image_gen_only
# CUDA_VISIBLE_DEVICES=4,5 python train.py --sks_name bo --batch_size 4 --data_root ./yollava-data/train/ --exp_name bz4-caption-v2 --model_id leloy/Anole-7b-v0.1-hf --image_gen_only
# CUDA_VISIBLE_DEVICES=6,7 python train.py --sks_name mam --batch_size 4 --data_root ./yollava-data/train/ --exp_name bz4-caption-v2 --model_id leloy/Anole-7b-v0.1-hf --image_gen_only


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


