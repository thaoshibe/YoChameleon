#wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip

cd evaluation/forgetting
# mkdir data
# cd data
# wget http://images.cocodataset.org/zips/val2014.zip
# unzip val2014.zip
# cd ..
# wget wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json

EVAL_FOLDER='/sensei-fs/users/thaon/code/eval-llava'
SAVE_LOCATION=$EVAL_FOLDER/textvqa/anole.jsonl
IMAGE_FOLDER='/mnt/localssd/code/YoChameleon/evaluation/forgetting/data'
CKPT_PATH="/sensei-fs/users/thaon/ckpt-supp/real"

# CUDA_VISIBLE_DEVICES=6 python model_vqa_loader.py \
#     --question_file $EVAL_FOLDER/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image_folder $IMAGE_FOLDER/train_images \
#     --save_location $SAVE_LOCATION \
#     --temperature 0 \
#     --max_new_tokens 10

BATCH_SIZE=2

CUDA_VISIBLE_DEVICES=1 python model_vqa_loader.py \
    --question_file $EVAL_FOLDER/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image_folder $IMAGE_FOLDER/train_images \
    --save_location $EVAL_FOLDER/thao-real-16.jsonl \
    --max_new_tokens 10 \
    --batch_size $BATCH_SIZE \
    --model_path $CKPT_PATH/thao/16-token.pt # &

# CUDA_VISIBLE_DEVICES=3 python model_vqa_loader.py \
#     --question_file $EVAL_FOLDER/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image_folder $IMAGE_FOLDER/train_images \
#     --save_location $EVAL_FOLDER/thao-real-wholemodel-1.jsonl \
#     --max_new_tokens 10 \
#     --batch_size $BATCH_SIZE \
#     --model_path /mnt/localssd/code/ckpt-supp/real-wholemodel/thao/1-model.pt &

# CUDA_VISIBLE_DEVICES=4 python model_vqa_loader.py \
#     --question_file $EVAL_FOLDER/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image_folder $IMAGE_FOLDER/train_images \
#     --save_location $EVAL_FOLDER/thao-real-10.jsonl \
#     --max_new_tokens 10 \
#     --batch_size $BATCH_SIZE \
#     --model_path $CKPT_PATH/thao/10-token.pt &

# CUDA_VISIBLE_DEVICES=5 python model_vqa_loader.py \
#     --question_file $EVAL_FOLDER/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image_folder $IMAGE_FOLDER/train_images \
#     --save_location $EVAL_FOLDER/thao-real-wholemodel-2.jsonl \
#     --max_new_tokens 10 \
#     --batch_size $BATCH_SIZE \
#     --model_path /mnt/localssd/code/ckpt-supp/real-wholemodel/thao/2-model.pt &
# wait

# python eval_textvqa.py \
#     --annotation-file TextVQA_0.5.1_val.json \
#     --result-file $EVAL_FOLDER/thao-real-wholemodel-2.jsonl