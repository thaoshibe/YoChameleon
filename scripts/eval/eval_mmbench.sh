# EVAL_FOLDER='/sensei-fs/users/thaon/code/eval-llava'
# SAVE_LOCATION=$EVAL_FOLDER/textvqa/anole.jsonl
# IMAGE_FOLDER='/mnt/localssd/code/YoChameleon/evaluation/forgetting/data'
CKPT_PATH="/sensei-fs/users/thaon/ckpt-supp/real"

# BATCH_SIZE=2
cd evaluation/forgetting

# CUDA_VISIBLE_DEVICES=1 python model_vqa_loader.py \
#     --question_file /mnt/localssd/code/mmbench_dev_20230712.tsv \
#     --image_folder $IMAGE_FOLDER/train_images \
#     --save_location $EVAL_FOLDER/thao-real-16.jsonl \
#     --max_new_tokens 10 \
#     --batch_size $BATCH_SIZE \
#     --model_path $CKPT_PATH/thao/16-token.pt

#!/bin/bash

SPLIT="mmbench_dev_20230712"
EXP_NAME='Anole-7b-v0.1-hf'

python model_vqa_mmbench.py \
    --question-file /mnt/localssd/code/$SPLIT.tsv \
    --answers-file /mnt/localssd/code/$SPLIT/base.jsonl \
    --single-pred-prompt \
    --temperature 0 &

python model_vqa_mmbench.py \
    --question-file /mnt/localssd/code/$SPLIT.tsv \
    --answers-file /mnt/localssd/code/$SPLIT/thao-12.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --model_path $CKPT_PATH/thao/12-token.pt
wait

# python model_vqa_mmbench.py \
#     --question-file /mnt/localssd/code/$SPLIT.tsv \
#     --answers-file /mnt/localssd/code/$SPLIT/thao-wholemodel-2.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --model_path /mnt/localssd/code/ckpt-supp/real-wholemodel/thao/2-model.pt

# python model_vqa_mmbench.py \
#     --question-file /mnt/localssd/code/$SPLIT.tsv \
#     --answers-file /mnt/localssd/code/$SPLIT/thao-wholemodel-1.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --model_path /mnt/localssd/code/ckpt-supp/real-wholemodel/thao/1-model.pt

# python model_vqa_mmbench.py \
#     --question-file /mnt/localssd/code/$SPLIT.tsv \
#     --answers-file /mnt/localssd/code/$SPLIT/thao-wholemodel-3.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --model_path /mnt/localssd/code/ckpt-supp/real-wholemodel/thao/3-model.pt

# mkdir -p /sensei-fs/users/yuhli/chameleon1.5/eval/mmbench/answers_upload/$SPLIT
# EXP_NAME='thao-wholemodel-2'
# python convert_mmbench.py \
#     --annotation-file /mnt/localssd/code/$SPLIT.tsv \
#     --result-dir /mnt/localssd/code/$SPLIT \
#     --upload-dir /mnt/localssd/code/$SPLIT/upload \
#     --experiment $EXP_NAME

# EXP_NAME='thao-wholemodel-1'
# python convert_mmbench.py \
#     --annotation-file /mnt/localssd/code/$SPLIT.tsv \
#     --result-dir /mnt/localssd/code/$SPLIT \
#     --upload-dir /mnt/localssd/code/$SPLIT/upload \
#     --experiment $EXP_NAME

EXP_NAME='thao-12'
python convert_mmbench.py \
    --annotation-file /mnt/localssd/code/$SPLIT.tsv \
    --result-dir /mnt/localssd/code/$SPLIT \
    --upload-dir /mnt/localssd/code/$SPLIT/upload \
    --experiment $EXP_NAME

EXP_NAME='base'
python convert_mmbench.py \
    --annotation-file /mnt/localssd/code/$SPLIT.tsv \
    --result-dir /mnt/localssd/code/$SPLIT \
    --upload-dir /mnt/localssd/code/$SPLIT/upload \
    --experiment $EXP_NAME