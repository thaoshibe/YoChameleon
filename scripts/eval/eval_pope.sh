# Define the SKS_NAMES array
SKS_NAMES=("thao" "bo" "mam")

# Change to the evaluation/forgetting directory
cd evaluation/forgetting

# Set other variables
EVAL_FOLDER='/sensei-fs/users/thaon/code/eval-llava'
CKPT_PATH="/sensei-fs/users/thaon/ckpt/1000neg-text/"
CKPT_PATH="/sensei-fs/users/thaon/ckpt-supp/real"

BATCH_SIZE=2

# CUDA_VISIBLE_DEVICES=0 python model_vqa_loader.py \
#     --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
#     --image_folder $EVAL_FOLDER/pope/val2014 \
#     --save_location $EVAL_FOLDER/pope/thao-real-16.jsonl \
#     --max_new_tokens 3 \
#     --batch_size $BATCH_SIZE \
#     --model_path $CKPT_PATH/thao/16-token.pt &

# CUDA_VISIBLE_DEVICES=1 python model_vqa_loader.py \
#     --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
#     --image_folder $EVAL_FOLDER/pope/val2014 \
#     --save_location $EVAL_FOLDER/pope/thao-real-wholemodel-1.jsonl \
#     --max_new_tokens 3 \
#     --batch_size $BATCH_SIZE \
#     --model_path /mnt/localssd/code/ckpt-supp/real-wholemodel/thao/1-model.pt &

# CUDA_VISIBLE_DEVICES=2 python model_vqa_loader.py \
#     --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
#     --image_folder $EVAL_FOLDER/pope/val2014 \
#     --save_location $EVAL_FOLDER/pope/thao-real-10.jsonl \
#     --max_new_tokens 3 \
#     --batch_size $BATCH_SIZE \
#     --model_path $CKPT_PATH/thao/10-token.pt &

CUDA_VISIBLE_DEVICES=3 python model_vqa_loader.py \
    --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
    --image_folder $EVAL_FOLDER/pope/val2014 \
    --save_location $EVAL_FOLDER/pope/thao-real-wholemodel-2.jsonl \
    --max_new_tokens 3 \
    --batch_size $BATCH_SIZE \
    --model_path /mnt/localssd/code/ckpt-supp/real-wholemodel/thao/2-model.pt
# wait

# SKS_NAME='thao'
# # python eval_pope.py \
# #     --annotation-dir $EVAL_FOLDER/pope/ \
# #     --question-file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
# #     --result-file $EVAL_FOLDER/pope/anole.jsonl \
# #     --save_to_txt $EVAL_FOLDER/pope/pope.log

python eval_pope.py \
    --annotation-dir $EVAL_FOLDER/pope/ \
    --question-file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
    --result-file $EVAL_FOLDER/pope/thao-real-wholemodel-1.jsonl \
    --save_to_txt $EVAL_FOLDER/pope/pope.log
# /sensei-fs/users/thaon/code/eval-llava/pope/thao-real-16.jsonl
# python eval_pope.py \
#     --annotation-dir $EVAL_FOLDER/pope/ \
#     --question-file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
#     --result-file $EVAL_FOLDER/pope/$SKS_NAME-10.jsonl \
#     --save_to_txt $EVAL_FOLDER/pope/pope.log

# python eval_pope.py \
#     --annotation-dir $EVAL_FOLDER/pope/ \
#     --question-file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
#     --result-file $EVAL_FOLDER/pope/$SKS_NAME-15.jsonl \
#     --save_to_txt $EVAL_FOLDER/pope/pope.log

