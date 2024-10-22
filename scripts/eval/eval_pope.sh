cd evaluation/forgetting

EVAL_FOLDER='/sensei-fs/users/thaon/code/eval-llava'
SAVE_LOCATION=$EVAL_FOLDER/pope/thao-25.jsonl
CKPT_PATH="/mnt/localssd/code/ckpt/wholemodel"
SKS_NAME='thao'
BATCH_SIZE=32

# For basemodel

CUDA_VISIBLE_DEVICES=0 python model_vqa_loader.py \
    --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
    --image_folder $EVAL_FOLDER/pope/val2014 \
    --save_location $EVAL_FOLDER/pope/anole.jsonl \
    --max_new_tokens 3 \
    --batch_size $BATCH_SIZE &

CUDA_VISIBLE_DEVICES=1 python model_vqa_loader.py \
    --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
    --image_folder $EVAL_FOLDER/pope/val2014 \
    --save_location $EVAL_FOLDER/pope/$SKS_NAME-5.jsonl \
    --max_new_tokens 3 \
    --batch_size $BATCH_SIZE \
    --model_path $CKPT_PATH/$SKS_NAME/5-model.pt &

CUDA_VISIBLE_DEVICES=2 python model_vqa_loader.py \
    --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
    --image_folder $EVAL_FOLDER/pope/val2014 \
    --save_location $EVAL_FOLDER/pope/$SKS_NAME-10.jsonl \
    --max_new_tokens 3 \
    --batch_size $BATCH_SIZE \
    --model_path $CKPT_PATH/$SKS_NAME/10-model.pt &

CUDA_VISIBLE_DEVICES=3 python model_vqa_loader.py \
    --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
    --image_folder $EVAL_FOLDER/pope/val2014 \
    --save_location $EVAL_FOLDER/pope/$SKS_NAME-15.jsonl \
    --max_new_tokens 3 \
    --batch_size $BATCH_SIZE \
    --model_path $CKPT_PATH/$SKS_NAME/15-model.pt &

CUDA_VISIBLE_DEVICES=4 python model_vqa_loader.py \
    --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
    --image_folder $EVAL_FOLDER/pope/val2014 \
    --save_location $EVAL_FOLDER/pope/$SKS_NAME-20.jsonl \
    --max_new_tokens 3 \
    --batch_size $BATCH_SIZE \
    --model_path $CKPT_PATH/$SKS_NAME/20-model.pt &
    
wait
