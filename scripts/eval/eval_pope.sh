cd evaluation/forgetting

EVAL_FOLDER='/sensei-fs/users/thaon/code/eval-llava'
CKPT_PATH="/mnt/localssd/code/ckpt/wholemodel"
SKS_NAME='khanhvy'
BATCH_SIZE=16
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
    --save_location $EVAL_FOLDER/pope/thao-5.jsonl \
    --max_new_tokens 3 \
    --batch_size $BATCH_SIZE \
    --model_path $CKPT_PATH/thao/5-model.pt &

CUDA_VISIBLE_DEVICES=2 python model_vqa_loader.py \
    --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
    --image_folder $EVAL_FOLDER/pope/val2014 \
    --save_location $EVAL_FOLDER/pope/khanhvy-5.jsonl \
    --max_new_tokens 3 \
    --batch_size $BATCH_SIZE \
    --model_path $CKPT_PATH/khanhvy/10-model.pt &

CUDA_VISIBLE_DEVICES=3 python model_vqa_loader.py \
    --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
    --image_folder $EVAL_FOLDER/pope/val2014 \
    --save_location $EVAL_FOLDER/pope/yuheng-5.jsonl \
    --max_new_tokens 3 \
    --batch_size $BATCH_SIZE \
    --model_path $CKPT_PATH/yuheng/15-model.pt &
wait

python eval_pope.py \
    --annotation-dir $EVAL_FOLDER/pope/ \
    --question-file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
    --result-file $EVAL_FOLDER/pope/anole.jsonl \
    --save_to_txt $EVAL_FOLDER/pope/pope.log

python eval_pope.py \
    --annotation-dir $EVAL_FOLDER/pope/ \
    --question-file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
    --result-file $EVAL_FOLDER/pope/thao-5.jsonl \
    --save_to_txt $EVAL_FOLDER/pope/pope.log

python eval_pope.py \
    --annotation-dir $EVAL_FOLDER/pope/ \
    --question-file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
    --result-file $EVAL_FOLDER/pope/khanhvy-5.jsonl \
    --save_to_txt $EVAL_FOLDER/pope/pope.log

python eval_pope.py \
    --annotation-dir $EVAL_FOLDER/pope/ \
    --question-file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
    --result-file $EVAL_FOLDER/pope/yuheng-5.jsonl \
    --save_to_txt $EVAL_FOLDER/pope/pope.log

