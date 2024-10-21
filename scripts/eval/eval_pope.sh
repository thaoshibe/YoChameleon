cd evaluation/forgetting

EVAL_FOLDER='/sensei-fs/users/thaon/code/eval-llava'
SAVE_LOCATION=$EVAL_FOLDER/pope/thao-25.jsonl

# CUDA_VISIBLE_DEVICES=1 python model_vqa_loader.py \
#     --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
#     --image_folder $EVAL_FOLDER/pope/val2014 \
#     --save_location $EVAL_FOLDER/pope/thao-25.jsonl \
#     --temperature 0 \
#     --max_new_tokens 3 \

python eval_pope.py \
    --annotation-dir $EVAL_FOLDER/pope/ \
    --question-file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
    --result-file $EVAL_FOLDER/pope/thao-25.jsonl