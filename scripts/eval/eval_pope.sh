# Define the SKS_NAMES array
SKS_NAMES=("thao" "yuheng" "thuytien" "viruss")

# Change to the evaluation/forgetting directory
cd evaluation/forgetting

# Set other variables
EVAL_FOLDER='/sensei-fs/users/thaon/code/eval-llava'
CKPT_PATH="/sensei-fs/users/thaon/ckpt/1000neg-text/"

BATCH_SIZE=16

# Loop through each SKS_NAME
for SKS_NAME in "${SKS_NAMES[@]}"; do

  # Run all processes in the background, assigning different CUDA devices
  CUDA_VISIBLE_DEVICES=0 python model_vqa_loader.py \
      --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
      --image_folder $EVAL_FOLDER/pope/val2014 \
      --save_location $EVAL_FOLDER/pope/${SKS_NAME}-0.jsonl \
      --max_new_tokens 3 \
      --batch_size $BATCH_SIZE \
      --model_path $CKPT_PATH/$SKS_NAME/0-token.pt &

  CUDA_VISIBLE_DEVICES=1 python model_vqa_loader.py \
      --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
      --image_folder $EVAL_FOLDER/pope/val2014 \
      --save_location $EVAL_FOLDER/pope/${SKS_NAME}-5.jsonl \
      --max_new_tokens 3 \
      --batch_size $BATCH_SIZE \
      --model_path $CKPT_PATH/$SKS_NAME/5-token.pt &

  CUDA_VISIBLE_DEVICES=2 python model_vqa_loader.py \
      --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
      --image_folder $EVAL_FOLDER/pope/val2014 \
      --save_location $EVAL_FOLDER/pope/${SKS_NAME}-10.jsonl \
      --max_new_tokens 3 \
      --batch_size $BATCH_SIZE \
      --model_path $CKPT_PATH/$SKS_NAME/10-token.pt &

  CUDA_VISIBLE_DEVICES=3 python model_vqa_loader.py \
      --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
      --image_folder $EVAL_FOLDER/pope/val2014 \
      --save_location $EVAL_FOLDER/pope/${SKS_NAME}-15.jsonl \
      --max_new_tokens 3 \
      --batch_size $BATCH_SIZE \
      --model_path $CKPT_PATH/$SKS_NAME/15-token.pt &

  # CUDA_VISIBLE_DEVICES=4 python model_vqa_loader.py \
  #     --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
  #     --image_folder $EVAL_FOLDER/pope/val2014 \
  #     --save_location $EVAL_FOLDER/pope/${SKS_NAME}-500.jsonl \
  #     --max_new_tokens 3 \
  #     --batch_size $BATCH_SIZE \
  #     --model_path $CKPT_PATH/$SKS_NAME/500-token.pt &

  # CUDA_VISIBLE_DEVICES=5 python model_vqa_loader.py \
  #     --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
  #     --image_folder $EVAL_FOLDER/pope/val2014 \
  #     --save_location $EVAL_FOLDER/pope/${SKS_NAME}-600.jsonl \
  #     --max_new_tokens 3 \
  #     --batch_size $BATCH_SIZE \
  #     --model_path $CKPT_PATH/$SKS_NAME/600-token.pt &

  # CUDA_VISIBLE_DEVICES=6 python model_vqa_loader.py \
  #     --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
  #     --image_folder $EVAL_FOLDER/pope/val2014 \
  #     --save_location $EVAL_FOLDER/pope/${SKS_NAME}-700.jsonl \
  #     --max_new_tokens 3 \
  #     --batch_size $BATCH_SIZE \
  #     --model_path $CKPT_PATH/$SKS_NAME/700-token.pt &

  # CUDA_VISIBLE_DEVICES=7 python model_vqa_loader.py \
  #     --question_file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
  #     --image_folder $EVAL_FOLDER/pope/val2014 \
  #     --save_location $EVAL_FOLDER/pope/${SKS_NAME}-800.jsonl \
  #     --max_new_tokens 3 \
  #     --batch_size $BATCH_SIZE \
  #     --model_path $CKPT_PATH/$SKS_NAME/800-token.pt &
  # Wait for all background processes to finish before starting the next SKS_NAME
  wait
done
SKS_NAME='thao'
# python eval_pope.py \
#     --annotation-dir $EVAL_FOLDER/pope/ \
#     --question-file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
#     --result-file $EVAL_FOLDER/pope/anole.jsonl \
#     --save_to_txt $EVAL_FOLDER/pope/pope.log

python eval_pope.py \
    --annotation-dir $EVAL_FOLDER/pope/ \
    --question-file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
    --result-file $EVAL_FOLDER/pope/thao-5.jsonl \
    --save_to_txt $EVAL_FOLDER/pope/pope.log

python eval_pope.py \
    --annotation-dir $EVAL_FOLDER/pope/ \
    --question-file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
    --result-file $EVAL_FOLDER/pope/$SKS_NAME-10.jsonl \
    --save_to_txt $EVAL_FOLDER/pope/pope.log

python eval_pope.py \
    --annotation-dir $EVAL_FOLDER/pope/ \
    --question-file $EVAL_FOLDER/pope/llava_pope_test.jsonl \
    --result-file $EVAL_FOLDER/pope/$SKS_NAME-15.jsonl \
    --save_to_txt $EVAL_FOLDER/pope/pope.log

