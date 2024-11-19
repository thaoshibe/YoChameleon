USER="$(whoami)"
echo "USER"
echo $USER

NAMES=("mam" "ciin" "willinvietnam" "chua-thien-mu")

export WANDB_API_KEY="563710e55fec9aac8f27c7ab80cfed931a2096f5"

### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
###                                                               #
###                    CREATE TRAINING DATA                       #
###                                                               #
### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #

### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
###  Retrieve negative image

# process_image() {
#   NAME=$1
#   INPUT_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
#   SAVE_FOLDER="${INPUT_FOLDER}/negative_example"
#   LIMIT=3000
#   echo "Processing folder: ${NAME}"
#   cd /mnt/localssd/code/YoChameleon

#   python create_training_data/retrieve_negative/load_similar_example.py \
#     --input_folder "$INPUT_FOLDER" \
#     --save_folder "$SAVE_FOLDER" \
#     --limit "$LIMIT" \
#     --origin "l2"
# }

# export -f process_image # Ensure the function is exported to be available for xargs
# echo "${NAMES[@]}" | tr ' ' '\n' | xargs -n 1 -P 6 -I {} bash -c 'process_image "$@"' _ {}

### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
###  Soft positive

# for NAME in "${NAMES[@]}"; do
#   POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
#   NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
#   OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
#   echo "Processing folder: ${NAME}"
#   cd /mnt/localssd/code/YoChameleon
#   mkdir "/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
#   python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder "$POSITIVE_IMAGE_FOLDER" \
#     --save_folder "$OUTPUT_FILE" \
#     --version '500' \
#     --num_of_real_images -100 \
#     --token_length 16 \
#     --spacing 1 \
#     --negative_image True \
#     --limit_negative 500 \
#     --consistent_prompt True 
# done

# for NAME in "${NAMES[@]}"; do
#   POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
#   NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
#   OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
#   echo "Processing folder: ${NAME}"
#   cd /mnt/localssd/code/YoChameleon
#   python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder "$POSITIVE_IMAGE_FOLDER" \
#     --save_folder "$OUTPUT_FILE" \
#     --version '1000' \
#     --num_of_real_images -100 \
#     --token_length 16 \
#     --spacing 1 \
#     --negative_image True \
#     --limit_negative 1000 \
#     --consistent_prompt True 
# done

# for NAME in "${NAMES[@]}"; do
#   POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
#   NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
#   OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
#   echo "Processing folder: ${NAME}"
#   cd /mnt/localssd/code/YoChameleon
#   python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder "$POSITIVE_IMAGE_FOLDER" \
#     --save_folder "$OUTPUT_FILE" \
#     --version '1500' \
#     --num_of_real_images -100 \
#     --token_length 16 \
#     --spacing 1 \
#     --negative_image True \
#     --limit_negative 1500 \
#     --consistent_prompt True 
# done

# for NAME in "${NAMES[@]}"; do
#   POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
#   NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
#   OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
#   echo "Processing folder: ${NAME}"
#   cd /mnt/localssd/code/YoChameleon
#   python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder "$POSITIVE_IMAGE_FOLDER" \
#     --save_folder "$OUTPUT_FILE" \
#     --version '2000' \
#     --num_of_real_images -100 \
#     --token_length 16 \
#     --spacing 1 \
#     --negative_image True \
#     --limit_negative 2000 \
#     --consistent_prompt True 
# done

# for NAME in "${NAMES[@]}"; do
#   POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
#   NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
#   OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
#   echo "Processing folder: ${NAME}"
#   cd /mnt/localssd/code/YoChameleon
#   python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder "$POSITIVE_IMAGE_FOLDER" \
#     --save_folder "$OUTPUT_FILE" \
#     --version '2500' \
#     --num_of_real_images -100 \
#     --token_length 16 \
#     --spacing 1 \
#     --negative_image True \
#     --limit_negative 2500 \
#     --consistent_prompt True 
# done

# for NAME in "${NAMES[@]}"; do
#   POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
#   NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
#   OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
#   echo "Processing folder: ${NAME}"
#   cd /mnt/localssd/code/YoChameleon
#   python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
#     --input_folder "$POSITIVE_IMAGE_FOLDER" \
#     --save_folder "$OUTPUT_FILE" \
#     --version '3000' \
#     --num_of_real_images -100 \
#     --token_length 16 \
#     --spacing 1 \
#     --negative_image True \
#     --limit_negative 3000 \
#     --consistent_prompt True 
# done

### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
###                                                               #
###        THAO: REMEMEBER TO CHANGE THE CONFIG FILE HERE         #
###                                                               #
### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #

# CONFIG_FILE="./config/ablation-500.yaml"

# cd $WORKING_FOLDER/YoChameleon
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/ablation-500.yaml --sks_name "${NAMES[0]}" &
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/ablation-500.yaml --sks_name "${NAMES[1]}" &
# CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/ablation-500.yaml --sks_name "${NAMES[2]}" &
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config ./config/ablation-500.yaml --sks_name "${NAMES[3]}" 
# wait

# CONFIG_FILE="./config/ablation-3000.yaml"

# cd $WORKING_FOLDER/YoChameleon
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/ablation-3000.yaml --sks_name "${NAMES[0]}" &
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/ablation-3000.yaml --sks_name "${NAMES[1]}" &
# CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/ablation-3000.yaml --sks_name "${NAMES[2]}" &
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config ./config/ablation-3000.yaml --sks_name "${NAMES[3]}" 
# wait


# CONFIG_FILE="./config/256.yaml"

# cd $WORKING_FOLDER/YoChameleon
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config $CONFIG_FILE --sks_name "${NAMES[0]}" &
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config $CONFIG_FILE --sks_name "${NAMES[1]}" &
# CUDA_VISIBLE_DEVICES=4,5 python train.py --config $CONFIG_FILE --sks_name "${NAMES[2]}" &
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config $CONFIG_FILE --sks_name "${NAMES[3]}" 
# wait

# CONFIG_FILE="./config/64.yaml"

# cd $WORKING_FOLDER/YoChameleon
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config $CONFIG_FILE --sks_name "${NAMES[0]}" &
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config $CONFIG_FILE --sks_name "${NAMES[1]}" &
# CUDA_VISIBLE_DEVICES=4,5 python train.py --config $CONFIG_FILE --sks_name "${NAMES[2]}" &
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config $CONFIG_FILE --sks_name "${NAMES[3]}" 
# wait

# CONFIG_FILE="./config/32.yaml"

# cd $WORKING_FOLDER/YoChameleon
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config $CONFIG_FILE --sks_name "${NAMES[0]}" &
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config $CONFIG_FILE --sks_name "${NAMES[1]}" &
# CUDA_VISIBLE_DEVICES=4,5 python train.py --config $CONFIG_FILE --sks_name "${NAMES[2]}" &
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config $CONFIG_FILE --sks_name "${NAMES[3]}" 
# wait

# CONFIG_FILE="./config/ablation-32.yaml"

# cd $WORKING_FOLDER/YoChameleon
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config $CONFIG_FILE --sks_name "${NAMES[0]}" &
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config $CONFIG_FILE --sks_name "${NAMES[1]}" &
# CUDA_VISIBLE_DEVICES=4,5 python train.py --config $CONFIG_FILE --sks_name "${NAMES[2]}" &
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config $CONFIG_FILE --sks_name "${NAMES[3]}" 
# wait


CONFIG_FILE="./config/ablation-2000.yaml"

cd $WORKING_FOLDER/YoChameleon
CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/ablation-2000.yaml --sks_name "${NAMES[0]}" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/ablation-2000.yaml --sks_name "${NAMES[1]}" &
CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/ablation-2000.yaml --sks_name "${NAMES[2]}" &
CUDA_VISIBLE_DEVICES=6,7 python train.py --config ./config/ablation-2000.yaml --sks_name "${NAMES[3]}" 
wait

CONFIG_FILE="./config/ablation-1000.yaml"

cd $WORKING_FOLDER/YoChameleon
CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/ablation-1000.yaml --sks_name "${NAMES[0]}" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/ablation-1000.yaml --sks_name "${NAMES[1]}" &
CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/ablation-1000.yaml --sks_name "${NAMES[2]}" &
CUDA_VISIBLE_DEVICES=6,7 python train.py --config ./config/ablation-1000.yaml --sks_name "${NAMES[3]}" 
wait
# cd $WORKING_FOLDER/YoChameleon
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/ablation-500.yaml --sks_name "${NAMES[1]}" &
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/ablation-1000.yaml --sks_name "${NAMES[1]}" &
# CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/ablation-2000.yaml --sks_name "${NAMES[1]}" &
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config ./config/ablation-3000.yaml --sks_name "${NAMES[1]}" 
# wait

# cd $WORKING_FOLDER/YoChameleon
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/ablation-500.yaml --sks_name "${NAMES[2]}" &
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/ablation-1000.yaml --sks_name "${NAMES[2]}" &
# CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/ablation-2000.yaml --sks_name "${NAMES[2]}" &
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config ./config/ablation-3000.yaml --sks_name "${NAMES[2]}" 
# wait

