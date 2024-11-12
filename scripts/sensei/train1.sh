USER="$(whoami)"
echo "USER"
echo $USER

NAMES=("bo" "duck-banana" "marie-cat" "pusheen-cup" "brown-duck" "dug" "mydieu" "shiba-black")
# NAMES=("tokyo-keyboard" "butin" "elephant" "neurips-cup")
# NAMES=("shiba-gray" "toodles-galore" "cat-cup" "fire")
# NAMES=("nha-tho-hanoi" "shiba-sleep" "viruss" "chua-thien-mu")
# NAMES=("henry" "nha-tho-hcm" "shiba-yellow" "water")
# NAMES=("ciin" "khanhvy" "oong" "thao")
# NAMES=("willinvietnam" "denisdang" "lamb" "phuc-map")
# NAMES=("thap-but" "yellow-duck" "dragon" "mam")
# NAMES=("pig-cup" "thap-cham" "yuheng", "thuytien")

WORKING_FOLDER="/mnt/localssd/code"
CODE_FOLDER="/sensei-fs/users/$USER/code/YoChameleon"
export WANDB_API_KEY="563710e55fec9aac8f27c7ab80cfed931a2096f5"

### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
###                                                               #
###        THAO: REMEMEBER TO CHANGE THE CONFIG FILE HERE         #
###                                                               #
### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #

CONFIG_FILE="./config/selfprompting.yaml"

### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
### Setting up -- this part, you might not need to worry about ...

if [ ! -d "/home/user/" ]; then
  sudo mkdir /home/user/
fi

if [ ! -d "/home/$USER" ]; then
  sudo mkdir /home/$USER
fi

sudo chmod 777 -R /home/

echo "Launching training script"
mkdir -p $WORKING_FOLDER
mkdir -p $WORKING_FOLDER/data

cd $WORKING_FOLDER
cp -r $CODE_FOLDER $WORKING_FOLDER

DATA_ZIP_FILE="/sensei-fs/users/$USER/data/yochameleon-data.zip"

cp -r $DATA_ZIP_FILE $WORKING_FOLDER/data
cd $WORKING_FOLDER/data
unzip $WORKING_FOLDER/data/yochameleon-data.zip

cd $WORKING_FOLDER/YoChameleon
bash scripts/install.sh

### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
###                                                               #
###                    CREATE TRAINING DATA                       #
###                                                               #
### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #

### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
###  Retrieve negative image

process_image() {
  NAME=$1
  INPUT_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  SAVE_FOLDER="${INPUT_FOLDER}/negative_example"
  LIMIT=1000
  echo "Processing folder: ${NAME}"
  cd /mnt/localssd/code/YoChameleon

  python create_training_data/retrieve_negative/load_similar_example.py \
    --input_folder "$INPUT_FOLDER" \
    --save_folder "$SAVE_FOLDER" \
    --limit "$LIMIT" \
    --origin "l2"
}

export -f process_image # Ensure the function is exported to be available for xargs
echo "${NAMES[@]}" | tr ' ' '\n' | xargs -n 1 -P 6 -I {} bash -c 'process_image "$@"' _ {}

### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
###  Create recognition data

for NAME in "${NAMES[@]}"; do
  PROMPT_FILE_PATH="./create_training_data/conversation_data/template-answer/recognition-chameleon.json"
  POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
  RANDOM_NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/random_negative_example"
  OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
  echo "Processing folder: ${NAME}"
  cd /mnt/localssd/code/YoChameleon
  mkdir -p "/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
  python create_training_data/conversation_data/create_conversation.py \
    --prompt_file_path "$PROMPT_FILE_PATH" \
    --positive_image_folder "$POSITIVE_IMAGE_FOLDER" \
    --negative_image_folder "$NEGATIVE_IMAGE_FOLDER" \
    --random_negative_image_folder "$RANDOM_NEGATIVE_IMAGE_FOLDER" \
    --output_file "$OUTPUT_FILE" \
    --limit_positive 5 \
    --limit_negative 100
done

### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
###  Soft positive

for NAME in "${NAMES[@]}"; do
  POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
  OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
  echo "Processing folder: ${NAME}"
  cd /mnt/localssd/code/YoChameleon
  python create_training_data/retrieve_negative/create_conversation_by_ranking.py \
    --input_folder "$POSITIVE_IMAGE_FOLDER" \
    --save_folder "$OUTPUT_FILE" \
    --version '1000' \
    --num_of_real_images -100 \
    --token_length 16 \
    --spacing 1 \
    --negative_image True \
    --limit_negative 1000 \
    --consistent_prompt True 
done

### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
###  Simple text conv data

for NAME in "${NAMES[@]}"; do
  # Define the positive image folder based on the name
  POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
  OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
  echo "Processing folder: ${NAME}"
  cd /mnt/localssd/code/YoChameleon
  python create_training_data/dense_caption/gpt4o-api.py \
    --input_image_folder "$POSITIVE_IMAGE_FOLDER" \
    --prompt_file_path ./create_training_data/dense_caption/system-prompts/text-conversation.txt \
    --output_file "$OUTPUT_FILE" \
    --text_conversation \
    --limit 5
done

## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
##                                                               #
##                   TRAIN - TRAIN - TRAIN                       #
##                                                               #
## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #

cd $WORKING_FOLDER/YoChameleon
CUDA_VISIBLE_DEVICES=0,1 python train.py --config $CONFIG_FILE --sks_name "${NAMES[0]}" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config $CONFIG_FILE --sks_name "${NAMES[1]}" &
CUDA_VISIBLE_DEVICES=4,5 python train.py --config $CONFIG_FILE --sks_name "${NAMES[2]}" &
CUDA_VISIBLE_DEVICES=6,7 python train.py --config $CONFIG_FILE --sks_name "${NAMES[3]}" 
wait

cd $WORKING_FOLDER/YoChameleon
CUDA_VISIBLE_DEVICES=0,1 python train.py --config $CONFIG_FILE --sks_name "${NAMES[4]}" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config $CONFIG_FILE --sks_name "${NAMES[5]}" &
CUDA_VISIBLE_DEVICES=4,5 python train.py --config $CONFIG_FILE --sks_name "${NAMES[6]}" &
CUDA_VISIBLE_DEVICES=6,7 python train.py --config $CONFIG_FILE --sks_name "${NAMES[7]}" 
wait

