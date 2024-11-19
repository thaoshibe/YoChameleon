# ------------------------------------------ #
#                                            #
#          Simple text conv data             #
#                                            #
# ------------------------------------------ #

cd create_training_data/dense_caption

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
#                                                          #
#          This block will use Chameleon/ Anole            #
#                                                          #
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #

# List of names or folders to process -- For human
# NAMES=("thuytien" "viruss" "ciin" "khanhvy" "oong" "thao" "willinvietnam" "denisdang" "phuc-map" "yuheng")

# for NAME in "${NAMES[@]}"; do
#   POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
#   NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
#   OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
#   echo "Processing folder: ${NAME}"
#   python chameleon_self_label.py \
#     --input_image_folder "$POSITIVE_IMAGE_FOLDER" \
#     --prompt_file_path ./system-prompts/text-conversation.txt \
#     --output_file "$OUTPUT_FILE" \
#     --text_conversation \
#     --human \
#     --limit 5
# done

# List of names or folders to process -- For object
# NAMES=("bo" "duck-banana" "marie-cat" "pusheen-cup"
#        "brown-duck" "dug" "mydieu" "shiba-black" "tokyo-keyboard"
#        "butin" "elephant" "neurips-cup" "shiba-gray" "toodles-galore"
#        "cat-cup" "fire" "nha-tho-hanoi" "shiba-sleep"
#        "chua-thien-mu" "henry" "nha-tho-hcm" "shiba-yellow" "water"
#        "lamb" "thap-but" "yellow-duck"
#        "dragon" "mam" "pig-cup" "thap-cham")

# Loop through each folder
# for NAME in "${NAMES[@]}"; do
#   # Define the positive image folder based on the name
#   POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
#   NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
  
#   OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
#   echo "Processing folder: ${NAME}"
  
#   python chameleon_self_label.py \
#     --input_image_folder "$POSITIVE_IMAGE_FOLDER" \
#     --prompt_file_path ./system-prompts/text-conversation.txt \
#     --output_file "$OUTPUT_FILE" \
#     --text_conversation \
#     --limit 5
# done


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
#                                                          #
#          ATTENTION: THIS WILL CALL GPT-4o                #
#                                                          #
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #

# List of names or folders to process -- For human
NAMES=("thuytien" "viruss" "ciin" "khanhvy" "oong" "thao" "willinvietnam" "denisdang" "phuc-map" "yuheng")

for NAME in "${NAMES[@]}"; do
  POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
  OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
  echo "Processing folder: ${NAME}"
  python gpt4o-api.py \
    --input_image_folder "$POSITIVE_IMAGE_FOLDER" \
    --prompt_file_path ./system-prompts/text-conversation.txt \
    --output_file "$OUTPUT_FILE" \
    --text_conversation \
    --human \
    --limit 5
done

# List of names or folders to process -- For object
NAMES=("bo" "duck-banana" "marie-cat" "pusheen-cup"
       "brown-duck" "dug" "mydieu" "shiba-black" "tokyo-keyboard"
       "butin" "elephant" "neurips-cup" "shiba-gray" "toodles-galore"
       "cat-cup" "fire" "nha-tho-hanoi" "shiba-sleep"
       "chua-thien-mu" "henry" "nha-tho-hcm" "shiba-yellow" "water"
       "lamb" "thap-but" "yellow-duck"
       "dragon" "mam" "pig-cup" "thap-cham")

# Loop through each folder
for NAME in "${NAMES[@]}"; do
  # Define the positive image folder based on the name
  POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
  
  OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
  echo "Processing folder: ${NAME}"
  
  python gpt4o-api.py \
    --input_image_folder "$POSITIVE_IMAGE_FOLDER" \
    --prompt_file_path ./system-prompts/text-conversation.txt \
    --output_file "$OUTPUT_FILE" \
    --text_conversation \
    --limit 5
done