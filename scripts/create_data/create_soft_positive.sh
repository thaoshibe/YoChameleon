cd create_training_data/retrieve_negative
# List of names or folders to process
# NAMES=("bo" "duck-banana" "marie-cat" "pusheen-cup" "thuytien"
#        "brown-duck" "dug" "mydieu" "shiba-black" "tokyo-keyboard"
#        "butin" "elephant" "neurips-cup" "shiba-gray" "toodles-galore"
#        "cat-cup" "fire" "nha-tho-hanoi" "shiba-sleep" "viruss"
#        "chua-thien-mu" "henry" "nha-tho-hcm" "shiba-yellow" "water"
#        "ciin" "khanhvy" "oong" "thao" "willinvietnam"
#        "denisdang" "lamb" "phuc-map" "thap-but" "yellow-duck"
#        "dragon" "mam" "pig-cup" "thap-cham" "yuheng")

NAMES=("bo" "mam" "thuytien" "viruss" "ciin" "khanhvy" "oong" "thao" "willinvietnam" "denisdang" "phuc-map" "yuheng")

# Loop through each folder
for NAME in "${NAMES[@]}"; do
  # Define the positive image folder based on the name
  POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
  # Define the output file path for the JSON result
  OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
  
  # Log which folder is being processed
  echo "Processing folder: ${NAME}"
  
  # Execute the Python script with the required arguments
  python create_conversation_by_ranking.py \
    --input_folder "$POSITIVE_IMAGE_FOLDER" \
    --save_folder "$OUTPUT_FILE" \
    --version '1000' \
    --num_of_real_images -100 \
    --token_length 16 \
    --spacing 1 \
    --negative_image True \
    --limit_negative 1000 \
    --consistent_prompt True # This mean, we will use the same prompt for all negative images and positive images
done


###--- Create image generation data ---###
# cd create_training_data/retrieve_negative
# # List of names or folders to process
# NAMES=("bo" "duck-banana" "marie-cat" "pusheen-cup" "thuytien"
#        "brown-duck" "dug" "mydieu" "shiba-black" "tokyo-keyboard"
#        "butin" "elephant" "neurips-cup" "shiba-gray" "toodles-galore"
#        "cat-cup" "fire" "nha-tho-hanoi" "shiba-sleep" "viruss"
#        "chua-thien-mu" "henry" "nha-tho-hcm" "shiba-yellow" "water"
#        "ciin" "khanhvy" "oong" "thao" "willinvietnam"
#        "denisdang" "lamb" "phuc-map" "thap-but" "yellow-duck"
#        "dragon" "mam" "pig-cup" "thap-cham" "yuheng")

# # Loop through each folder
# for NAME in "${NAMES[@]}"; do
#   # Define the positive image folder based on the name
#   POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  
#   # Define the output file path for the JSON result
#   OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
  
#   # Log which folder is being processed
#   echo "Processing folder: ${NAME}"
  
#   # Execute the Python script with the required arguments
#   python create_conversation_by_ranking.py \
#     --input_folder "$POSITIVE_IMAGE_FOLDER" \
#     --save_folder "$OUTPUT_FILE" \
#     --version image_gen_positive_only \
#     --num_of_real_images 100 \
#     --token_length 16 \
#     --spacing 16
# done