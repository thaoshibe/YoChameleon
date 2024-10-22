###--- Retrieve negative examples ---###
# cd create_training_data/conversation_data
# NAMES=("bo" "duck-banana" "marie-cat" "pusheen-cup" "thuytien"
#        "brown-duck" "dug" "mydieu" "shiba-black" "tokyo-keyboard"
#        "butin" "elephant" "neurips-cup" "shiba-gray" "toodles-galore"
#        "cat-cup" "fire" "nha-tho-hanoi" "shiba-sleep" "viruss"
#        "chua-thien-mu" "henry" "nha-tho-hcm" "shiba-yellow" "water"
#        "ciin" "khanhvy" "oong" "thao" "willinvietnam"
#        "denisdang" "lamb" "phuc-map" "thap-but" "yellow-duck"
#        "dragon" "mam" "pig-cup" "thap-cham" "yuheng")

# for NAME in "${NAMES[@]}"; do
#   INPUT_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
#   SAVE_FOLDER="${INPUT_FOLDER}/negative_example"
#   LIMIT=5000 # Number of negative examples to retrieve
#   echo "Processing folder: ${NAME}"
  
#   python create_training_data/retrieve_negative/load_similar_example.py \
#     --input_folder $INPUT_FOLDER \
#     --save_folder $SAVE_FOLDER \
#     --limit $LIMIT \
#     --origin "l2"
# done

##--- Create recognition data ---###
# cd create_training_data/conversation_data
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
  
#   # Define the negative image folder (assuming it's fixed or can vary similarly)
#   NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
  
#   # Define the output file path for the JSON result
#   OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
  
#   # Log which folder is being processed
#   echo "Processing folder: ${NAME}"
  
#   # Execute the Python script with the required arguments
#   python create_conversation.py \
#     --positive_image_folder "$POSITIVE_IMAGE_FOLDER" \
#     --negative_image_folder "$NEGATIVE_IMAGE_FOLDER" \
#     --output_file "$OUTPUT_FILE" \
#     --limit_positive 5 \
#     --limit_negative 100
# done

###--- Create image generation data ---###
cd create_training_data/retrieve_negative
# List of names or folders to process
NAMES=("bo" "duck-banana" "marie-cat" "pusheen-cup" "thuytien"
       "brown-duck" "dug" "mydieu" "shiba-black" "tokyo-keyboard"
       "butin" "elephant" "neurips-cup" "shiba-gray" "toodles-galore"
       "cat-cup" "fire" "nha-tho-hanoi" "shiba-sleep" "viruss"
       "chua-thien-mu" "henry" "nha-tho-hcm" "shiba-yellow" "water"
       "ciin" "khanhvy" "oong" "thao" "willinvietnam"
       "denisdang" "lamb" "phuc-map" "thap-but" "yellow-duck"
       "dragon" "mam" "pig-cup" "thap-cham" "yuheng")

# Loop through each folder
for NAME in "${NAMES[@]}"; do
  # Define the positive image folder based on the name
  POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  
  # Define the output file path for the JSON result
  OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
  
  # Log which folder is being processed
  echo "Processing folder: ${NAME}"
  
  # Execute the Python script with the required arguments
  python create_conversation_by_ranking.py \
    --input_folder "$POSITIVE_IMAGE_FOLDER" \
    --save_folder "$OUTPUT_FILE" \
    --version image_gen_positive_only \
    --num_of_real_images 100 \
    --token_length 16 \
    --spacing 16
done

###--- Simple conversation data ---###

# cd create_training_data/dense_caption

# # List of names or folders to process -- For human
# NAMES=("thuytien" "viruss" "ciin" "khanhvy" "oong" "thao" "willinvietnam" "denisdang" "phuc-map" "yuheng")

# # Loop through each folder
# for NAME in "${NAMES[@]}"; do
#   # Define the positive image folder based on the name
#   POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  
#   # Define the negative image folder (assuming it's fixed or can vary similarly)
#   NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
  
#   # Define the output file path for the JSON result
#   OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
  
#   # Log which folder is being processed
#   echo "Processing folder: ${NAME}"
  
#   # Execute the Python script with the required arguments
#   python gpt4o-api.py \
#     --input_image_folder "$POSITIVE_IMAGE_FOLDER" \
#     --prompt_file_path ./system-prompts/text-conversation.txt \
#     --output_file "$OUTPUT_FILE" \
#     --text_conversation \
#     --human \
#     --limit 5
# done

# # List of names or folders to process -- For object
# NAMES=("bo" "duck-banana" "marie-cat" "pusheen-cup"
#        "brown-duck" "dug" "mydieu" "shiba-black" "tokyo-keyboard"
#        "butin" "elephant" "neurips-cup" "shiba-gray" "toodles-galore"
#        "cat-cup" "fire" "nha-tho-hanoi" "shiba-sleep"
#        "chua-thien-mu" "henry" "nha-tho-hcm" "shiba-yellow" "water"
#        "lamb" "thap-but" "yellow-duck"
#        "dragon" "mam" "pig-cup" "thap-cham")

# # Loop through each folder
# for NAME in "${NAMES[@]}"; do
#   # Define the positive image folder based on the name
#   POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  
#   # Define the negative image folder (assuming it's fixed or can vary similarly)
#   NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
  
#   # Define the output file path for the JSON result
#   OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
  
#   # Log which folder is being processed
#   echo "Processing folder: ${NAME}"
  
#   # Execute the Python script with the required arguments
#   python gpt4o-api.py \
#     --input_image_folder "$POSITIVE_IMAGE_FOLDER" \
#     --prompt_file_path ./system-prompts/text-conversation.txt \
#     --output_file "$OUTPUT_FILE" \
#     --text_conversation \
#     --limit 5
# done