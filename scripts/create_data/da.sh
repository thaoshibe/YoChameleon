cd create_training_data/retrieve_negative
# List of names or folders to process

NAMES=("bo" "mam" "thao")

# Loop through each folder
for NAME in "${NAMES[@]}"; do
  # Define the positive image folder based on the name
  POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/300/${NAME}"
  NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/300/${NAME}/negative_example"
  # Define the output file path for the JSON result
  OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
  
  # Log which folder is being processed
  echo "Processing folder: ${NAME}"
  
  # Execute the Python script with the required arguments
  python create_conversation_by_ranking.py \
    --input_folder "$POSITIVE_IMAGE_FOLDER" \
    --save_folder "$OUTPUT_FILE" \
    --version 'real' \
    --num_of_real_images 100 \
    --token_length 16 \
    --spacing 16 \
    --limit_negative 1000 \
    --consistent_prompt True # This mean, we will use the same prompt for all negative images and positive images
done