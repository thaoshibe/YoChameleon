NAMES=("bo" "duck-banana" "marie-cat" "pusheen-cup" "thuytien"
       "brown-duck" "dug" "mydieu" "shiba-black" "tokyo-keyboard"
       "butin" "elephant" "neurips-cup" "shiba-gray" "toodles-galore"
       "cat-cup" "fire" "nha-tho-hanoi" "shiba-sleep" "viruss"
       "chua-thien-mu" "henry" "nha-tho-hcm" "shiba-yellow" "water"
       "ciin" "khanhvy" "oong" "thao" "willinvietnam"
       "denisdang" "lamb" "phuc-map" "thap-but" "yellow-duck"
       "dragon" "mam" "pig-cup" "thap-cham" "yuheng")

process_image() {
  NAME=$1
  INPUT_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  SAVE_FOLDER="${INPUT_FOLDER}/negative_example"
  LIMIT=5000
  echo "Processing folder: ${NAME}"

  python create_training_data/retrieve_negative/load_similar_example.py \
    --input_folder "$INPUT_FOLDER" \
    --save_folder "$SAVE_FOLDER" \
    --limit "$LIMIT" \
    --origin "l2"
}

export -f process_image # Ensure the function is exported to be available for xargs

# Use `xargs` to run the process_image function in parallel
echo "${NAMES[@]}" | tr ' ' '\n' | xargs -n 1 -P 6 -I {} bash -c 'process_image "$@"' _ {}
