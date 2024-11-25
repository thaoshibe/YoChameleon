##--- Retrieve negative examples ---###
# cd create_training_data/conversation_data
# NAMES=("bo" "duck-banana" "marie-cat" "pusheen-cup" "thuytien"
#        "brown-duck" "dug" "mydieu" "shiba-black" "tokyo-keyboard"
#        "butin" "elephant" "neurips-cup" "shiba-gray" "toodles-galore"
#        "cat-cup" "fire" "nha-tho-hanoi" "shiba-sleep" "viruss"
#        "chua-thien-mu" "henry" "nha-tho-hcm" "shiba-yellow" "water"
#        "ciin" "khanhvy" "oong" "thao" "willinvietnam"
#        "denisdang" "lamb" "phuc-map" "thap-but" "yellow-duck"
#        "dragon" "mam" "pig-cup" "thap-cham" "yuheng")
NAMES=("thuytien" "viruss" "ciin" "khanhvy" "oong" "thao" "willinvietnam" "denisdang" "phuc-map" "yuheng"
       "bo" "butin" "neurips-cup"
       "nha-tho-hanoi"
       "chua-thien-mu" "henry" "nha-tho-hcm" "thap-but"
       "mam" "thap-cham")

for NAME in "${NAMES[@]}"; do
  INPUT_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  SAVE_FOLDER="${INPUT_FOLDER}/negative_example"
  LIMIT=1000 # Number of negative examples to retrieve
  echo "Processing folder: ${NAME}"
  
  python create_training_data/retrieve_negative/load_similar_example.py \
    --input_folder $INPUT_FOLDER \
    --save_folder $SAVE_FOLDER \
    --limit $LIMIT \
    --origin "l2"
done
