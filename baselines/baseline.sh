# CUDA_VISIBLE_DEVICES=0,1 python anole.py --start 0 --end 10 --image_prompt True &
# CUDA_VISIBLE_DEVICES=2,3 python anole.py --start 10 --end 20 --image_prompt True &
# CUDA_VISIBLE_DEVICES=4,5 python anole.py --start 20 --end 30 --image_prompt True &
# CUDA_VISIBLE_DEVICES=6,7 python anole.py --start 30 --end 40 --image_prompt True


#!/bin/bash

# Define the SUBJECT_NAMES array
SUBJECT_NAMES=("bo" "duck-banana" "marie-cat" "pusheen-cup" "brown-duck" "dug" "mydieu" "shiba-black" 
    "tokyo-keyboard" "butin" "elephant" "neurips-cup" "shiba-gray" "toodles-galore" "cat-cup" "fire"
    "nha-tho-hanoi" "shiba-sleep" "viruss" "chua-thien-mu" "henry" "nha-tho-hcm" "shiba-yellow" "water"
    "ciin" "khanhvy" "oong" "thao" "willinvietnam" "denisdang" "lamb" "phuc-map" "thap-but" 
    "yellow-duck" "dragon" "mam" "pig-cup" "thap-cham" "yuheng" "thuytien")

# Loop through each name in the SUBJECT_NAMES array
for NAME in "${SUBJECT_NAMES[@]}"; do
    # Run the Python script with the current NAME
    python clip_image_similarity.py --real_folder "/mnt/localssd/code/data/yochameleon-data/train/$NAME" --fake_folder "/sensei-fs/users/thaon/generated_images/chameleon/image_prompt/$NAME"
done
