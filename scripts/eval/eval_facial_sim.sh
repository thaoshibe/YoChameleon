#!/bin/bash
cd ../../evaluation/

EXP_FOLDER="64-5000"
FAKE_FOLDER_BASE="/sensei-fs/users/thaon/code/generated_images"

# Define the real folder
REAL_FOLDER="/mnt/localssd/code/data/yollava-data/train/thao"

# Define an array of fake folders
FAKE_FOLDERS=(
    # Local and aligned folders
    # "/mnt/localssd/code/data/dathao_algined"
    # "/mnt/localssd/code/data/yollava-data/train/khanhvy"
    # "/mnt/localssd/code/data/yollava-data/train/thao/negative_example"
    # Generated image sets using FAKE_FOLDER_BASE
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/1000"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/2000"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/3000"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/4000"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/4050"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/4100"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/4150"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/4200"
)

# Loop through each fake folder and run the Python evaluation script
for FAKE_FOLDER in "${FAKE_FOLDERS[@]}"
do
    echo "Running evaluation with fake folder: $FAKE_FOLDER"
    python insightface_verify.py --real_folder "$REAL_FOLDER" --fake_folder "$FAKE_FOLDER"
done

echo "All evaluations completed!"
