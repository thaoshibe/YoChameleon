#!/bin/bash

# Define the real folder
REAL_FOLDER="/mnt/localssd/code/data/yollava-data/train/thao"

# Define the base path for generated image folders
FAKE_FOLDER_BASE="/sensei-fs/users/thaon/code/generated_images/"

# Define an array of fake folders
FAKE_FOLDERS=(
    # Local and aligned folders
    # "/mnt/localssd/code/data/dathao_algined"
    # "/mnt/localssd/code/data/yollava-data/train/thuytien"
    # "/mnt/localssd/code/data/yollava-data/train/thao/negative_example"
    # Generated image sets using FAKE_FOLDER_BASE
    # "${FAKE_FOLDER_BASE}neg-128-16-v2/300"
    # "${FAKE_FOLDER_BASE}neg-128-16-v2/500"
    # "${FAKE_FOLDER_BASE}neg-128-16-v2/550"
    # "${FAKE_FOLDER_BASE}neg-128-16-v2/600"

    # "${FAKE_FOLDER_BASE}neg-64-8-v2/300"
    # "${FAKE_FOLDER_BASE}neg-64-8-v2/500"
    # "${FAKE_FOLDER_BASE}neg-64-8-v2/550"
    # "${FAKE_FOLDER_BASE}neg-64-8-v2/600"

    # "${FAKE_FOLDER_BASE}neg-16-4-v2/300"
    # "${FAKE_FOLDER_BASE}neg-16-4-v2/500"
    # "${FAKE_FOLDER_BASE}neg-16-4-v2/550"
    # "${FAKE_FOLDER_BASE}neg-16-4-v2/600"

    "${FAKE_FOLDER_BASE}neg-64-64-v2/300"
    "${FAKE_FOLDER_BASE}neg-64-64-v2/500"
    "${FAKE_FOLDER_BASE}neg-64-64-v2/550"
    "${FAKE_FOLDER_BASE}neg-64-64-v2/600"
)

# Loop through each fake folder and run the Python evaluation script
for FAKE_FOLDER in "${FAKE_FOLDERS[@]}"
do
    echo "Running evaluation with fake folder: $FAKE_FOLDER"
    python evaluation/insightface_verify.py --real_folder "$REAL_FOLDER" --fake_folder "$FAKE_FOLDER"
done

echo "All evaluations completed!"
