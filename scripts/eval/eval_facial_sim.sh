# Define the list of SKS_NAMES
SKS_NAMES=("yuheng" "khanhvy")

# Please change this according to your setting name
EXP_FOLDER="1000neg-text-recog"
FAKE_FOLDER_BASE="/sensei-fs/users/thaon/code/generated_images"

# Loop through each SKS_NAME
for SKS_NAME in "${SKS_NAMES[@]}"
do
    # Define the real folder for the current SKS_NAME
    REAL_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${SKS_NAME}"

    # Define an array of fake folders for the current SKS_NAME
    FAKE_FOLDERS=(
        # Local and aligned folders
        # "/mnt/localssd/code/data/dathao_algined"
        # "/mnt/localssd/code/data/yochameleon-data/train/${SKS_NAME}/"
        # "/mnt/localssd/code/data/yollava-data/train/thao/negative_example"
        # Generated image sets using FAKE_FOLDER_BASE
        "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/5/${SKS_NAME}"
        "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/10/${SKS_NAME}"
        "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/15/${SKS_NAME}"
        "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/20/${SKS_NAME}"
        "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/25/${SKS_NAME}"
        "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/30/${SKS_NAME}"
        # "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/700/${SKS_NAME}"
        # "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/800/${SKS_NAME}"
        # "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/900/${SKS_NAME}"
        # "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/1100/${SKS_NAME}"
        # "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/1200/${SKS_NAME}"
        # "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/1300/${SKS_NAME}"
        # "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/1400/${SKS_NAME}"
        # "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/1500/${SKS_NAME}"
        # "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/1600/${SKS_NAME}"
        # "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/1700/${SKS_NAME}"
    )

    # Loop through each fake folder for the current SKS_NAME and run the evaluation
    for FAKE_FOLDER in "${FAKE_FOLDERS[@]}"
    do
        echo "Running evaluation for ${SKS_NAME} with fake folder: $FAKE_FOLDER"
        python evaluation/insightface_verify.py \
            --real_folder "$REAL_FOLDER" \
            --fake_folder "$FAKE_FOLDER" --output_file "$FAKE_FOLDER_BASE/facial_sim.txt"
    done

done

echo "All evaluations completed!"

# EXP_FOLDER="1000neg-text-recog"
# FAKE_FOLDER_BASE="/sensei-fs/users/thaon/code/generated_images"

# # Loop through each SKS_NAME
# for SKS_NAME in "${SKS_NAMES[@]}"
# do
#     # Define the real folder for the current SKS_NAME
#     REAL_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${SKS_NAME}"

#     # Define an array of fake folders for the current SKS_NAME
#     FAKE_FOLDERS=(
#         # Local and aligned folders
#         # "/mnt/localssd/code/data/dathao_algined"
#         # "/mnt/localssd/code/data/yochameleon-data/train/${SKS_NAME}/"
#         # "/mnt/localssd/code/data/yollava-data/train/thao/negative_example"
#         # Generated image sets using FAKE_FOLDER_BASE
#         "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/5/${SKS_NAME}"
#         "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/10/${SKS_NAME}"
#         "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/15/${SKS_NAME}"
#         "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/15ft/${SKS_NAME}"
#         "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/17ft/${SKS_NAME}"
#         "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/20ft/${SKS_NAME}"
#     )

#     # Loop through each fake folder for the current SKS_NAME and run the evaluation
#     for FAKE_FOLDER in "${FAKE_FOLDERS[@]}"
#     do
#         echo "Running evaluation for ${SKS_NAME} with fake folder: $FAKE_FOLDER"
#         python evaluation/insightface_verify.py \
#             --real_folder "$REAL_FOLDER" \
#             --fake_folder "$FAKE_FOLDER" --output_file "$FAKE_FOLDER_BASE/facial_sim.txt"
#     done

# done

# echo "All evaluations completed!"
