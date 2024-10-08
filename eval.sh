NAME="thao"

INPUT_FOLDER="/mnt/localssd/code/data/yollava-data/train/${NAME}"

# -- Thao's v3 -- epoch 20
# FAKE_FOLDER="/sensei-fs/users/thaon/generated_images/${NAME}-v3/20/"
# python evaluation/face_verification.py --real_folder $INPUT_FOLDER --fake_folder $FAKE_FOLDER 

# # -- Thao's real images
# FAKE_FOLDER="/sensei-fs/users/thaon/generated_images/${NAME}-real/20/"
# python evaluation/face_verification.py --real_folder $INPUT_FOLDER --fake_folder $FAKE_FOLDER

# # -- Thao's v1 -- epoch 20
# FAKE_FOLDER="/sensei-fs/users/thaon/generated_images/${NAME}-v1/20/"
# python evaluation/face_verification.py --real_folder $INPUT_FOLDER --fake_folder $FAKE_FOLDER 

# -- Thao's v1 -- epoch 50
# FAKE_FOLDER="/sensei-fs/users/thaon/generated_images/${NAME}-v1/50/"
# python evaluation/face_verification.py --real_folder $INPUT_FOLDER --fake_folder $FAKE_FOLDER 


# -- Thao's v2 -- epoch 20
FAKE_FOLDER="/sensei-fs/users/thaon/generated_images/${NAME}-real/20/"
python evaluation/face_verification.py --real_folder $INPUT_FOLDER --fake_folder $FAKE_FOLDER 

# ---- BASELINE
# python evaluation/face_verification.py --real_folder /mnt/localssd/code/data/yollava-data/train/thao --fake_folder /sensei-fs/users/thaon/generated_images/chameleon
# python evaluation/face_verification.py --real_folder /mnt/localssd/code/data/yollava-data/train/thao --fake_folder /sensei-fs/users/thaon/generated_images/emu2