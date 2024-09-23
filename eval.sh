NAME="yuheng"

INPUT_FOLDER="/mnt/localssd/code/data/yollava-data/train/${NAME}"
FAKE_FOLDER="/sensei-fs/users/thaon/generated_images/${NAME}-v1/20/"

python evaluation/face_verification.py --real_folder $INPUT_FOLDER --fake_folder $FAKE_FOLDER 