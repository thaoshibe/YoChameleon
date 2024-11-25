USER="$(whoami)"
echo "USER"
echo $USER

NAMES=("ciin" "thuytien" "khanhvy")

export WANDB_API_KEY="563710e55fec9aac8f27c7ab80cfed931a2096f5"

### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
###                                                               #
###        THAO: REMEMEBER TO CHANGE THE CONFIG FILE HERE         #
###                                                               #
### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #

# CONFIG_FILE="./config/da-w.yaml"

# cd $WORKING_FOLDER/YoChameleon
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config $CONFIG_FILE --sks_name "${NAMES[0]}" &
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config $CONFIG_FILE --sks_name "${NAMES[1]}" &
# CUDA_VISIBLE_DEVICES=4,5 python train.py --config $CONFIG_FILE --sks_name "${NAMES[2]}" &
# # CUDA_VISIBLE_DEVICES=6,7 python train.py --config $CONFIG_FILE --sks_name "${NAMES[3]}" 
# wait

CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/100.yaml --sks_name "${NAMES[0]}" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/100.yaml --sks_name "${NAMES[1]}" &
CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/100.yaml --sks_name "${NAMES[2]}" &
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config $CONFIG_FILE --sks_name "${NAMES[3]}" 
wait

CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/200.yaml --sks_name "${NAMES[0]}" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/200.yaml --sks_name "${NAMES[1]}" &
CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/200.yaml --sks_name "${NAMES[2]}" &
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config $CONFIG_FILE --sks_name "${NAMES[3]}" 
wait

CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/400.yaml --sks_name "${NAMES[0]}" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/400.yaml --sks_name "${NAMES[1]}" &
CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/400.yaml --sks_name "${NAMES[2]}" &
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config $CONFIG_FILE --sks_name "${NAMES[3]}" 
wait


CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/1000.yaml --sks_name "${NAMES[0]}" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/1000.yaml --sks_name "${NAMES[1]}" &
CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/1000.yaml --sks_name "${NAMES[2]}" &
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config $CONFIG_FILE --sks_name "${NAMES[3]}" 
wait

CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/600.yaml --sks_name "${NAMES[0]}" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/600.yaml --sks_name "${NAMES[1]}" &
CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/600.yaml --sks_name "${NAMES[2]}" &
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config $CONFIG_FILE --sks_name "${NAMES[3]}" 
wait

CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/800.yaml --sks_name "${NAMES[0]}" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/800.yaml --sks_name "${NAMES[1]}" &
CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/800.yaml --sks_name "${NAMES[2]}" &
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config $CONFIG_FILE --sks_name "${NAMES[3]}" 
wait