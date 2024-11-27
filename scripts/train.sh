# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
#                                                               #
#           REMEMEBER TO CHANGE THE CONFIG FILE HERE            #
#                                                               #
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #

CONFIG_FILE="./config/basic.yaml"

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #

CUDA_VISIBLE_DEVICES=0,1 python train.py --config $CONFIG_FILE --sks_name "thao" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config $CONFIG_FILE --sks_name "yuheng" &
CUDA_VISIBLE_DEVICES=4,5 python train.py --config $CONFIG_FILE --sks_name "ciin" &
CUDA_VISIBLE_DEVICES=6,7 python train.py --config $CONFIG_FILE --sks_name "khanhvy" 
wait

CUDA_VISIBLE_DEVICES=0,1 python train.py --config $CONFIG_FILE --sks_name "oong" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config $CONFIG_FILE --sks_name "willinvietnam" &
CUDA_VISIBLE_DEVICES=4,5 python train.py --config $CONFIG_FILE --sks_name "phuc-map" &
CUDA_VISIBLE_DEVICES=6,7 python train.py --config $CONFIG_FILE --sks_name "denisdang" 
wait

CUDA_VISIBLE_DEVICES=0,1 python train.py --config $CONFIG_FILE --sks_name "bo" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config $CONFIG_FILE --sks_name "mam"
CUDA_VISIBLE_DEVICES=4,5 python train.py --config $CONFIG_FILE --sks_name "thuytien" &
CUDA_VISIBLE_DEVICES=6,7 python train.py --config $CONFIG_FILE --sks_name "viruss" 
wait
