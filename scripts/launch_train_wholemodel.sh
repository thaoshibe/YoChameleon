# USER="$(whoami)"
# echo "USER"
# echo $USER

# if [ ! -d "/home/user/" ]; then
#   sudo mkdir /home/user/
# fi

# if [ ! -d "/home/$USER" ]; then
#   sudo mkdir /home/$USER
# fi

# sudo chmod 777 -R /home/

# echo "Launching training script"
# mkdir /mnt/localssd/code
# cd /mnt/localssd/code
# cp -r /sensei-fs/users/thaon/code/YoChameleon /mnt/localssd/code

# cd /mnt/localssd/code/YoChameleon
# bash scripts/install.sh

# mkdir /mnt/localssd/code/data
# cd /mnt/localssd/code/data
# cp -r /sensei-fs/users/thaon/data/yochameleon-data.zip /mnt/localssd/code/data
# unzip /mnt/localssd/code/data/yochameleon-data.zip

# # cd /mnt/localssd/code/YoChameleon
# export WANDB_API_KEY="563710e55fec9aac8f27c7ab80cfed931a2096f5"

# # Train script
# cd /mnt/localssd/code/YoChameleon
# ("thao" "yuheng" "thuytien" "viruss" "ciin" "khanhvy" "oong" "willinvietnam" "denisdang" "phuc-map")

# CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/universal_wholemodel.yaml --sks_name "thao" &
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/universal_wholemodel.yaml --sks_name "yuheng" &
# CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/universal_wholemodel.yaml --sks_name "thuytien" &
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config ./config/universal_wholemodel.yaml --sks_name "viruss" 
# wait

# CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/universal_wholemodel.yaml --sks_name "ciin" &
# CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/universal_wholemodel.yaml --sks_name "khanhvy" &
# CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/universal_wholemodel.yaml --sks_name "oong" &
# CUDA_VISIBLE_DEVICES=6,7 python train.py --config ./config/universal_wholemodel.yaml --sks_name "willinvietnam" 
# wait

CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/universal_wholemodel.yaml --sks_name "denisdang" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/universal_wholemodel.yaml --sks_name "phuc-map"
CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/universal_wholemodel.yaml --sks_name "oong" &
CUDA_VISIBLE_DEVICES=6,7 python train.py --config ./config/universal_wholemodel.yaml --sks_name "willinvietnam" 
wait