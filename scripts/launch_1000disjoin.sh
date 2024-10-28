USER="$(whoami)"
echo "USER"
echo $USER

WORKING_FOLDER="/mnt/localssd/code"
DATA_ZIP_FILE="/sensei-fs/users/thaon/data/yochameleon-data.zip"
CODE_FOLDER="/sensei-fs/users/thaon/code/YoChameleon"

if [ ! -d "/home/user/" ]; then
  sudo mkdir /home/user/
fi

if [ ! -d "/home/$USER" ]; then
  sudo mkdir /home/$USER
fi

sudo chmod 777 -R /home/

echo "Launching training script"
mkdir -p $WORKING_FOLDER
cd $WORKING_FOLDER
cp -r $CODE_FOLDER $WORKING_FOLDER

cd $WORKING_FOLDER/YoChameleon
bash scripts/install.sh

mkdir -p $WORKING_FOLDER/data
cd $WORKING_FOLDER/data
cp -r $DATA_ZIP_FILE $WORKING_FOLDER/data
unzip $WORKING_FOLDER/data/yochameleon-data.zip

# cd $WORKING_FOLDER/YoChameleon
export WANDB_API_KEY="563710e55fec9aac8f27c7ab80cfed931a2096f5"

# Train script
cd $WORKING_FOLDER/YoChameleon

CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/1000disjoin.yaml --sks_name "thao" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/1000disjoin.yaml --sks_name "yuheng" &
CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/1000disjoin.yaml --sks_name "ciin" &
CUDA_VISIBLE_DEVICES=6,7 python train.py --config ./config/1000disjoin.yaml --sks_name "khanhvy" 
wait

CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/1000disjoin.yaml --sks_name "oong" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/1000disjoin.yaml --sks_name "willinvietnam" &
CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/1000disjoin.yaml --sks_name "phuc-map" &
CUDA_VISIBLE_DEVICES=6,7 python train.py --config ./config/1000disjoin.yaml --sks_name "denisdang" 
wait

CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/1000disjoin.yaml --sks_name "bo" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/1000disjoin.yaml --sks_name "mam"
CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/1000disjoin.yaml --sks_name "thuytien" &
CUDA_VISIBLE_DEVICES=6,7 python train.py --config ./config/1000disjoin.yaml --sks_name "viruss" 
wait
