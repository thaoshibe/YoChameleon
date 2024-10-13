mkdir /mnt/localssd/code
cd /mnt/localssd/code
cp -r /sensei-fs/users/thaon/code/YoChameleon /mnt/localssd/code

cd /mnt/localssd/code/YoChameleon
bash install.sh

mkdir /mnt/localssd/code/data
cd /mnt/localssd/code/data
cp /sensei-fs/users/thaon/data/yollava-data.zip /mnt/localssd/code/data
unzip /mnt/localssd/code/data/yollava-data.zip

cd /mnt/localssd/code/YoChameleon
export WANDB_API_KEY="563710e55fec9aac8f27c7ab80cfed931a2096f5"

# Train script

CUDA_VISIBLE_DEVICES=0,1,2 python train.py --config ./config/16-4.yaml &
CUDA_VISIBLE_DEVICES=3,4,5 python train.py --config ./config/64-8.yaml &
CUDA_VISIBLE_DEVICES=6,7 python train.py --config ./config/128-16.yaml
wait