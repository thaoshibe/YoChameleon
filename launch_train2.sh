USER="$(whoami)"
echo "USER"
echo $USER

if [ ! -d "/home/user/" ]; then
  sudo mkdir /home/user/
fi

if [ ! -d "/home/$USER" ]; then
  sudo mkdir /home/$USER
fi

sudo chmod 777 -R /home/

echo "Launching training script"
mkdir /mnt/localssd/code
cd /mnt/localssd/code
cp -r /sensei-fs/users/thaon/code/YoChameleon /mnt/localssd/code

cd /mnt/localssd/code/YoChameleon
bash install.sh

mkdir /mnt/localssd/code/data
cd /mnt/localssd/code/data
cp -r /sensei-fs/users/thaon/data/yollava-data.zip /mnt/localssd/code/data
unzip /mnt/localssd/code/data/yollava-data.zip

# cd /mnt/localssd/code/YoChameleon
export WANDB_API_KEY="563710e55fec9aac8f27c7ab80cfed931a2096f5"

# Train script
cd /mnt/localssd/code/YoChameleon
CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/500.yaml &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/1000.yaml &
CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/2000.yaml &
CUDA_VISIBLE_DEVICES=6,7 python train.py --config ./config/5000.yaml
wait