export WANDB_API_KEY="YOUR_WANDB_API_KEY"

CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/universal_yollava.yaml --sks_name "thao" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/universal_yollava.yaml --sks_name "yuheng" &
CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/universal_yollava.yaml --sks_name "thuytien" &
CUDA_VISIBLE_DEVICES=6,7 python train.py --config ./config/universal_yollava.yaml --sks_name "viruss" 
wait

CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/universal_yollava.yaml --sks_name "ciin" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/universal_yollava.yaml --sks_name "khanhvy" &
CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/universal_yollava.yaml --sks_name "oong" &
CUDA_VISIBLE_DEVICES=6,7 python train.py --config ./config/universal_yollava.yaml --sks_name "willinvietnam" 
wait

CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/universal_yollava.yaml --sks_name "denisdang" &
CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/universal_yollava.yaml --sks_name "phuc-map"
CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/universal_yollava.yaml --sks_name "bo" &
CUDA_VISIBLE_DEVICES=6,7 python train.py --config ./config/universal_yollava.yaml --sks_name "mam" 
wait