# CUDA_VISIBLE_DEVICES=2,3 python train.py --sks_name mam --batch_size 4 --data_root ./yollava-data/train/ --exp_name bz4-caption-v1 --model_id leloy/Anole-7b-v0.1-hf --image_gen_only
# CUDA_VISIBLE_DEVICES=0,1 python train.py --sks_name bo --batch_size 4 --data_root ./yollava-data/train/ --exp_name bz4-caption-v1 --model_id leloy/Anole-7b-v0.1-hf --image_gen_only
# CUDA_VISIBLE_DEVICES=4,5 python train.py --sks_name bo --batch_size 4 --data_root ./yollava-data/train/ --exp_name bz4-caption-v2 --model_id leloy/Anole-7b-v0.1-hf --image_gen_only
# CUDA_VISIBLE_DEVICES=6,7 python train.py --sks_name mam --batch_size 4 --data_root ./yollava-data/train/ --exp_name bz4-caption-v2 --model_id leloy/Anole-7b-v0.1-hf --image_gen_only

CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/mam-v0.yaml
CUDA_VISIBLE_DEVICES=0,1 python train.py --config ./config/bo-v0.yaml

# CUDA_VISIBLE_DEVICES=2,3 python train.py --config ./config/mam-v2.yaml
# CUDA_VISIBLE_DEVICES=4,5 python train.py --config ./config/bo-v2.yaml