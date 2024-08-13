CUDA_VISIBLE_DEVICES=0,1 python test_yochameleon.py --sks_name bo --epoch 14 --model_id leloy/Anole-7b-v0.1-hf --exp_name bz4-caption-v2 &
CUDA_VISIBLE_DEVICES=2,3 python test_yochameleon.py --sks_name mam --epoch 14 --model_id leloy/Anole-7b-v0.1-hf --exp_name bz4-caption-v2 &
CUDA_VISIBLE_DEVICES=4,5 python test_yochameleon.py --sks_name bo --epoch 14 --model_id leloy/Anole-7b-v0.1-hf --exp_name bz4-caption-v1 &
CUDA_VISIBLE_DEVICES=6,7 python test_yochameleon.py --sks_name mam --epoch 14 --model_id leloy/Anole-7b-v0.1-hf --exp_name bz4-caption-v1
# python test_yochameleon.py --sks_name thao --epoch 18 --model_id leloy/Anole-7b-v0.1-hf --exp_name anole &
# python test_yochameleon.py --sks_name yuheng --epoch 18 --model_id leloy/Anole-7b-v0.1-hf --exp_name anole &
# python test_yochameleon.py --sks_name bo --epoch 18 --model_id leloy/Anole-7b-v0.1-hf --exp_name chameleon &
# python test_yochameleon.py --sks_name mam --epoch 18 --model_id leloy/Anole-7b-v0.1-hf --exp_name chameleon
# CUDA_VISIBLE_DEVICES=0,1 python test_yochameleon.py --sks_name bo --epoch 18 --model_id leloy/Anole-7b-v0.1-hf --exp_name bz4 \
# & CUDA_VISIBLE_DEVICES=2,3 python test_yochameleon.py --sks_name mam --epoch 18 --model_id leloy/Anole-7b-v0.1-hf --exp_name bz4 \

# wait
# python anole.py --sks_name mam --prompt 'Here is a photo of Mam. Can you generate another photo of him?<image>'
# python anole.py --sks_name bo --prompt 'Here is a photo of Bo. Can you generate another photo of him?<image>'
# python anole.py --sks_name thao --prompt 'Here is a photo of me. Can you generate another photo of me?<image>'
# python anole.py --sks_name yuheng --prompt 'Here is a photo of my friend. Can you generate another photo of him?<image>'