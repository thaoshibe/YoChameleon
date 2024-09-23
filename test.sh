# CUDA_VISIBLE_DEVICES=0,1 python test_yochameleon.py --sks_name bo --epoch 14 --model_id leloy/Anole-7b-v0.1-hf --exp_name bz4-caption-v2 &
# CUDA_VISIBLE_DEVICES=2,3 python test_yochameleon.py --sks_name mam --epoch 14 --model_id leloy/Anole-7b-v0.1-hf --exp_name bz4-caption-v2 &
# CUDA_VISIBLE_DEVICES=4,5 python test_yochameleon.py --sks_name bo --epoch 14 --model_id leloy/Anole-7b-v0.1-hf --exp_name bz4-caption-v1 &
# CUDA_VISIBLE_DEVICES=6,7 python test_yochameleon.py --sks_name mam --epoch 14 --model_id leloy/Anole-7b-v0.1-hf --exp_name bz4-caption-v1
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

# CUDA_VISIBLE_DEVICES=3,4 python test.py --sks_name bo --exp_name bo-inpaint-sdxl --epoch 5 & \
# CUDA_VISIBLE_DEVICES=3,4 python test.py --sks_name bo --exp_name v5-qa --epoch 10 & \
# CUDA_VISIBLE_DEVICES=3,4 python test.py --sks_name bo --exp_name v5-qa --epoch 20
# python test.py --sks_name bo --exp_name bo-v4 --epoch 20 --prompt "A photo of <reserved16300> is <reserved16301><reserved16302><reserved16303><reserved16304><reserved16305><reserved16306><reserved16307><reserved16308><reserved16309><reserved16310><reserved16311><reserved16312><reserved16313><reserved16314><reserved16315><reserved16316>"
# python test.py --sks_name bo --exp_name bo-v2 --epoch 5 --prompt "A photo of <reserved16300>"
# python test.py --sks_name bo --exp_name bo-v3 --epoch 5 --prompt "A photo of <reserved16300>"
# python test.py --sks_name bo --exp_name bo-v4 --epoch 5 --prompt "A photo of <reserved16300>"

# python test.py --sks_name bo --exp_name yollava --epoch 5 --prompt "A photo of <reserved16300> in a sunflower field"
# python test.py --sks_name bo --exp_name bo-v2 --epoch 5 --prompt "A photo of <reserved16300> in a sunflower field"
# python test.py --sks_name bo --exp_name bo-v3 --epoch 5 --prompt "A photo of <reserved16300> in a sunflower field"
# python test.py --sks_name bo --exp_name bo-v4 --epoch 5 --prompt "A photo of <reserved16300> in a sunflower field"

# python test.py --sks_name bo --exp_name yollava --epoch 5 --prompt "A photo of <reserved16300> in a winter village"
# python test.py --sks_name bo --exp_name bo-v2 --epoch 5 --prompt "A photo of <reserved16300> in a winter village"
# python test.py --sks_name bo --exp_name bo-v3 --epoch 5 --prompt "A photo of <reserved16300> in a winter village"
# python test.py --sks_name bo --exp_name bo-v4 --epoch 5 --prompt "A photo of <reserved16300> in a winter village"

#!/bin/bash

# python test.py --sks_name thao --exp_name thao-v1 --epoch 30 --prompt "A photo of <reserved16300>" --token_len 2
# python test.py --sks_name thao --exp_name thao-v1 --epoch 20 --prompt "A photo of <reserved16300>" --token_len 3
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300>" --token_len 4
# python test.py --sks_name thao --exp_name thao-v1 --epoch 20 --prompt "A photo of <reserved16300>" --token_len 5
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300>" --token_len 6
# python test.py --sks_name thao --exp_name thao-v1 --epoch 20 --prompt "A photo of <reserved16300>" --token_len 7
# python test.py --sks_name thao --exp_name thao-v1 --epoch 30 --prompt "A photo of <reserved16300>" --token_len 8
# python test.py --sks_name thao --exp_name thao-v1 --epoch 20 --prompt "A photo of <reserved16300>" --token_len 9
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300>" --token_len 10
# python test.py --sks_name thao --exp_name thao-v1 --epoch 20 --prompt "A photo of <reserved16300>" --token_len 11
# python test.py --sks_name thao --exp_name thao-v1 --epoch 30 --prompt "A photo of <reserved16300>" --token_len 12
# python test.py --sks_name thao --exp_name thao-v1 --epoch 20 --prompt "A photo of <reserved16300>" --token_len 13
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300>" --token_len 14
# python test.py --sks_name thao --exp_name thao-v1 --epoch 20 --prompt "A photo of <reserved16300>" --token_len 15
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300>" --token_len 16
# python test.py --sks_name thao --exp_name thao-v1 --epoch 20 --prompt "A photo of <reserved16300>" --token_len 17

#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test.py --sks_name thao --exp_name thao-v1 --epoch 20 --prompt "A photo of <reserved16300>" --token_len 2 &
CUDA_VISIBLE_DEVICES=0 python test.py --sks_name thao --exp_name thao-v1 --epoch 20 --prompt "A photo of <reserved16300>" --token_len 3 &
CUDA_VISIBLE_DEVICES=0 python test.py --sks_name thao --exp_name thao-v1 --epoch 20 --prompt "A photo of <reserved16300>" --token_len 5 &
CUDA_VISIBLE_DEVICES=1 python test.py --sks_name thao --exp_name thao-v1 --epoch 20 --prompt "A photo of <reserved16300>" --token_len 7 &
CUDA_VISIBLE_DEVICES=1 python test.py --sks_name thao --exp_name thao-v1 --epoch 20 --prompt "A photo of <reserved16300>" --token_len 9 &
CUDA_VISIBLE_DEVICES=2 python test.py --sks_name thao --exp_name thao-v1 --epoch 20 --prompt "A photo of <reserved16300>" --token_len 11 &
CUDA_VISIBLE_DEVICES=2 python test.py --sks_name thao --exp_name thao-v1 --epoch 20 --prompt "A photo of <reserved16300>" --token_len 13 &
CUDA_VISIBLE_DEVICES=3 python test.py --sks_name thao --exp_name thao-v1 --epoch 20 --prompt "A photo of <reserved16300>" --token_len 15 &
CUDA_VISIBLE_DEVICES=3 python test.py --sks_name thao --exp_name thao-v1 --epoch 20 --prompt "A photo of <reserved16300>" --token_len 17 &

# Wait for all background processes to finish
wait


# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300> in a sunflower " --token_len 2
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300> in a sunflower " --token_len 3
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300> in a sunflower " --token_len 4
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300> in a sunflower " --token_len 5
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300> in a sunflower " --token_len 6
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300> in a sunflower " --token_len 7
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300> in a sunflower " --token_len 8
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300> in a sunflower " --token_len 9
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300> in a sunflower " --token_len 10
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300> in a sunflower " --token_len 11
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300> in a sunflower " --token_len 12
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300> in a sunflower " --token_len 13
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300> in a sunflower " --token_len 14
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300> in a sunflower " --token_len 15
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300> in a sunflower " --token_len 16
# python test.py --sks_name thao --exp_name thao-v1 --epoch 18 --prompt "A photo of <reserved16300> in a sunflower " --token_len 17


# python test.py --sks_name bo --exp_name bo-v4 --epoch 20 --prompt "A photo of <reserved16300> is <reserved16301><reserved16302><reserved16303><reserved16304><reserved16305><reserved16306><reserved16307><reserved16308><reserved16309><reserved16310><reserved16311><reserved16312><reserved16313><reserved16314>" &

# python test.py --sks_name bo --exp_name bo-v4 --epoch 20 --prompt "A photo of <reserved16300> is <reserved16301><reserved16302><reserved16303><reserved16304><reserved16305><reserved16306><reserved16307><reserved16308><reserved16309><reserved16310><reserved16311><reserved16312>" &

# python test.py --sks_name bo --exp_name bo-v4 --epoch 20 --prompt "A photo of <reserved16300> is <reserved16301><reserved16302><reserved16303><reserved16304><reserved16305><reserved16306><reserved16307><reserved16308><reserved16309><reserved16310><reserved16311>" &

# python test.py --sks_name bo --exp_name bo-v4 --epoch 20 --prompt "A photo of <reserved16300> is <reserved16301><reserved16302><reserved16303><reserved16304><reserved16305><reserved16306><reserved16307><reserved16308><reserved16309><reserved16310>" &

# python test.py --sks_name bo --exp_name bo-v4 --epoch 10 --prompt "A photo of <reserved16300> is <reserved16301><reserved16302><reserved16303><reserved16304><reserved16305><reserved16306><reserved16307><reserved16308>" &

# python test.py --sks_name bo --exp_name bo-v4 --epoch 20 --prompt "A photo of <reserved16300> is <reserved16301><reserved16302><reserved16303><reserved16304><reserved16305><reserved16306>" &

# python test.py --sks_name bo --exp_name bo-v4 --epoch 20 --prompt "A photo of <reserved16300> is <reserved16301><reserved16302><reserved16303><reserved16304>" &

# python test.py --sks_name bo --exp_name bo-v4 --epoch 20 --prompt "A photo of <reserved16300> is <reserved16301><reserved16302>" &

# python test.py --sks_name bo --exp_name bo-v4 --epoch 20 --prompt "A photo of <reserved16300> is" &

# Wait for all parallel processes to complete
# wait
