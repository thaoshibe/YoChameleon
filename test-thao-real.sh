aNAME="thao"

CUDA_VISIBLE_DEVICES=4 python test.py --sks_name $NAME --exp_name ${NAME}-real --epoch 20 --prompt "A photo of <reserved16300>" --token_len 3 &
CUDA_VISIBLE_DEVICES=4 python test.py --sks_name $NAME --exp_name ${NAME}-real --epoch 20 --prompt "A photo of <reserved16300>" --token_len 5 &
CUDA_VISIBLE_DEVICES=5 python test.py --sks_name $NAME --exp_name ${NAME}-real --epoch 20 --prompt "A photo of <reserved16300>" --token_len 7 &
CUDA_VISIBLE_DEVICES=5 python test.py --sks_name $NAME --exp_name ${NAME}-real --epoch 20 --prompt "A photo of <reserved16300>" --token_len 9 &
CUDA_VISIBLE_DEVICES=6 python test.py --sks_name $NAME --exp_name ${NAME}-real --epoch 20 --prompt "A photo of <reserved16300>" --token_len 11 &
CUDA_VISIBLE_DEVICES=6 python test.py --sks_name $NAME --exp_name ${NAME}-real --epoch 20 --prompt "A photo of <reserved16300>" --token_len 13 &
CUDA_VISIBLE_DEVICES=7 python test.py --sks_name $NAME --exp_name ${NAME}-real --epoch 20 --prompt "A photo of <reserved16300>" --token_len 15 &
CUDA_VISIBLE_DEVICES=7 python test.py --sks_name $NAME --exp_name ${NAME}-real --epoch 20 --prompt "A photo of <reserved16300>" --token_len 17 &

# Wait for all background processes to finish
wait