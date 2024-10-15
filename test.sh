CUDA_VISIBLE_DEVICES=0 python test.py --config config/16-4.yaml --iteration 300 &
CUDA_VISIBLE_DEVICES=1 python test.py --config config/64-8.yaml --iteration 300 &
CUDA_VISIBLE_DEVICES=2 python test.py --config config/128-16.yaml --iteration 300 &
CUDA_VISIBLE_DEVICES=3 python test.py --config config/64-64.yaml --iteration 300 &

CUDA_VISIBLE_DEVICES=4 python test.py --config config/16-4.yaml --iteration 600 --finetune &
CUDA_VISIBLE_DEVICES=5 python test.py --config config/64-8.yaml --iteration 600 --finetune &
CUDA_VISIBLE_DEVICES=6 python test.py --config config/64-64.yaml --iteration 600 --finetune &
CUDA_VISIBLE_DEVICES=7 python test.py --config config/128-16.yaml --iteration 600 --finetune &
# CUDA_VISIBLE_DEVICES=3 python test.py --config config/64-64.yaml --iteration 500 &
# CUDA_VISIBLE_DEVICES=4 python test.py --config config/16-4.yaml --iteration 550 &
wait