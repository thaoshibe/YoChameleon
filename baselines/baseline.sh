# CUDA_VISIBLE_DEVICES=0 python anole.py --start 0 --end 10 &
# CUDA_VISIBLE_DEVICES=1 python anole.py --start 10 --end 20 &
# CUDA_VISIBLE_DEVICES=2 python anole.py --start 20 --end 30 &
# CUDA_VISIBLE_DEVICES=3 python anole.py --start 30 --end 40 &
# CUDA_VISIBLE_DEVICES=4 python anole.py --start 40 --end 50 &
# CUDA_VISIBLE_DEVICES=5 python anole.py --start 50 --end 60 &
# CUDA_VISIBLE_DEVICES=6 python anole.py --start 60 --end 70 &
# CUDA_VISIBLE_DEVICES=7 python anole.py --start 70 --end 80
# CUDA_VISIBLE_DEVICES=2,3 python anole.py --start 80 --end 90 &
# CUDA_VISIBLE_DEVICES=3,4 python anole.py --start 90 --end 100


CUDA_VISIBLE_DEVICES=0,1 python anole.py --start 0 --end 25 --image_prompt True &
CUDA_VISIBLE_DEVICES=2,3 python anole.py --start 25 --end 50 --image_prompt True &
CUDA_VISIBLE_DEVICES=4,5 python anole.py --start 50 --end 75 --image_prompt True &
CUDA_VISIBLE_DEVICES=6,7 python anole.py --start 75 --end 100 --image_prompt True
