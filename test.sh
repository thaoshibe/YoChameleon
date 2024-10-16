CONFIGS=("config/id16.yaml" "config/id64.yaml" "config/id6464.yaml" "config/id128.yaml")

for CONFIG in "${CONFIGS[@]}"
do
  CUDA_VISIBLE_DEVICES=0 python test.py --config $CONFIG --iteration 300 &
  CUDA_VISIBLE_DEVICES=1 python test.py --config $CONFIG --iteration 500 &
  CUDA_VISIBLE_DEVICES=2 python test.py --config $CONFIG --iteration 800 &
  CUDA_VISIBLE_DEVICES=3 python test.py --config $CONFIG --iteration 1000 &

  CUDA_VISIBLE_DEVICES=4 python test.py --config $CONFIG --iteration 1050 --finetune &
  CUDA_VISIBLE_DEVICES=5 python test.py --config $CONFIG --iteration 1100 --finetune &
  CUDA_VISIBLE_DEVICES=6 python test.py --config $CONFIG --iteration 1150 --finetune &
  CUDA_VISIBLE_DEVICES=7 python test.py --config $CONFIG --iteration 1200 --finetune &

  wait
done
