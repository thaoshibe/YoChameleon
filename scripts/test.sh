# CONFIGS=("config/id16.yaml" "config/id64.yaml" "config/id6464.yaml" "config/id128.yaml")
cd ..

CONFIGS=("config/500.yaml")

for CONFIG in "${CONFIGS[@]}"
do
  CUDA_VISIBLE_DEVICES=0 python test.py --config $CONFIG --iteration 200 &
  CUDA_VISIBLE_DEVICES=1 python test.py --config $CONFIG --iteration 300 &
  CUDA_VISIBLE_DEVICES=2 python test.py --config $CONFIG --iteration 400 &
  CUDA_VISIBLE_DEVICES=3 python test.py --config $CONFIG --iteration 500 &

  CUDA_VISIBLE_DEVICES=4 python test.py --config $CONFIG --iteration 550 --finetune &
  CUDA_VISIBLE_DEVICES=5 python test.py --config $CONFIG --iteration 600 --finetune &
  CUDA_VISIBLE_DEVICES=6 python test.py --config $CONFIG --iteration 650 --finetune &
  CUDA_VISIBLE_DEVICES=7 python test.py --config $CONFIG --iteration 700 --finetune &

  wait
done

CONFIGS=("config/1000.yaml")

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

CONFIGS=("config/2000.yaml")

for CONFIG in "${CONFIGS[@]}"
do
  CUDA_VISIBLE_DEVICES=0 python test.py --config $CONFIG --iteration 300 &
  CUDA_VISIBLE_DEVICES=1 python test.py --config $CONFIG --iteration 500 &
  CUDA_VISIBLE_DEVICES=2 python test.py --config $CONFIG --iteration 800 &
  CUDA_VISIBLE_DEVICES=3 python test.py --config $CONFIG --iteration 1500 &

  CUDA_VISIBLE_DEVICES=4 python test.py --config $CONFIG --iteration 1550 --finetune &
  CUDA_VISIBLE_DEVICES=5 python test.py --config $CONFIG --iteration 1600 --finetune &
  CUDA_VISIBLE_DEVICES=6 python test.py --config $CONFIG --iteration 1650 --finetune &
  CUDA_VISIBLE_DEVICES=7 python test.py --config $CONFIG --iteration 1700 --finetune &

  wait
done

CONFIGS=("config/5000.yaml")
for CONFIG in "${CONFIGS[@]}"
do
  CUDA_VISIBLE_DEVICES=0 python test.py --config $CONFIG --iteration 1000 &
  CUDA_VISIBLE_DEVICES=1 python test.py --config $CONFIG --iteration 2000 &
  CUDA_VISIBLE_DEVICES=2 python test.py --config $CONFIG --iteration 3000 &
  CUDA_VISIBLE_DEVICES=3 python test.py --config $CONFIG --iteration 4000 &

  CUDA_VISIBLE_DEVICES=4 python test.py --config $CONFIG --iteration 4050 --finetune &
  CUDA_VISIBLE_DEVICES=5 python test.py --config $CONFIG --iteration 4100 --finetune &
  CUDA_VISIBLE_DEVICES=6 python test.py --config $CONFIG --iteration 4150 --finetune &
  CUDA_VISIBLE_DEVICES=7 python test.py --config $CONFIG --iteration 4200 --finetune &

  wait
done