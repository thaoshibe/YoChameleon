# # CONFIGS=("config/id16.yaml" "config/id64.yaml" "config/id6464.yaml" "config/id128.yaml")
# # cd sc

# CONFIGS=("./config/1000negative.yaml" "./config/1000neg-text.yaml" "./config/1000neg-text-recog.yaml" "./config/universal_prefix_positive.yaml")

# for CONFIG in "${CONFIGS[@]}"
# do
#   CUDA_VISIBLE_DEVICES=0 python test.py --sks_name "thao" --config $CONFIG --iteration 5 &
#   CUDA_VISIBLE_DEVICES=1 python test.py --sks_name "bo" --config $CONFIG --iteration 5 &
#   CUDA_VISIBLE_DEVICES=2 python test.py --sks_name "mam" --config $CONFIG --iteration 5 &
#   CUDA_VISIBLE_DEVICES=3 python test.py --sks_name "yuheng" --config $CONFIG --iteration 5 &
#   CUDA_VISIBLE_DEVICES=4 python test.py --sks_name "ciin" --config $CONFIG --iteration 5 &
#   CUDA_VISIBLE_DEVICES=5 python test.py --sks_name "willinvietnam" --config $CONFIG --iteration 5 &
#   CUDA_VISIBLE_DEVICES=6 python test.py --sks_name "oong" --config $CONFIG --iteration 5 &
#   CUDA_VISIBLE_DEVICES=7 python test.py --sks_name "khanhvy" --config $CONFIG --iteration 5 &
#   wait

#   CUDA_VISIBLE_DEVICES=0 python test.py --sks_name "thao" --config $CONFIG --iteration 10 &
#   CUDA_VISIBLE_DEVICES=1 python test.py --sks_name "bo" --config $CONFIG --iteration 10 &
#   CUDA_VISIBLE_DEVICES=2 python test.py --sks_name "mam" --config $CONFIG --iteration 10 &
#   CUDA_VISIBLE_DEVICES=3 python test.py --sks_name "yuheng" --config $CONFIG --iteration 10 &
#   CUDA_VISIBLE_DEVICES=4 python test.py --sks_name "ciin" --config $CONFIG --iteration 10 &
#   CUDA_VISIBLE_DEVICES=5 python test.py --sks_name "willinvietnam" --config $CONFIG --iteration 10 &
#   CUDA_VISIBLE_DEVICES=6 python test.py --sks_name "oong" --config $CONFIG --iteration 10 &
#   CUDA_VISIBLE_DEVICES=7 python test.py --sks_name "khanhvy" --config $CONFIG --iteration 10 &
#   wait

#   CUDA_VISIBLE_DEVICES=0 python test.py --sks_name "thao" --config $CONFIG --iteration 15 &
#   CUDA_VISIBLE_DEVICES=1 python test.py --sks_name "bo" --config $CONFIG --iteration 15 &
#   CUDA_VISIBLE_DEVICES=2 python test.py --sks_name "mam" --config $CONFIG --iteration 15 &
#   CUDA_VISIBLE_DEVICES=3 python test.py --sks_name "yuheng" --config $CONFIG --iteration 15 &
#   CUDA_VISIBLE_DEVICES=4 python test.py --sks_name "ciin" --config $CONFIG --iteration 15 &
#   CUDA_VISIBLE_DEVICES=5 python test.py --sks_name "willinvietnam" --config $CONFIG --iteration 15 &
#   CUDA_VISIBLE_DEVICES=6 python test.py --sks_name "oong" --config $CONFIG --iteration 15 &
#   CUDA_VISIBLE_DEVICES=7 python test.py --sks_name "khanhvy" --config $CONFIG --iteration 15 &
#   wait

#   CUDA_VISIBLE_DEVICES=0 python test.py --sks_name "thao" --config $CONFIG --iteration 15 --finetune &
#   CUDA_VISIBLE_DEVICES=1 python test.py --sks_name "bo" --config $CONFIG --iteration 15 --finetune &
#   CUDA_VISIBLE_DEVICES=2 python test.py --sks_name "mam" --config $CONFIG --iteration 15 --finetune &
#   CUDA_VISIBLE_DEVICES=3 python test.py --sks_name "yuheng" --config $CONFIG --iteration 15 --finetune &
#   CUDA_VISIBLE_DEVICES=4 python test.py --sks_name "ciin" --config $CONFIG --iteration 15 --finetune &
#   CUDA_VISIBLE_DEVICES=5 python test.py --sks_name "willinvietnam" --config $CONFIG --iteration 15 --finetune &
#   CUDA_VISIBLE_DEVICES=6 python test.py --sks_name "oong" --config $CONFIG --iteration 15 --finetune &
#   CUDA_VISIBLE_DEVICES=7 python test.py --sks_name "khanhvy" --config $CONFIG --iteration 15 --finetune &
#   wait

#   CUDA_VISIBLE_DEVICES=0 python test.py --sks_name "thao" --config $CONFIG --iteration 17 --finetune &
#   CUDA_VISIBLE_DEVICES=1 python test.py --sks_name "bo" --config $CONFIG --iteration 17 --finetune &
#   CUDA_VISIBLE_DEVICES=2 python test.py --sks_name "mam" --config $CONFIG --iteration 17 --finetune &
#   CUDA_VISIBLE_DEVICES=3 python test.py --sks_name "yuheng" --config $CONFIG --iteration 17 --finetune &
#   CUDA_VISIBLE_DEVICES=4 python test.py --sks_name "ciin" --config $CONFIG --iteration 17 --finetune &
#   CUDA_VISIBLE_DEVICES=5 python test.py --sks_name "willinvietnam" --config $CONFIG --iteration 17 --finetune &
#   CUDA_VISIBLE_DEVICES=6 python test.py --sks_name "oong" --config $CONFIG --iteration 17 --finetune &
#   CUDA_VISIBLE_DEVICES=7 python test.py --sks_name "khanhvy" --config $CONFIG --iteration 17 --finetune &
#   wait

#   CUDA_VISIBLE_DEVICES=0 python test.py --sks_name "thao" --config $CONFIG --iteration 20 --finetune &
#   CUDA_VISIBLE_DEVICES=1 python test.py --sks_name "bo" --config $CONFIG --iteration 20 --finetune &
#   CUDA_VISIBLE_DEVICES=2 python test.py --sks_name "mam" --config $CONFIG --iteration 20 --finetune &
#   CUDA_VISIBLE_DEVICES=3 python test.py --sks_name "yuheng" --config $CONFIG --iteration 20 --finetune &
#   CUDA_VISIBLE_DEVICES=4 python test.py --sks_name "ciin" --config $CONFIG --iteration 20 --finetune &
#   CUDA_VISIBLE_DEVICES=5 python test.py --sks_name "willinvietnam" --config $CONFIG --iteration 20 --finetune &
#   CUDA_VISIBLE_DEVICES=6 python test.py --sks_name "oong" --config $CONFIG --iteration 20 --finetune &
#   CUDA_VISIBLE_DEVICES=7 python test.py --sks_name "khanhvy" --config $CONFIG --iteration 20 --finetune &
#   wait
# done

# CONFIGS=("./config/1000negative.yaml" "./config/1000neg-text.yaml" "./config/1000neg-text-recog.yaml" "./config/universal_prefix_positive.yaml")
CONFIGS=("./config/universal_yollava.yaml" "./config/universal_wholemodel.yaml")
SKS_NAMES=("thao" "bo" "mam" "yuheng" "ciin" "willinvietnam" "oong" "khanhvy")
ITERATIONS=(5 10 15)
FINETUNE_ITERATIONS=(15 17 20)

run_tests() {
  local config=$1
  local iteration=$2
  local finetune=$3

  for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python test.py --sks_name "${SKS_NAMES[$i]}" --config $config --iteration $iteration $finetune &
  done
  wait
}

for CONFIG in "${CONFIGS[@]}"; do
  for ITER in "${ITERATIONS[@]}"; do
    run_tests $CONFIG $ITER ""
  done

  for FINETUNE_ITER in "${FINETUNE_ITERATIONS[@]}"; do
    run_tests $CONFIG $FINETUNE_ITER "--finetune"
  done
done
