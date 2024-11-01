# Configuration and arrays
CONFIGS=("./config/1000neg-text.yaml" "./config/1000neg-text-recog.yaml")
SKS_NAMES=("thao" )
# ITERATIONS=(1000 1100 1200 1300 1400 1500 1600 1700)

# Loop through each sks_name and iteration
for SKS_NAME in "${SKS_NAMES[@]}"; do
  # for ITERATION in "${ITERATIONS[@]}"; do
  # Execute the Python script with the current sks_name and iteration
  CUDA_VISIBLE_DEVICES=0 python test.py --sks_name "$SKS_NAME" --config "${CONFIGS[0]}" --iteration 15 &
  CUDA_VISIBLE_DEVICES=1 python test.py --sks_name "$SKS_NAME" --config "${CONFIGS[0]}" --iteration 20 &
  CUDA_VISIBLE_DEVICES=2 python test.py --sks_name "$SKS_NAME" --config "${CONFIGS[0]}" --iteration 25 &
  CUDA_VISIBLE_DEVICES=3 python test.py --sks_name "$SKS_NAME" --config "${CONFIGS[0]}" --iteration 30 &
  CUDA_VISIBLE_DEVICES=4 python test.py --sks_name "$SKS_NAME" --config "${CONFIGS[1]}" --iteration 15 &
  CUDA_VISIBLE_DEVICES=5 python test.py --sks_name "$SKS_NAME" --config "${CONFIGS[1]}" --iteration 20 &
  CUDA_VISIBLE_DEVICES=6 python test.py --sks_name "$SKS_NAME" --config "${CONFIGS[1]}" --iteration 25 &
  CUDA_VISIBLE_DEVICES=7 python test.py --sks_name "$SKS_NAME" --config "${CONFIGS[1]}" --iteration 30 &
  wait
  # done
done

