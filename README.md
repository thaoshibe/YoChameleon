# Yo'Chameleon

<img src="./images/yochameleon.png" alt="YoChameleon" width="400">

### TODO
- [ ] Training
  - [x] Change to bfloat16?
  - [ ] Link the retrieval and training together? -- Low priority
  - [ ] Create augmented data for the model?
    - [ ] Train with augmented dataset
  - [x] Support: Setting E ("A photo of <sks>") -- Create json file
  - [x] Support: Two set of tokens?

- Dataset right now: Missing 100 random images!

- [ ] Evaluation -- High priority now
  - [ ] Recognition evaluation: Yes/No accuracy/ Recall?
  - [ ] Diversity image generation evaluation (CLIP)
  - [ ] Image quality generation: CLIP score/ DINO scores?
  
- Baselines:
  - Code baseline
- [ ] Minor
  - [x] Create a list of <reserved> tokens for the model?


### Getting Started

```
bash install.sh
```

### 🛠️ Creating dataset

```
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   bash scripts provided in `scripts/create_data` folder
#   
#   scripts/create_data
#   ├── retrieve.sh               # retrieve negative examples
#   ├── recognition.sh            # recognition data (100 hard neg, 100 easy neg, & positive)
#   ├── create_soft_positive.sh   # image generation data
#   ├── simple_conversation.sh    # simple conversation data
#   └── text_only_data.sh         # call GPT-4o for text-only response data
#
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#
# Remember to check and fill the relative path in the script before running
#

bash scripts/create_soft_positive.sh

```

<details>
<summary> Retrieve negative examples </summary>

```
cd create_training_data/conversation_data
NAMES=("bo" "duck-banana" "marie-cat" "pusheen-cup" "thuytien"
       "brown-duck" "dug" "mydieu" "shiba-black" "tokyo-keyboard"
       "butin" "elephant" "neurips-cup" "shiba-gray" "toodles-galore"
       "cat-cup" "fire" "nha-tho-hanoi" "shiba-sleep" "viruss"
       "chua-thien-mu" "henry" "nha-tho-hcm" "shiba-yellow" "water"
       "ciin" "khanhvy" "oong" "thao" "willinvietnam"
       "denisdang" "lamb" "phuc-map" "thap-but" "yellow-duck"
       "dragon" "mam" "pig-cup" "thap-cham" "yuheng")

for NAME in "${NAMES[@]}"; do
  INPUT_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  SAVE_FOLDER="${INPUT_FOLDER}/negative_example"
  LIMIT=5000 # Number of negative examples to retrieve
  echo "Processing folder: ${NAME}"
  
  python create_training_data/retrieve_negative/load_similar_example.py \
    --input_folder $INPUT_FOLDER \
    --save_folder $SAVE_FOLDER \
    --limit $LIMIT \
    --origin "l2"
done
```
</details>

<details>
<summary> Create recognition data </summary>

```
cd create_training_data/conversation_data
# List of names or folders to process
NAMES=("bo" "duck-banana" "marie-cat" "pusheen-cup" "thuytien"
       "brown-duck" "dug" "mydieu" "shiba-black" "tokyo-keyboard"
       "butin" "elephant" "neurips-cup" "shiba-gray" "toodles-galore"
       "cat-cup" "fire" "nha-tho-hanoi" "shiba-sleep" "viruss"
       "chua-thien-mu" "henry" "nha-tho-hcm" "shiba-yellow" "water"
       "ciin" "khanhvy" "oong" "thao" "willinvietnam"
       "denisdang" "lamb" "phuc-map" "thap-but" "yellow-duck"
       "dragon" "mam" "pig-cup" "thap-cham" "yuheng")

# Loop through each folder
for NAME in "${NAMES[@]}"; do
  # Define the positive image folder based on the name
  POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  
  # Define the negative image folder (assuming it's fixed or can vary similarly)
  NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
  
  # Define the output file path for the JSON result
  OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
  
  # Log which folder is being processed
  echo "Processing folder: ${NAME}"
  
  # Execute the Python script with the required arguments
  python create_conversation.py \
    --positive_image_folder "$POSITIVE_IMAGE_FOLDER" \
    --negative_image_folder "$NEGATIVE_IMAGE_FOLDER" \
    --output_file "$OUTPUT_FILE" \
    --limit_positive 5 \
    --limit_negative 100
done
```
</details>

<details>
<summary> Simple conversation data </summary> 

```
cd create_training_data/dense_caption

# List of names or folders to process -- For human
NAMES=("thuytien" "viruss" "ciin" "khanhvy" "oong" "thao" "willinvietnam" "denisdang" "phuc-map" "yuheng")

# Loop through each folder
for NAME in "${NAMES[@]}"; do
  # Define the positive image folder based on the name
  POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  
  # Define the negative image folder (assuming it's fixed or can vary similarly)
  NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
  
  # Define the output file path for the JSON result
  OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
  
  # Log which folder is being processed
  echo "Processing folder: ${NAME}"
  
  # Execute the Python script with the required arguments
  python gpt4o-api.py \
    --input_image_folder "$POSITIVE_IMAGE_FOLDER" \
    --prompt_file_path ./system-prompts/text-conversation.txt \
    --output_file "$OUTPUT_FILE" \
    --text_conversation \
    --human \
    --limit 5
done

# List of names or folders to process -- For object
NAMES=("bo" "duck-banana" "marie-cat" "pusheen-cup"
       "brown-duck" "dug" "mydieu" "shiba-black" "tokyo-keyboard"
       "butin" "elephant" "neurips-cup" "shiba-gray" "toodles-galore"
       "cat-cup" "fire" "nha-tho-hanoi" "shiba-sleep"
       "chua-thien-mu" "henry" "nha-tho-hcm" "shiba-yellow" "water"
       "lamb" "thap-but" "yellow-duck"
       "dragon" "mam" "pig-cup" "thap-cham")

# Loop through each folder
for NAME in "${NAMES[@]}"; do
  # Define the positive image folder based on the name
  POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  
  # Define the negative image folder (assuming it's fixed or can vary similarly)
  NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
  
  # Define the output file path for the JSON result
  OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
  
  # Log which folder is being processed
  echo "Processing folder: ${NAME}"
  
  # Execute the Python script with the required arguments
  python gpt4o-api.py \
    --input_image_folder "$POSITIVE_IMAGE_FOLDER" \
    --prompt_file_path ./system-prompts/text-conversation.txt \
    --output_file "$OUTPUT_FILE" \
    --text_conversation \
    --limit 5
done
```
</details>

<details>
<summary> Image generation data </summary>

```
cd create_training_data/retrieve_negative
# List of names or folders to process
NAMES=("bo" "duck-banana" "marie-cat" "pusheen-cup" "thuytien"
       "brown-duck" "dug" "mydieu" "shiba-black" "tokyo-keyboard"
       "butin" "elephant" "neurips-cup" "shiba-gray" "toodles-galore"
       "cat-cup" "fire" "nha-tho-hanoi" "shiba-sleep" "viruss"
       "chua-thien-mu" "henry" "nha-tho-hcm" "shiba-yellow" "water"
       "ciin" "khanhvy" "oong" "thao" "willinvietnam"
       "denisdang" "lamb" "phuc-map" "thap-but" "yellow-duck"
       "dragon" "mam" "pig-cup" "thap-cham" "yuheng")

# Loop through each folder
for NAME in "${NAMES[@]}"; do
  # Define the positive image folder based on the name
  POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  
  # Define the output file path for the JSON result
  OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
  
  # Log which folder is being processed
  echo "Processing folder: ${NAME}"
  
  # Execute the Python script with the required arguments
  python create_conversation_by_ranking.py \
    --input_folder "$POSITIVE_IMAGE_FOLDER" \
    --save_folder "$OUTPUT_FILE" \
    --version image_gen_positive_only \
    --num_of_real_images 100 \ch
    --token_length 16 \
    --spacing 16
done
```
</details>

<details>
<summary> Soft negative ideas data </summary>

```
cd create_training_data/retrieve_negative
# List of names or folders to process
# NAMES=("bo" "duck-banana" "marie-cat" "pusheen-cup" "thuytien"
#        "brown-duck" "dug" "mydieu" "shiba-black" "tokyo-keyboard"
#        "butin" "elephant" "neurips-cup" "shiba-gray" "toodles-galore"
#        "cat-cup" "fire" "nha-tho-hanoi" "shiba-sleep" "viruss"
#        "chua-thien-mu" "henry" "nha-tho-hcm" "shiba-yellow" "water"
#        "ciin" "khanhvy" "oong" "thao" "willinvietnam"
#        "denisdang" "lamb" "phuc-map" "thap-but" "yellow-duck"
#        "dragon" "mam" "pig-cup" "thap-cham" "yuheng")

NAMES=("bo" "mam" "thuytien" "viruss" "ciin" "khanhvy" "oong" "thao" "willinvietnam" "denisdang" "phuc-map" "yuheng")
# NAMES=("bo")
# Loop through each folder
for NAME in "${NAMES[@]}"; do
  # Define the positive image folder based on the name
  POSITIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}"
  NEGATIVE_IMAGE_FOLDER="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/negative_example"
  # Define the output file path for the JSON result
  OUTPUT_FILE="/mnt/localssd/code/data/yochameleon-data/train/${NAME}/json"
  
  # Log which folder is being processed
  echo "Processing folder: ${NAME}"
  
  # Execute the Python script with the required arguments
  python create_conversation_by_ranking.py \
    --input_folder "$POSITIVE_IMAGE_FOLDER" \
    --save_folder "$OUTPUT_FILE" \
    --version '2000' \
    --num_of_real_images -100 \
    --token_length 16 \
    --spacing 1 \
    --negative_image True \
    --limit_negative 2000
done
```
</details>

### 🧑‍🏫 Train

```
python train.py --config config/basic.yaml
```

### 🧪 Test

```
#--- This test will generated "A photo of <sks>" and saved to some directory

python test.py --config config/basic.yaml 

# Flexible test
```


### 📊 Evaluation

Please reference the README.md in the `evaluation` folder for more details.

### 🤗 Acknowledgements

This project will not be possible without the following open-source projects:
- [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://github.com/facebookresearch/chameleon)
- [Anole: An Open, Autoregressive and Native Multimodal Models for Interleaved Image-Text Generation](https://gair-nlp.github.io/anole/)
- and amazing HuggingFace's community: [Chamleon on HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/chameleon), [Anole on HuggingFace](https://github.com/huggingface/transformers/pull/32013)

Thank you!

