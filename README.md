# Yo'Chameleon

<img src="./images/yochameleon.png" alt="YoChameleon" width="300">

⭑.ᐟ *Hello, this is [Yo'LLaVA](https://thaoshibe.github.io/YoLLaVA/) meets [Chameleon](https://arxiv.org/abs/2405.09818)!* ⭑.ᐟ
ㅤ
---

##### Table of Contents

1. [Getting Started](#-getting-started)
1. [Creating Dataset](#-creating-dataset)
1. [Train](#-train)
1. [Test](#-test)
1. [Evaluation](#-evaluation)
  - [Detailed Caption](#detailed-caption)
  - [Facial Similarity Scores](#facial-similarity-scores)
  - [CLIP Image-to-Image Similarity](#clip-image-to-image-similarity)
  - [Recognition Accuracy](#recognition-accuracy)
1. [Acknowledgements](#-acknowledgements)

### 🚀 Getting Started

```
# Clone the repository
git clone https://github.com/thaoshibe/YoChameleon.git
cd YoChameleon

# Install via pip
pip install -r requirements.txt

# Or run the bash script
bash install.sh
```

### Quick Start

This provide a quick start to train the model with the provided dataset.

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
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   ATTENTION: PLEASE CHECK/EDIT THE CONFIG FILE BEFORE RUNNING (IF NEEDED)
#
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

python train.py --config config/basic.yaml
```

If do you NOT want to use the `wandb` for logging (e.g., for debugging), you can turn off by

```
python train.py --config config/basic.yaml --no_wandb
```

Or multiple concept training bash script are given in `scripts` folder

```
bash train.sh
```

### 🧪 Test

```
# This test will generated "A photo of <sks>" and saved to some directory
python test.py --config config/basic.yaml 
# A bash script is also provided
bash scripts/test.sh

# Or you can test in a flexible mode
#
#       THAO: THIS IS TODO
#
python test_flexible.py --config config/basic.yaml --prompt "A photo of a cat"... 
```


### 📊 Evaluation

#### Detailed Captions

Detailed captions for each subject in [Yo'LLaVA datasets](https://github.com/WisconsinAIVision/YoLLaVA) are given in [baselines/subject-detailed-captions.json](./baselines/subject-detailed-captions.json).

For example, the detailed caption for `bo` is given as follows:

```
"bo": "<sks> is a charming cinnamon-colored Shiba Inu with cream accents and a cheerful personality, appears in various indoor and outdoor settings—posing on rugs, floors, and sidewalks. Often seen with a playful expression or tongue out, this Shiba enjoys relaxing, smiling for the camera, and is sometimes accompanied by a plush toy or sitting attentively in anticipation of a walk."
```
<img src="https://thaoshibe.github.io/visii/images/1_0.png" alt="Bo" width="300">

#### Facial similarity scores

Edit the file `bash scripts/eval/eval_facial_sim.sh`.

<details>
<summary> facial similarity compute between fake/ real folders</summary>

```
#!/bin/bash
cd ../evaluation/

EXP_FOLDER="64-5000"
FAKE_FOLDER_BASE="/sensei-fs/users/thaon/code/generated_images"

# Define the real folder
REAL_FOLDER="/mnt/localssd/code/data/yollava-data/train/thao"

# Define an array of fake folders
FAKE_FOLDERS=(
    # Local and aligned folders
    # "/mnt/localssd/code/data/dathao_algined"
    # "/mnt/localssd/code/data/yollava-data/train/khanhvy"
    # "/mnt/localssd/code/data/yollava-data/train/thao/negative_example"
    # Generated image sets using FAKE_FOLDER_BASE
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/1000"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/2000"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/3000"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/4000"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/4050"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/4100"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/4150"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/4200"
)

# Loop through each fake folder and run the Python evaluation script
for FAKE_FOLDER in "${FAKE_FOLDERS[@]}"
do
    echo "Running evaluation with fake folder: $FAKE_FOLDER"
    python insightface_verify.py --real_folder "$REAL_FOLDER" --fake_folder "$FAKE_FOLDER"
done

echo "All evaluations completed!"

```
</details>

#### CLIP Image-to-Image Similarity

```
python clip_image_similarity.py --real_folder --fake_folder
```

<details>
<summary> clip similarity score between fake/ real folders</summary>

```
#!/bin/bash
cd ../evaluation/

EXP_FOLDER="64-5000"
FAKE_FOLDER_BASE="/sensei-fs/users/thaon/code/generated_images"

# Define the real folder
REAL_FOLDER="/mnt/localssd/code/data/yollava-data/train/thao"

# Define an array of fake folders
FAKE_FOLDERS=(
    # Local and aligned folders
    # "/mnt/localssd/code/data/dathao_algined"
    # "/mnt/localssd/code/data/yollava-data/train/khanhvy"
    # "/mnt/localssd/code/data/yollava-data/train/thao/negative_example"
    # Generated image sets using FAKE_FOLDER_BASE
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/1000"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/2000"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/3000"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/4000"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/4050"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/4100"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/4150"
    "${FAKE_FOLDER_BASE}/${EXP_FOLDER}/4200"
)

# Loop through each fake folder and run the Python evaluation script
for FAKE_FOLDER in "${FAKE_FOLDERS[@]}"
do
    echo "Running evaluation with fake folder: $FAKE_FOLDER"
    python clip_image_similarity.py --real_folder "$REAL_FOLDER" --fake_folder "$FAKE_FOLDER"
done

echo "All evaluations completed!"

```
</details>

#### Recognition Accuracy

```
python evaluation/recognition.py --config ./config/recog.yaml --sks_name "thao" --iteration 6
```

### 🤗 Acknowledgements

This project will not be possible without the following open-source projects:
- [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://github.com/facebookresearch/chameleon)
- [Anole: An Open, Autoregressive and Native Multimodal Models for Interleaved Image-Text Generation](https://gair-nlp.github.io/anole/)
- and amazing HuggingFace's community: [Chamleon on HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/chameleon), [Anole on HuggingFace](https://github.com/huggingface/transformers/pull/32013)

And the unwavering supports from my Adobe's main mentor [Dr. Yuheng Li](https://yuheng-li.github.io/), my advisor [Prof. Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/), and meaningful discussion with [Dr. Krishna](https://krsingh.cs.ucdavis.edu/), [Dr. Jing Shi](https://jshi31.github.io/jingshi/), and [Dr. Trung Bui](https://sites.google.com/site/trungbuistanford/).

Sending thanks to my fellow mentee [Eslam Abdelrahman](https://scholar.google.com/citations?user=0OEerycAAAAJ&hl=en) for discussion about Vision-Langauge Model's latent space from time to time.
Big shout out to my (main) fellow mentee [Sicheng Mo](https://sichengmo.github.io/) -- He taught me a lot about coding. Without him, I would still using TensorBoard instead of WanDB (haha, also, he knows a lot of good foods/ restaurants!) (-- Last but not the least, thank mentor [Fangzhou Mu](https://fmu2.github.io/) for all the friday dinners/ board games he hosted during summer 🥓 🍣 🍱 But he does not like Thai foods though, meh~)



  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⣀⠀⠀⠀⢀⡤⠤⠤⣄⠀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
  ⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⠤⢴⣴⠒⠉⠹⣴⣏⠀⠀⠀⡀⠈⢇⠀⠀⣼⠀⠀⠀⠘⣶⠇⠀⢨⢃⡾⠓⠲⢤⠀⠀⠀⠀⠀⠀
  ⠀⠀⠀⣀⠤⠔⠒⠙⣯⣇⠀⠈⣿⣇⠀⠀⣿⣿⣿⠀⠀⣷⠀⠘⡄⠀⣿⠀⠀⠀⠀⢹⠀⠀⢸⡏⠇⠀⢀⠇⣀⠤⠒⠒⠤⣄
  ⢰⡖⠉⠀⠀⠀⠀⣀⣸⣿⠀⠀⠉⠉⠀⠀⢸⠁⣿⠀⠈⠉⠁⠀⢱⠀⣿⠀⠀⣦⠀⠀⠀⠀⣿⡸⠀⠀⠘⠉⠀⠀⣀⣤⣴⠟
  ⢼⢣⣀⣴⡀⠀⠘⡿⠏⠗⡆⠀⠠⣶⡆⠀⠸⡄⡏⠀⠀⣶⣷⠀⠀⢧⣿⠀⠀⣿⡆⠀⠀⢸⣿⠃⠀⢰⡄⠀⠐⡿⠛⠋⠀⠀
  ⠘⢿⡿⢿⣧⠀⠀⢳⠀⢸⠸⠀⠀⢹⣧⢀⣀⣷⣧⣤⣤⠛⣏⣦⣤⣾⣿⢦⣤⣿⢸⣄⣀⣼⡏⠀⢠⡟⡇⠀⠀⡇⠀⠀⠀⠀
  ⠀⠀⠀⠀⢏⢇⠀⠀⣣⠀⣆⣷⣶⣿⣿⡿⠿⠿⢷⡿⠟⣠⠟⠋⠛⢿⡛⠛⠿⡼⠿⠿⢿⣿⣿⣶⠞⡅⢸⠀⠀⢸⠀⠀⠀⠀
  ⠀⠀⠀⠀⠘⣾⣿⣿⠇⢠⣟⠉⠙⠷⡿⠀⠀⠀⢸⢀⡼⠁⠀⣀⠀⠀⠹⡄⡼⡇⠀⠀⡜⣸⡏⠙⠢⣧⣾⣦⣀⢸⠀⠀⠀⠀
  ⠀⠀⠀⠀⠀⠈⠀⠀⠀⢿⣿⣷⣦⡀⠀⠀⠀⠀⣇⡾⠀⠀⣼⣿⢷⠀⠀⢻⢱⠀⠀⢀⣿⡿⠀⠀⢠⠋⢻⡿⠿⣏⠀⠀⠀⠀
  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⣿⣿⠆⠀⠀⢸⡏⡇⠀⠀⡏⡟⡟⠀⠀⢸⡸⠀⠀⢸⣿⠃⠀⠀⡜⡰⢩⠃⠀⠈⣱⠀⠀⠀
  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢹⠀⠀⠀⢸⠀⡇⠀⠀⠙⠋⠀⠀⢀⡏⡇⠀⠀⠘⠋⠀⠀⣰⣱⢣⠇⠀⠀⣰⠃⠀⠀⠀
  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡘⡎⠀⠀⠀⡏⣿⣧⡀⠀⠀⠀⠀⢀⣾⣷⡇⠀⠀⠀⠀⠀⢠⣯⣧⣾⣦⣄⣰⠃⠀⠀⠀⠀
  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣧⣧⣤⣶⣶⠃⠘⢿⣿⣷⣶⣶⣾⠟⠉⣿⣿⣦⣄⣀⣠⣴⢏⣽⠋⠉⠙⢿⠁⠀⠀⠀⠀⠀
  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠛⠛⠛⠋⠁⠀⠀⠀⠉⠉⠉⠉⠀⠀⠀⠈⠛⠻⠿⠟⠋⠁⣿⣿⣦⣀⣀⡼⠀⠀⠀⠀⠀⠀
  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠛⠛⠛⠁⠀⠀⠀⠀⠀⠀⠀


*Finally, I've wrapped up this project -- I'll go home and hug my pets now ☕,  hehe (.❛ ᴗ ❛.)*

---