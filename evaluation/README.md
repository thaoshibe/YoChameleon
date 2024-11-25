# Evaluation - Evaluation - Evaluation

### Facial similarity scores

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

### CLIP Image-to-Image Similarity

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

### Recognition Accuracy

```
python evaluation/recognition.py --config ./config/recog.yaml --sks_name "thao" --iteration 6
```

```
# If you want to test with concatenate prompt
python evaluation/recognition_concatenate.py --config ./config/recog.yaml --sks_name "thao" --iteration "20" --exp_name "gen" --exp_name_2 "recog"
```

## Forgetting evaluation

### POPE

You will have to download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view), unzip it.
After extract, you should find `pope/llava_pope_test.jsonl`.

Then, you can run the following command:

```bash
bash scripts/eval/eval_pope.sh
```

### MMBench

This is also almost done, but this requires external online evaluator, but i haven't figure out how to submit. 
Anyway, currently i have eval code and also conversion code (convert into submittable format). But in the conversion code, i manually calculate accuracy which we can use it for relative comparison for now. (but this number is not the metric used in the paper, thus, cannot report in the paper) 

You need to have `mmbench_dev_20230712.tsv`, if not, run `download_data.sh`


