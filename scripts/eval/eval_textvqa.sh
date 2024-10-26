#wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip

cd evaluation/forgetting
mkdir data
cd data
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip
cd ..

EVAL_FOLDER='/sensei-fs/users/thaon/code/eval-llava'
SAVE_LOCATION=$EVAL_FOLDER/textvqa/anole.jsonl
IMAGE_FOLDER='data'

CUDA_VISIBLE_DEVICES=1 python model_vqa_loader.py \
    --question_file $EVAL_FOLDER/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image_folder $IMAGE_FOLDER/train_images \
    --save_location $SAVE_LOCATION \
    --temperature 0 \
    --max_new_tokens 10

# python eval_textvqa.py \
#     --annotation-file $EVAL_FOLDER/textvqa/TextVQA_0.5.1_val.json \
#     --result-file $EVAL_FOLDER/textvqa/answers/result.jsonl