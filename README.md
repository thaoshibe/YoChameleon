# Yo'Chameleon

<img src="./images/yochameleon.png" alt="YoChameleon" width="300">

### Getting Started

```
# extract the dataset
rsync --progress /sensei-fs/users/thaon/data.tar.bz2 .
tar -cjvf data.tar.bz2 data
```

### Prerequisites

```
git clone https://github.com/huggingface/transformers.git
cd transformers
git fetch origin pull/32013/head:pr-32013
git checkout pr-32013
pip install -e .
```

### Training

```
# Chameleon + YoLLaVA
python train_yochameleon.py --sks_name bo
test_yochameleon.py --sks_name bo --epoch 10

# Anole + YoLLaVA
python train_anole.py --sks_name bo

# Chameleon + Anole + YoLLaVA
python train.py --sks_name bo
```

### Create training json

```
# this will create llava-like json file
cd preprocess
python create_img_gen.py --image_folder /mnt/localssd/code/YoChameleon/yollava-data/train --output_dir /mnt/localssd/code/YoChameleon/example_training_data/v1 --sks_name mam
```

### Create augmented training photos

```
python data_augmentation.py --image_folder /mnt/localssd/code/data/minimam --output_folder /mnt/localssd/code/data/minimam/augmented --num_augmented_images 500
```