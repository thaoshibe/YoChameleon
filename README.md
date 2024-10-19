# Yo'Chameleon

<img src="./images/yochameleon.png" alt="YoChameleon" width="400">

### Getting Started

```
bash install.sh
```

### Creating dataset

```
# Retrieve data from LAION-2B
cd create_training_data/retrieve_negative
bash retrieve.sh
```

### Training

```

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