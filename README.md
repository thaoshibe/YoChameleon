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


### TODO 📝

- [ ] Clean code first -- So that we can easily add more data augmentation
	+ [ ] Write dataloader from file? (json file)
	+ [ ] Write config file? -- Maybe not necessary
	+ [ ] Check data format: `personalized prompt -- caption -- <image>`
- [ ] Data Augmentation
	+ [ ] Captioning
		+ [ ] Write code for detail caption for each image
	+ [ ] Image augmentation
		+ [ ] Normal image augmentation
		+ [ ] Part-segmentation
	+ [ ] Compute loss on subject only?

In general, we have these version:
- [ ] v0: 500 imgs
- [ ] v1: data augmentation only
- [ ] v2: detail image captioning
- [ ] v3: part-segementation
- [ ] v4: compute loss on image segmentation
- [ ] v5: all?