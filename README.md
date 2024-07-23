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
# Download pretrained models? -- This one is for code with TRANSFORMERS
mdkir chameleon-hf
cd chameleon-hf
git clone git@hf.co:facebook/chameleon-30b
git clone git@hf.co:facebook/chameleon-7b

# Instrall transformers with chameleon
cd ..
git clone -b chameleon https://github.com/huggingface/transformers.git
cd transformers
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

- [ ] YoLLaVA + Chameleon
	+ [x] Add new token
	+ [x] Dataloader
	+ [x] Training part
	+ [x] Run dummi code
	+ [x] Check loss -- Mask out question/prompt
- [ ] Chameleon + Anole?
	+ [x] Test text generation ability
	+ [x] Test image encoding
	+ [x] Test image generation
- [ ] YoLLaVA + Anole
	+ [x] Test text generation ability
	+ [x] Test image encoding
	+ [x] Test image generation
	+ [x] Dataloader
	+ [x] Training part
	+ [x] Run dummi code
	+ [x] Check loss -- Compute loss on images?
- [ ] YoLLaVA + Total
	+ [x] Test text generation ability
	+ [x] Test image encoding
	+ [x] Test image generation
	+ [x] Dataloader
	+ [x] Training part
	+ [x] Run dummi code
	+ [x] Check loss -- Compute loss on images?

- [ ] Note
	+ [ ] ⚠️ Currently all Q/A has images -- IMPORTANT, need to fix soon
	+ [x] Generate corresponding answer for Chameleon is not ready yet
	+ [ ] Right now code is syned between 2 GPU -- How to fit them into a single GPU?
	+ [ ] ⚠️ Check what's the logic of the generate mode in Chameleon inference code