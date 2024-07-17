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

### TODO 📝

- [ ] YoLLaVA + Chameleon
	+ [ ] Add new token
	+ [ ] Dataloader
	+ [ ] Training part
	+ [ ] Run dummi code