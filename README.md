# My Chameleon

### Getting Started

```
# extract the dataset
rsync --progress /sensei-fs/users/thaon/data.tar.bz2 .
tar -cjvf data.tar.bz2 data
```

### Prerequisites

```
# Download pretrained models?
mdkir chameleon-hf
cd chameleon-hf
git clone git@hf.co:facebook/chameleon-30b
git clone git@hf.co:facebook/chameleon-7b
```

### TODO 📝

- [ ] YoLLaVA + Chameleon
	+ [ ] Add new token
	+ [ ] Dataloader
	+ [ ] Training part
	+ [ ] Run dummi code