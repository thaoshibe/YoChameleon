# Create Training Data

### Image Retrieval


### Image Inpainting

### Prompt for image inpainting captions

```
# Create background replacement
python create_augmentation_mask.py --image_folder ../../data/minibo/ --output_folder /mnt/localssd/code/data/minibo
python create_augmentation_mask.py --image_folder ../../data/minimam/ --output_folder /mnt/localssd/code/data/minimam  

python sdxl-inpainting.py --image_folder /mnt/localssd/code/data/minibo/foreground --mask_folder /mnt/localssd/code/data/minibo/mask --output_folder /mnt/localssd/code/data/minibo/inpainted

```

### Image Captioning

-- Change to folder?

