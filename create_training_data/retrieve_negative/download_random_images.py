#!/usr/bin/env python

# import cv2
import functools
import os
import pytorch_lightning
import torch
import torchvision

from PIL import Image
from tqdm import tqdm

import piat # https://git.corp.adobe.com/sniklaus/piat

objDataloader = piat.Dataloader(
    intBatchsize=1,
    intWorkers=2,
    intThreads=4,
    strQueryfile=['s3://sniklaus-clio-query/*/origin=l2'],
    intSeed=0,
    funcStages=[
        functools.partial(piat.image_load, {'strSource': '1024-pil-antialias'}),
        functools.partial(piat.image_alpha_smartskip, {}),
        # functools.partial(piat.image_resize_antialias, {'intSize': 128}),
        # functools.partial(piat.image_crop_smart, {'intSize': 128}),
        functools.partial(piat.text_load, {}),
        functools.partial(piat.output_image, {'fltMean': [0.5, 0.5, 0.5], 'fltStd': [0.5, 0.5, 0.5]}),
        functools.partial(piat.output_text, {}),
    ],
)

index = 0
for intBatch, objBatch in enumerate(tqdm(objDataloader)):
    image_tensor = objBatch['tenImage'][0]
    image = (0.5 * image_tensor.flip([0])) + 0.5
    image_np = image.numpy().transpose(1, 2, 0)
    image_np = image_np[:, :, [2, 1, 0]]
    pil_image = Image.fromarray((image_np * 255).astype('uint8'))  # Scale to [0, 255]

    save_location = os.path.join('../../../data/yochameleon-data/random_negative_example', f'{index}.png')
    pil_image.save(save_location)
    print(f"Image saved as {save_location}")
    index += 1
    if index == 1000:
        exit()