import cv2
import glob
import numpy as np
import os
import requests
import torch

from PIL import Image
from diffusers import DiffusionPipeline
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
# For the first time of using,
# you need to download the huggingface repo "BAAI/Emu2-GEN" to local first
path = "/mnt/localssd/code/Emu2-Gen"

multimodal_encoder = AutoModelForCausalLM.from_pretrained(
    f"{path}/multimodal_encoder",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="bf16"
)
tokenizer = AutoTokenizer.from_pretrained(f"{path}/tokenizer")

pipe = DiffusionPipeline.from_pretrained(
    path,
    custom_pipeline="pipeline_emu2_gen",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="bf16",
    multimodal_encoder=multimodal_encoder,
    tokenizer=tokenizer,
)

# For the non-first time of using, you can init the pipeline directly
pipe = DiffusionPipeline.from_pretrained(
    path,
    custom_pipeline="pipeline_emu2_gen",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="bf16",
)

pipe.to("cuda")

for index in tqdm(range(0,100)):
    prompt = "Generater a photo of a person with long, dark hair, often seen wearing stylish, comfortable outfits."
    ret = pipe(prompt)
    ret.image.save(f"/sensei-fs/users/thaon/generated_images/emu2/0/{index}.png")

    # image editing
    list_images = glob.glob(os.path.join('/mnt/localssd/code/data/yollava-data/train/thao/', '*.png'))[:4]
    # image = Image.open('/mnt/localssd/code/data/yollava-data/train/thao/0.png')
    image = Image.open(list_images[index % 4])
    prompt = [image, "Another photo of this person"]
    ret = pipe(prompt)
    ret.image.save(f"/sensei-fs/users/thaon/generated_images/emu2/1/{index}.png")

# grounding generation
# def draw_box(left, top, right, bottom):
#     mask = np.zeros((448, 448, 3), dtype=np.uint8)
#     mask = cv2.rectangle(mask, (left, top), (right, bottom), (255, 255, 255), 3)
#     mask = Image.fromarray(mask)
#     return mask

# dog1 = Image.open(requests.get('https://github.com/baaivision/Emu/Emu2/examples/dog1.jpg?raw=true',stream=True).raw).convert('RGB')
# dog2 = Image.open(requests.get('https://github.com/baaivision/Emu/Emu2/examples/dog2.jpg?raw=true',stream=True).raw).convert('RGB')
# dog3 = Image.open(requests.get('https://github.com/baaivision/Emu/Emu2/examples/dog3.jpg?raw=true',stream=True).raw).convert('RGB')
# dog1_mask = draw_box( 22,  14, 224, 224)
# dog2_mask = draw_box(224,  10, 448, 224)
# dog3_mask = draw_box(120, 264, 320, 438)

# prompt = [
#     "<grounding>",
#     "An oil painting of three dogs,",
#     "<phrase>the first dog</phrase>"
#     "<object>",
#     dog1_mask,
#     "</object>",
#     dog1,
#     "<phrase>the second dog</phrase>"
#     "<object>",
#     dog2_mask,
#     "</object>",
#     dog2,
#     "<phrase>the third dog</phrase>"
#     "<object>",
#     dog3_mask,
#     "</object>",
#     dog3,
# ]
# ret = pipe(prompt)
# ret.image.save("three_dogs.png")

# # Autoencoding
# # to enable the autoencoding mode, you can only input exactly one image as prompt
# # if you want the model to generate an image,
# # please input extra empty text "" besides the image, e.g.
# #   autoencoding mode: prompt = image or [image]
# #   generation mode: prompt = ["", image] or [image, ""]
# prompt = Image.open("./examples/doodle.jpg").convert("RGB")
# ret = pipe(prompt)
# ret.image.save("doodle_ae.png")
