import requests
import torch

from PIL import Image
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor

processor = ChameleonProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
model = ChameleonForConditionalGeneration.from_pretrained(
    "leloy/Anole-7b-v0.1-hf",
    device_map="auto",
)

# Get image of a snowman
image_path = "./yollava-data/train/bo/0.png"
image_snowman = Image.open(image_path)

# Prepare a prompt
prompt = "Draw a snowman.<image>"

# Preprocess the prompt
inputs = processor(prompt, images=[image_snowman], return_tensors="pt", padding=True).to(model.device)

# Generate discrete image tokens
# Note: We need to set `max_new_tokens` to 1026 since the model generates the `image_start_token` marker token first, then 1024 image tokens, and finally the `image_end_token` marker token.
output_ids = model.generate(
    **inputs,
    multimodal_generation_mode="image-only",
    max_new_tokens=1026,
)

# Decode the generated image tokens

# Find boi token
boi_token = 8197
eoi_token = 8196
# breakpoint()
boi_index = (output_ids[0] == boi_token).nonzero().item()+1
eoi_index = (output_ids[0] == eoi_token).nonzero().item()

# pixel_values = model.decode_image_tokens(ouput_ids[:, 1:-1])
# breakpoint()
pixel_values = model.decode_image_tokens(output_ids[:, boi_index:eoi_index])
images = processor.postprocess_pixel_values(pixel_values.detach().cpu().numpy())

# Save the image
images[0].save("test.png")