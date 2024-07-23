# import requests
# import torch

# from PIL import Image
# from transformers import ChameleonForConditionalGeneration
# from transformers import ChameleonProcessor

# processor = ChameleonProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
# model = ChameleonForConditionalGeneration.from_pretrained(
#     "leloy/Anole-7b-v0.1-hf",
#     device_map="auto",
# )

# # Get image of a snowman
# image_path = "./yollava-data/train/bo/0.png"
# image_snowman = Image.open(image_path)

# # Prepare a prompt
# prompt = "Draw a snowman.<image>"

# # Preprocess the prompt
# inputs = processor(prompt, images=[image_snowman], return_tensors="pt", padding=True).to(model.device)

# # Generate discrete image tokens
# # Note: We need to set `max_new_tokens` to 1026 since the model generates the `image_start_token` marker token first, then 1024 image tokens, and finally the `image_end_token` marker token.
# output_ids = model.generate(
#     **inputs,
#     multimodal_generation_mode="image-only",
#     max_new_tokens=1026,
# )

# # Decode the generated image tokens

# # Find boi token
# boi_token = 8197
# eoi_token = 8196
# # breakpoint()
# boi_index = (output_ids[0] == boi_token).nonzero().item()+1
# eoi_index = (output_ids[0] == eoi_token).nonzero().item()

# # pixel_values = model.decode_image_tokens(ouput_ids[:, 1:-1])
# # breakpoint()
# pixel_values = model.decode_image_tokens(output_ids[:, boi_index:eoi_index])
# images = processor.postprocess_pixel_values(pixel_values.detach().cpu().numpy())

# # Save the image
# images[0].save("test.png")

# ---- HuggingFace code: https://github.com/leloykun/transformers/blob/fc--anole/docs/source/en/model_doc/chameleon.md
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor

processor = ChameleonProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
model = ChameleonForConditionalGeneration.from_pretrained(
    "leloy/Anole-7b-v0.1-hf",
    device_map="auto",
)

# Prepare a prompt
prompt = "A photo of a Shiba Inu."

# Preprocess the prompt
inputs = processor(prompt, return_tensors="pt", padding=True).to(model.device)

# Generate discrete image tokens
generate_ids = model.generate(
    **inputs,
    multimodal_generation_mode="image-only",
    # Note: We need to set `max_new_tokens` to 1026 since the model generates the `image_start_token` marker token first, then 1024 image tokens, and finally the `image_end_token` marker token.
    max_new_tokens=1026,
    # This is important because most of the image tokens during training were for "empty" patches, so greedy decoding of image tokens will likely result in a blank image.
    do_sample=True,
)

# Only keep the tokens from the response
response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]

# Decode the generated image tokens
# breakpoint()
pixel_values = model.decode_image_tokens(response_ids[:, 1:-1].cpu())
images = processor.postprocess_pixel_values(pixel_values.detach().cpu().numpy())

# Save the image
images[0].save("test.png")