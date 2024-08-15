import torch

from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor
from transformers.image_transforms import to_pil_image

processor = ChameleonProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
model = ChameleonForConditionalGeneration.from_pretrained(
    "leloy/Anole-7b-v0.1-hf",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Prepare a prompt
prompt = "Generate an image of a snowman."

# Preprocess the prompt
inputs = processor(prompt, padding=True, return_tensors="pt").to(model.device, dtype=model.dtype)

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
pixel_values = model.decode_image_tokens(response_ids[:, 1:-1])
images = processor.postprocess_pixel_values(pixel_values)
image = to_pil_image(images[0].detach().cpu())
# Save the image
image.save("test.png")