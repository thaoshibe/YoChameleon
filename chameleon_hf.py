import requests
import torch

from PIL import Image
from transformers import ChameleonForCausalLM
from transformers import ChameleonProcessor

processor = ChameleonProcessor.from_pretrained("meta-chameleon")
model = ChameleonForCausalLM.from_pretrained("meta-chameleon", torch_dtype=torch.float16, device_map="auto") 

# prepare image and text prompt
url = "https://bjiujitsu.com/wp-content/uploads/2021/01/jiu_jitsu_belt_white_1.jpg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "What color is the belt in this image?<image>"

inputs = processor(prompt, image, return_tensors="pt").to(model.device)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True))