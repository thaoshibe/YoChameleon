import requests
import torch

from PIL import Image
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor

processor = ChameleonProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf")
model = ChameleonForConditionalGeneration.from_pretrained("leloy/Anole-7b-v0.1-hf", torch_dtype=torch.bfloat16, device_map="cuda")

# prepare image and text prompt
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open('../../yochameleon-data/train/chua-thien-mu/0.png')
prompt = "This is a photo of a subject. <image> Can you describe the subject?"

inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True))