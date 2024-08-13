import requests
import torch

from PIL import Image
from transformers import SamModel
from transformers import SamProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

img_path = '../yollava-data/train/bo/9.png'
raw_image = Image.open(img_path).convert("RGB").resize((512, 512))
input_points = [[[256,256]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
# scores = outputs.iou_scores
mask = masks[0].squeeze(0).permute(1, 2, 0)  # change shape to [1764, 2646, 3]
# breakpoint()
mask = mask.cpu().numpy()[:, :, 0].astype('uint8')*255

Image.fromarray(mask, mode='L').save('bo-body.png')