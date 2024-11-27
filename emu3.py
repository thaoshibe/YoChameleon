import requests
import torch

from PIL import Image
from transformers import Emu3ForConditionalGeneration
from transformers import Emu3Processor

processor = Emu3Processor.from_pretrained("Emu3-community/Emu3-Gen-hf")
model = Emu3ForConditionalGeneration.from_pretrained("Emu3-community/Emu3-Gen-hf", torch_dtype="bfloat16", device_map="auto", attn_implementation="flash_attention_2")
breakpoint()

inputs = processor(
    text=["a portrait of young girl. masterpiece, film grained, best quality.", "a dog running under the rain"],
    padding=True,
    return_tensors="pt",
    return_for_image_generation=True,
)
inputs = inputs.to(device="cuda:0", dtype=torch.bfloat16)

neg_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
neg_inputs = processor(text=[neg_prompt] * 2, return_tensors="pt").to(device="cuda:0")

image_sizes = inputs.pop("image_sizes")
HEIGHT, WIDTH = image_sizes[0]
VISUAL_TOKENS = model.vocabulary_mapping.image_tokens

def prefix_allowed_tokens_fn(batch_id, input_ids):
    height, width = HEIGHT, WIDTH
    visual_tokens = VISUAL_TOKENS
    image_wrapper_token_id = processor.tokenizer.encode("<|image token|>", return_tensors="pt")[0].to(model.device)
    eoi_token_id = processor.tokenizer.encode("<|image end|>", return_tensors="pt")[0]
    eos_token_id = processor.tokenizer.encode("<|extra_204|>", return_tensors="pt")[0]
    pad_token_id = processor.tokenizer.encode("<|endoftext|>", return_tensors="pt")[0]
    eol_token_id = processor.tokenizer.encode("<|extra_200|>", return_tensors="pt")[0]
    eof_token_id = processor.tokenizer.encode("<|extra_201|>", return_tensors="pt")[0]

    position = torch.nonzero(input_ids == image_wrapper_token_id, as_tuple=True)[0][0]
    offset = input_ids.shape[0] - position
    if offset % (width + 1) == 0:
        return (eol_token_id, )
    elif offset == (width + 1) * height + 1:
        return (eof_token_id, )
    elif offset == (width + 1) * height + 2:
        return (eoi_token_id, )
    elif offset == (width + 1) * height + 3:
        return (eos_token_id, )
    elif offset > (width + 1) * height + 3:
        return (pad_token_id, )
    else:
        return visual_tokens

# breakpoint()
image = Image.open('/mnt/localssd/code/data/yochameleon-data/train/bo/8.png')
prompt = "What do you see in this image?<image>"

inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
image_tokens = model.get_image_tokens(inputs.pixel_values, image_sizes=inputs.image_sizes)
model.decode_image_tokens(image_tokens[None, :], width=64, height=64)

out = model.generate(
    **inputs,
    max_new_tokens=50_000, # make sure to have enough tokens for one image
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    return_dict_in_generate=True,
    negative_prompt_ids=neg_inputs.input_ids, # indicate for Classifier-Free Guidance
    negative_prompt_attention_mask=neg_inputs.attention_mask,
)

image = model.decode_image_tokens(out.sequences[:, inputs.input_ids.shape[1]: ], height=HEIGHT, width=WIDTH)
images = processor.postprocess(list(image.float()), return_tensors="PIL.Image.Image") # internally we convert to np but it's not supported in bf16 precision
for i, image in enumerate(images['pixel_values']):
    image.save(f"result{i}.png")
