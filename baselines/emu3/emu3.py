import torch

from PIL import Image
from transformers import AutoImageProcessor
from transformers import AutoModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers.generation import LogitsProcessorList
from transformers.generation import PrefixConstrainedLogitsProcessor
from transformers.generation import UnbatchedClassifierFreeGuidanceLogitsProcessor
from transformers.generation.configuration_utils import GenerationConfig

from emu3.mllm.processing_emu3 import Emu3Processor


# model path
EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"

# prepare model and processor
model = AutoModelForCausalLM.from_pretrained(
    EMU_HUB,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map="cuda:0", trust_remote_code=True).eval()
processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

# prepare input
POSITIVE_PROMPT = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."

classifier_free_guidance = 3.0
prompt = "a portrait of young girl."
prompt += POSITIVE_PROMPT

kwargs = dict(
    mode='G',
    ratio="1:1",
    image_area=model.config.image_area,
    return_tensors="pt",
)
pos_inputs = processor(text=prompt, **kwargs)
neg_inputs = processor(text=NEGATIVE_PROMPT, **kwargs)

# prepare hyper parameters
GENERATION_CONFIG = GenerationConfig(
    use_cache=True,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.pad_token_id,
    max_new_tokens=40960,
    do_sample=True,
    top_k=2048,
)

h, w = pos_inputs.image_size[0]
constrained_fn = processor.build_prefix_constrained_fn(h, w)
logits_processor = LogitsProcessorList([
    UnbatchedClassifierFreeGuidanceLogitsProcessor(
        classifier_free_guidance,
        model,
        unconditional_ids=neg_inputs.input_ids.to("cuda:0"),
    ),
    PrefixConstrainedLogitsProcessor(
        constrained_fn ,
        num_beams=1,
    ),
])

# generate
outputs = model.generate(
    pos_inputs.input_ids.to("cuda:0"),
    GENERATION_CONFIG,
    logits_processor=logits_processor
)

mm_list = processor.decode(outputs[0])
for idx, im in enumerate(mm_list):
    if not isinstance(im, Image.Image):
        continue
    im.save(f"result_{idx}.png")