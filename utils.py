import os 
import shutil

import torch

from functools import cached_property
from torch.utils.tensorboard import SummaryWriter
from transformers import ChameleonProcessor
from transformers import ChameleonVQVAE
from transformers import ChameleonVQVAEConfig
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union
# from utils import ChameleonImageVocabularyMapping
# from utils import ChameleonVQVAEPreprocessor

START_OF_IMAGE_INDEX = 8197 # <racm3:break>
END_OF_IMAGE_INDEX = 8196 # <eoss>
END_OF_TURN = 8710
PAD_INDEX = 1

class ChameleonVQVAEPreprocessor(ChameleonVQVAE):
    def __init__(self, config: ChameleonVQVAEConfig):
        super().__init__(config)
        self.vocabulary_mapping = ChameleonImageVocabularyMapping(
            config.vocabulary_map,
            config.image_token_id,
            config.boi_token_id,
            config.eoi_token_id,
        )
        
        self.register_buffer(
            "img2bpe_mapping_tensor",
            self.vocabulary_mapping.img2bpe_mapping_tensor,
            persistent=False,
        )
        self.register_buffer(
            "bpe2img_mapping_tensor",
            self.vocabulary_mapping.bpe2img_mapping_tensor,
            persistent=False,
        )
    @cached_property
    def img2bpe_mapping_tensor(self):
        mapping = torch.zeros(max(self.img2bpe.keys()) + 1, dtype=torch.int)
        for k, v in self.img2bpe.items():
            mapping[k] = v
        return mapping

    @cached_property
    def bpe2img(self):
        img_tkn_chr_mapping = {chr(ord("A") + i): str(i) for i in range(10)}

        def remap(old_name: str) -> str:
            return "".join(img_tkn_chr_mapping.get(c, c) for c in old_name[len("IMGIMG") : -1])

        return {tok: int(remap(self.val2name[tok])) for tok in self.image_token_ids}

    @cached_property
    def img2bpe(self):
        return {v: k for k, v in self.bpe2img.items()}

    def convert_img2bpe_tokens(self, img_batch: torch.LongTensor) -> torch.LongTensor:
        """
        Converts image tokens generated by the VQVAE model into BPE tokens compatible with the text tokenizer.

        Notes:
            - It is important to move the `img_batch` tensor to the same device as the `img2bpe_mapping_tensor` buffer
            as Accelerate may move the buffer to a different device when loading the model with `device_map="auto"`.
            - Accelerate up to version 0.33.0 (and also maybe later versions) has a bug where buffers in downstream modules
            may be ignored when inferring the proper device map. See: https://github.com/huggingface/accelerate/blob/79ca85c27df292dbf64cfa2bcc12dbb62fbe9267/src/accelerate/utils/modeling.py#L1273
            This causes the `img2bpe_mapping_tensor` buffer to be placed on the CPU by default, which may cause a performance
            loss--especially with prompts that contain many images. No action needs to be done when this bug is fixed.

        Args:
            img_batch (`torch.Tensor` of shape `(batch_size, image_seq_length)`):
                The image tokens generated by the VQVAE model.

        Returns:
            `torch.Tensor` of shape `(batch_size, image_seq_length)`:
                The image tokens converted to be compatible with the text tokenizer's BPE tokens.
        """
        # device = img_batch.device
        img_tokens = self.img2bpe_mapping_tensor[img_batch.to(self.img2bpe_mapping_tensor)]
        return img_tokens#.to(device)

    def get_image_tokens(self, pixel_values: torch.FloatTensor):
        """
        Tokenizes images into discrete tokens with VQGAN module. Converts
        obtained image tokens into BPE tokens and wraps with "boi" and "eoi"
        special tokens.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
                The tensors corresponding to the input images.

        Returns:
            `torch.Tensor` of shape `(batch_size, image_seq_length)`:
                The BPE tokens generated by the model.
        """
        _, _, image_toks = self.encode(pixel_values)
        return self.convert_img2bpe_tokens(image_toks)

    def decode_image_tokens(self, bpe_tokens: torch.LongTensor) -> torch.LongTensor:
        """
        Converts BPE tokens generated by the model into discrete image tokens
        compatible with the VQGAN module, then decodes them into pixel values.

        Args:
            bpe_tokens (`torch.tensor` of shape `(batch, image_seq_length)`):
                The BPE tokens generated by the model.

        Returns:
            `torch.Tensor` of shape `(batch, num_channels, 512, 512)`:
        """
        if bpe_tokens.shape[1] != self.image_seq_length:
            raise ValueError(f"All batches must have {self.image_seq_length} tokens.")
        image_tensor = self.convert_bpe2img_tokens(bpe_tokens)
        return self.decode(image_tensor)

class ChameleonImageVocabularyMapping:
    """
    A class for mapping discrete image tokens from VQGAN to BPE tokens.
    this code is copy from https://github.com/huggingface/transformers/blob/8820fe8b8c4b9da94cf1e4761876f85c562e0efe/src/transformers/models/chameleon/modeling_chameleon.py#L1083
    """

    def __init__(
        self,
        vocab_map: Dict[str, int],
        image_token_id: int,
        boi_token_id: int,
        eoi_token_id: int,
    ):
        self.vocab_map = vocab_map
        self.image_token_id = image_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id

    @cached_property
    def val2name(self):
        return {v: k for k, v in self.vocab_map.items()}

    @cached_property
    def image_token_ids(self):
        return sorted([val for name, val in self.vocab_map.items() if name.startswith("IMGIMG")])

    @cached_property
    def bpe2img(self):
        img_tkn_chr_mapping = {chr(ord("A") + i): str(i) for i in range(10)}

        def remap(old_name: str) -> str:
            return "".join(img_tkn_chr_mapping.get(c, c) for c in old_name[len("IMGIMG") : -1])

        return {tok: int(remap(self.val2name[tok])) for tok in self.image_token_ids}

    @cached_property
    def img2bpe(self):
        return {v: k for k, v in self.bpe2img.items()}

    @cached_property
    def bpe2img_mapping_tensor(self):
        mapping = torch.zeros(max(self.bpe2img.keys()) + 1, dtype=torch.int)
        for k, v in self.bpe2img.items():
            mapping[k] = v
        return mapping

    @cached_property
    def img2bpe_mapping_tensor(self):
        mapping = torch.zeros(max(self.img2bpe.keys()) + 1, dtype=torch.int)
        for k, v in self.img2bpe.items():
            mapping[k] = v
        return mapping

def setup(config):
    log_dir = f'./runs/{config.exp_name}/{config.sks_name}'
    
    if os.path.exists(log_dir):
        # Delete the directory and its contents
        shutil.rmtree(log_dir)
    writer = SummaryWriter(f'./runs/{config.exp_name}/{config.sks_name}')
    save_location = f'{config.savedir}/{config.exp_name}/{config.sks_name}'
    os.makedirs(save_location, exist_ok=True)
    return writer, save_location

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def collate_fn(batch):
    inputs = [item['input'] for item in batch]
    images = [item['image'] for item in batch]
    img_gen_bools = [item['image_generation'] for item in batch]
    # question = [f'{questions[i]}{answers[i]}' for i in range(len(questions))]
    example = processor(inputs, images, padding=True)
    example['labels'] = example['input_ids'].clone()

    # Find the index of the first occurrence of END_OF_TURN in each sequence
    batch_size, seq_len = example['labels'].shape
    eot_mask = example['labels'] == END_OF_TURN
    eot_indices = torch.argmax(eot_mask.int(), dim=1)

    # Create a mask for the positions to be replaced with -100
    mask = torch.arange(seq_len).expand(batch_size, seq_len) < eot_indices.unsqueeze(1)

    # Apply the mask to the labels
    example['labels'][mask] = -100
    example['img_gen_bools'] = img_gen_bools
    return example