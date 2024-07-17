import requests
import torch

import argparse

from PIL import Image
from dataset import PersonalizedDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
from transformers import ChameleonForCausalLM
from transformers import ChameleonProcessor

def get_args():
    parser = argparse.ArgumentParser(description='Chameleon')
    # model related
    parser.add_argument('--image', type=str, default='./chameleon/inference/examples/thao-bo.jpeg', help='Path to image')
    parser.add_argument('--prompt', type=str, default="What is the color of the dog? <image>", help='Prompt')
    parser.add_argument('--model_id', type=str, default='./chameleon-hf/chameleon-7b', help='Model ID')

    # personalized token related
    parser.add_argument('--sks_name', type=str, default='sks', help='Name of the personalized token')
    parser.add_argument('--prefix_token', type=int, default=16, help='Number of prefix tokens')

    # hyperparameters
    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    model_id = args.model_id
    processor = ChameleonProcessor.from_pretrained(model_id)
    model = ChameleonForCausalLM.from_pretrained(model_id, device_map="auto")
    print(f'Loaded {model_id}!')

    # --- Add personalized tokens
    prefix_tokens = [f'<reserved{16301+i}>' for i in range(args.prefix_token)]
    personalized_tokens = [f'<reserved16300>']
    personalized_tokens.extend(prefix_tokens)
    sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}."
    personalized_token_ids = processor.tokenizer.convert_tokens_to_ids(personalized_tokens)

    print(f'Personalized tokens will be: {personalized_tokens}')
    print(f'Personalized token ids will be: {personalized_token_ids}')
    print(f'Personalized prompt: {sks_prompt}')

    # --- Dataloader
    train_dataset = PersonalizedDataset(
        data_root="./example_training_data/",
        sks_name='mam',
        model_id=model_id
        )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=1
    )

    # --- Prepare for training
    token_embeds = model.get_input_embeddings().weight.data # this should have shape: torch.Size([65536, 8192]) which is #vocab x token-embed
    orig_embeds_params = model.get_input_embeddings().weight.data.clone()
    orig_lm_params = model.lm_head.weight.data.clone()
    trainable_params = [model.get_input_embeddings().weight, model.lm_head.weight]

    optimizer = torch.optim.AdamW(
        trainable_params, # for optimize the embeddings and the head
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    model.model.requires_grad_(False)
    model.model.embed_tokens.weight.requires_grad_(True)
    # --- Start training
    for epoch in tqdm(range(0, args.epoch)):
        for names, p in model.named_parameters():
            if p.requires_grad:
                print(names, "requires_grad")
        for step, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            # model(batch['input_ids'][0], batch['pixel_values'][0], batch['attention_mask'])
            breakpoint()
            # model(batch['input_ids'][0])
            # model(batch['input_ids'][0]).loss

            # --- Prepare the inputs
            # query = batch['query']
            # answer = batch['answer']
            # image = batch['image']
            # has_image = batch['has_image']
            # image_sizes = batch['image_sizes']
            # image = image.unsqueeze(0)
            # image = image.to('cuda')

            # # --- Prepare the prompt
            # prompt = query[0] + ' ' + sks_prompt
            # inputs = processor(prompt, image, return_tensors="pt")
            # inputs = {k: v.to('cuda') for k, v in inputs.items()}

            # --- Forward pass
