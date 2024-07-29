import argparse
import os
# from transformers import ChameleonForCausalLM
import shutil

import requests
import torch

import os

import torch

from PIL import Image
from dataset import PersonalizedDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor

START_OF_IMAGE_INDEX = 8197 # <racm3:break>
END_OF_IMAGE_INDEX = 8196 # <eoss>
END_OF_TURN = 8710
PAD_INDEX = 1

def collate_fn(batch):
    inputs = [item['input'] for item in batch]
    images = [item['image'] for item in batch]
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

    # If gen_image, then replace labels to image tokens

    #     vqgan_ids = model.model.get_image_tokens(pixel_values=batch['pixel_values']).to(batch['labels'].dtype)
    #     mask = (batch['labels'] != -100)
    #     batch['labels'][mask] = vqgan_ids.to(batch['labels'].device)

    # Move to device
    # example = {key: value.to(model.device) for key, value in example.items()}
    # eot_index = torch.nonzero(example['labels']==END_OF_TURN).item()
    # soi_index = torch.nonzero(example['labels']==START_OF_IMAGE_INDEX).item()
    # input_ids = torch.stack(input_ids)
    # attention_mask = torch.stack(attention_mask)

    return example

def get_args():
    parser = argparse.ArgumentParser(description='YoAnole')
    # model related
    parser.add_argument('--model_id', type=str, default='leloy/Anole-7b-v0.1-hf', help='Model ID')
    parser.add_argument('--data_root', type=str, default='./example_training_data/', help='Model ID')

    # personalized token related
    parser.add_argument('--sks_name', type=str, default='sks', help='Name of the personalized token')
    parser.add_argument('--prefix_token', type=int, default=16, help='Number of prefix tokens')

    # hyperparameters
    parser.add_argument('--epoch', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--savedir', type=str, default='./ckpt/', help='Directory to save the model')
    parser.add_argument('--exp_name', type=str, default='total', help='Name of experiement')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    log_dir = f'./runs/{args.exp_name}/{args.sks_name}'
    
    if os.path.exists(log_dir):
        # Delete the directory and its contents
        shutil.rmtree(log_dir)
    writer = SummaryWriter(f'./runs/{args.exp_name}/{args.sks_name}')
    save_location = f'./{args.savedir}/{args.exp_name}/{args.sks_name}'
    os.makedirs(save_location, exist_ok=True)

    model_id = args.model_id
    processor = ChameleonProcessor.from_pretrained(model_id)
    # model = ChameleonForCausalLM.from_pretrained(model_id, device_map="auto")
    model = ChameleonForConditionalGeneration.from_pretrained(model_id, device_map="auto")
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
        data_root=args.data_root,
        sks_name=args.sks_name,
        personalized_prompt = sks_prompt,
        )
    # print(train_dataset[100])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn
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
    index_no_updates = torch.ones((len(processor.tokenizer),), dtype=torch.bool)
    index_no_updates[personalized_token_ids] = False

    # --- Start training
    for epoch in tqdm(range(0, args.epoch)):
        for names, p in model.named_parameters():
            if p.requires_grad:
                print(names, "requires_grad")
        for step, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            
            # Move tensors to device
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            pixel_values = batch['pixel_values'].to(model.device)
            labels = batch['labels'].to(model.device)
            # Forward pass
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )
            loss = output.loss
            loss.backward()
            # print(loss)
            optimizer.step()
            with torch.no_grad():
                model.get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds_params[index_no_updates]
                model.lm_head.weight[index_no_updates] = orig_lm_params[index_no_updates]
            writer.add_scalar('Loss/Loss', loss, epoch*len(train_dataloader)+step)
            
            writer.add_scalar('Norm/Vocab-Not-Update-Norm', model.get_input_embeddings().weight[index_no_updates].norm(), epoch*len(train_dataloader)+step)
            writer.add_scalar('Norm/Vocab', model.get_input_embeddings().weight.norm(), epoch*len(train_dataloader)+step)
        if epoch % 2 == 0:
            print('Save model at: ', save_location)
            save_path_token = os.path.join(save_location, f'{epoch}-token.pt')
            save_path_lmhead = os.path.join(save_location, f'{epoch}-lmhead.pt')
            torch.save(model.get_input_embeddings().weight.data[personalized_token_ids], save_path_token)
            torch.save(model.lm_head.weight.data[personalized_token_ids], save_path_lmhead)

            # with torch.no_grad():
            #     try:
            #     inputs = processor(sks_prompt+"<racm3:break>", return_tensors="pt", padding=True).to(model.device)
            #     output_ids = model.generate(**inputs, multimodal_generation_mode="image-only", max_new_tokens=1027, do_sample=True,)
            #     # boi_index = (output_ids[0] == boi_token).nonzero().item()+1
            #     # eoi_index = (output_ids[0] == eoi_token).nonzero().item()
            #     image_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
            #     # image_ids = output_ids[:, boi_index:eoi_index]
            #     pixel_values = model.decode_image_tokens(image_ids[:, :-1].cpu())
            #     images = processor.postprocess_pixel_values(pixel_values.detach().cpu().numpy())