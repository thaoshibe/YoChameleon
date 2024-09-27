import argparse

import os
import requests
import torch
import yaml

from PIL import Image
from dataset import PersonalizedDataset
from itertools import cycle
from torchvision import datasets
from tqdm import tqdm
from transformers import ChameleonForConditionalGeneration
from transformers import ChameleonProcessor
from transformers.image_transforms import to_pil_image
from utils import ChameleonVQVAEPreprocessor
from utils import Config
from utils import collate_fn
from utils import setup

START_OF_IMAGE_INDEX = 8197 # <racm3:break>
END_OF_IMAGE_INDEX = 8196 # <eoss>
END_OF_TURN = 8710
PAD_INDEX = 1

def get_args():
    parser = argparse.ArgumentParser(description='Your Chameleon model')
    # model related
    parser.add_argument('--config', type=str, default='./config/basic.yml')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    config_dict = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    config = Config(config_dict)
    
    #--- Set up basic stuffs
    writer, save_location = setup(config)
    # breakpoint()
    processor = ChameleonProcessor.from_pretrained(config.model_id)
    # pretrained_vqvae = ChameleonVQVAEPreprocessor.from_pretrained(config.model_id)
    model = ChameleonForConditionalGeneration.from_pretrained(config.model_id, device_map="auto")#, torch_dtype=torch.float16)
    print(f'Loaded {config.model_id}!')

    # --- Add personalized tokens
    prefix_tokens = [f'<reserved{16301+i}>' for i in range(config.prefix_token)]
    personalized_tokens = [f'<reserved16300>']
    personalized_tokens.extend(prefix_tokens)
    sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}."
    personalized_token_ids = processor.tokenizer.convert_tokens_to_ids(personalized_tokens)

    print(f'Personalized tokens will be: {personalized_tokens}')
    print(f'Personalized token ids will be: {personalized_token_ids}')
    print(f'Personalized prompt: {sks_prompt}')
    # breakpoint()

    # --- Dataloader
    train_dataset = PersonalizedDataset(
            data_root=config.data_root,
            json_file=config.json_file,
            sks_name=config.sks_name,
            personalized_prompt = sks_prompt,
            processor=processor,
            # vqvae=pretrained_vqvae,
            )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1,
        # collate_fn=collate_fn
    )
    dataloader_iter = cycle(train_dataloader)
    # --- Prepare for training
    token_embeds = model.get_input_embeddings().weight.data # this should have shape: torch.Size([65536, 8192]) which is #vocab x token-embed
    orig_embeds_params = model.get_input_embeddings().weight.data.clone()
    orig_lm_params = model.lm_head.weight.data.clone()
    

    if config.whole_model:
        trainable_params = model.model.parameters()
        optimizer = torch.optim.AdamW(
            trainable_params, # for optimize the embeddings and the head
            lr=config.lr,
            betas=(0.9, 0.95),
            weight_decay=0.001,
            eps=1e-08,
        )
    else:
        trainable_params = [model.get_input_embeddings().weight, model.lm_head.weight]
        optimizer = torch.optim.AdamW(
            trainable_params, # for optimize the embeddings and the head
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08,
        )
    if config.resume:
        start_epoch = config.resume_epoch
        try:
            lm_head = torch.load(f'{config.savedir}/{config.exp_name}/{config.sks_name}/{config.resume_epoch}-lmhead.pt', map_location='cuda').to(model.lm_head.weight.data.device)
            lm_head = lm_head.to(model.dtype)
            model.lm_head.weight.data[personalized_token_ids] = lm_head
        except:
            state_dict = torch.load(f'{config.savedir}/{config.exp_name}/{config.sks_name}/{config.resume_epoch}-model.pt')
            model.model.load_state_dict(state_dict)
    else:
        start_epoch = 0

    if config.whole_model:
        model.model.requires_grad_(True)
        model.model.embed_tokens.weight.requires_grad_(True)
        model.model.vqmodel.requires_grad_(False)
        index_no_updates = torch.zeros((len(processor.tokenizer),), dtype=torch.bool)
    else:
        model.model.requires_grad_(False)
        model.model.embed_tokens.weight.requires_grad_(True)
        index_no_updates = torch.ones((len(processor.tokenizer),), dtype=torch.bool)
        index_no_updates[personalized_token_ids] = False
        model.model.resize_token_embeddings(len(processor.tokenizer))

    # --- Start training
    for iteration in tqdm(range(config.iteration)):
        optimizer.zero_grad()
        # check if in labels, there is any start-of-image-tokens
        batch = next(dataloader_iter)
        batch['pixel_values'] = batch['pixel_values'].to(model.dtype)
        for i, item in enumerate(batch['labels']):
            if len(torch.nonzero(batch['labels'][i]==START_OF_IMAGE_INDEX)) != 0:
                soi_index = torch.nonzero(batch['labels'][i]==START_OF_IMAGE_INDEX).item()+1
                eot_index = torch.nonzero(batch['labels'][i]==END_OF_IMAGE_INDEX).item()
                # current_img = batch['pixel_values'][None, i].to(model.dtype)
                image_tokens = model.model.get_image_tokens(pixel_values=batch['pixel_values'][None, i])[0]
                batch['labels'][i, soi_index:eot_index] = image_tokens
        batch = {k: v.to(model.device) for k, v in batch.items()}
        # Forward pass

        output = model(
            input_ids=batch['input_ids'],
            pixel_values=batch['pixel_values'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = output.loss
        loss.backward()
        # print(loss)
        optimizer.step()
        if config.whole_model == False:
            with torch.no_grad():
                model.get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds_params[index_no_updates]
                model.lm_head.weight[index_no_updates] = orig_lm_params[index_no_updates]
        writer.add_scalar('Loss/Loss', loss, iteration)

        if iteration % config.save_every == 0:
            print('Save model at: ', save_location)
            save_path_token = os.path.join(save_location, f'{iteration}-token.pt')
            save_path_lmhead = os.path.join(save_location, f'{iteration}-lmhead.pt')
            torch.save(model.get_input_embeddings().weight.data[personalized_token_ids], save_path_token)
            if config.whole_model:
                torch.save(model.model.state_dict(), os.path.join(save_location, f'{iteration}-model.pt'))
            else:
                torch.save(model.lm_head.weight.data[personalized_token_ids], save_path_lmhead)
            with torch.no_grad():
                print('Generate evaluation images')
                # inputs = processor(sks_prompt + ' A photo of <reserved16300>.', return_tensors="pt").to(model.device)
                inputs = processor(sks_prompt + ' A photo of <reserved16300>.', return_tensors="pt").to(model.device)
                generate_ids = model.generate(**inputs,
                    multimodal_generation_mode="image-only",
                    max_new_tokens=1026,
                    do_sample=True,)
                response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
                pixel_values = model.decode_image_tokens(response_ids[:, 1:-1])
                pixel_values = processor.postprocess_pixel_values(pixel_values)
                image = to_pil_image(pixel_values[0].detach().cpu())
                image.save(os.path.join(save_location, f'{iteration}-test.png'))
                print('Generated images are saved in ', save_location)