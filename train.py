import argparse
import warnings

import yaml
from PIL import Image
from tqdm import tqdm
from utils import Config, get_dataloader_iter, get_eval_dataloader
from yochameleon import YoChameleonTrainer
from yoemu3 import YoEmu3Trainer

warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description='Your Chameleon model')
    # model related
    parser.add_argument('--config', type=str, default='./config/basic.yml')
    parser.add_argument('--no_wandb', action='store_true', help='Turn off log to WanDB for debug reason', default=False)
    parser.add_argument('--sks_name', type=str, help='Override sks_name', default=None)
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    config_dict = yaml.safe_load(open(args.config, 'r'))
    config = Config(config_dict)
    config.no_wandb = args.no_wandb

    # Load the universal config only, override sks name with actual sks_name
    if args.sks_name is not None:
        config.sks_name = args.sks_name
        config.json_file = [x.replace('SKS_NAME', config.sks_name) for x in config.json_file]

    # call training loop
    if config.model_id == 'leloy/Anole-7b-v0.1-hf':
        trainer = YoChameleonTrainer(config)
    elif config.model_id == 'Emu3-community/Emu3-Gen-hf':
        trainer = YoEmu3Trainer(config)
    else:
        raise ValueError(f"Model ID {config.model_id} is not supported yet~")

    personalized_prompt = trainer.get_personalized_prompt()
    print(f"Personalized prompt: {personalized_prompt}")
    
    train_dataloader = get_dataloader_iter(
        config,
        trainer.processor,
        personalized_prompt=personalized_prompt
        )

    trainer.resume_training()
    trainer.configure_model() # this step will set up optimization
    if config.self_prompting:
        understanding_prompt = trainer.get_understanding_prompt()
    else:
        understanding_prompt = None
    recognition_dataloader_train = get_eval_dataloader(
        config,
        trainer.processor,
        image_folder=config.eval['recognition_path_train'],
        personalized_prompt=personalized_prompt,
        understanding_prompt=understanding_prompt
    )
    recognition_dataloader_test = get_eval_dataloader(
        config,
        trainer.processor,
        image_folder=config.eval['recognition_path_test'],
        personalized_prompt=personalized_prompt,
        understanding_prompt=understanding_prompt
    )
    if config.epoch > 0: #If you want to train with epoch... Fine, here you go
        config.iteration = config.epoch
        if config.task_disjoin:
            print('\n\n\n   Hello, this script will train with task disjoin !!!\n\n\n')
            trainer.train_epoch_disjoin(
                train_dataloader,
                recognition_dataloader_train,
                recognition_dataloader_test)
        else:
            trainer.train_epoch(
                train_dataloader,
                recognition_dataloader_train,
                recognition_dataloader_test
                )

        # -- Thao: Maybe we should move this to the finetuning stage for all
        if config.finetune['finetune']:
            config.finetune['finetune_iteration'] = config.finetune['finetune_epoch']
            positive_only_dataloader = get_dataloader_iter(config, trainer.processor, only_positive=True)
            trainer.finetune_epoch(positive_only_dataloader)

    else: # This support train with iteration
        print('Hello, train with iteration')
        trainer.train(train_dataloader)
        if config.finetune['finetune']:
            positive_only_dataloader = get_dataloader_iter(config, trainer.processor, only_positive=True)
            trainer.finetune(positive_only_dataloader)
    # trainer.train(train_dataloader)
