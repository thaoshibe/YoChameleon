import argparse
import yaml

from PIL import Image
from tqdm import tqdm
from utils import Config

from utils import get_dataloader_iter
from yochameleon import YoChameleonTrainer

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
    trainer = YoChameleonTrainer(config)
    personalized_prompt = trainer.get_personalized_prompt()
    print(f"Personalized prompt: {personalized_prompt}")
    
    train_dataloader = get_dataloader_iter(
        config,
        trainer.processor,
        personalized_prompt=personalized_prompt
        )
    trainer.resume_training()
    trainer.configure_model() # this step will set up optimization

    if config.epoch > 0: #If you want to train with epoch... Fine, here you go
        config.iteration = config.epoch

        if hasattr(config, 'task_disjoin'):
            print('\n   Hello, this script will train with task disjoin !!!\n')
            trainer.train_epoch_disjoin(train_dataloader)
        else:
            trainer.train_epoch(train_dataloader)

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
        
    # trainer.test()
