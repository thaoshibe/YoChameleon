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
    parser.add_argument('--no_wandb', action='store_true', help='Turn off log to WanDB for debug reason')
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    config_dict = yaml.safe_load(open(args.config, 'r'))
    config = Config(config_dict)
    config.no_wandb = args.no_wandb

    # Call training loop
    trainer = YoChameleonTrainer(config)
    dataloader_iter = get_dataloader_iter(config, trainer.processor)

    trainer.resume_training()
    trainer.configure_model()
    trainer.train(dataloader_iter)
