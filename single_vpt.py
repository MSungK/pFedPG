import os
from time import sleep
from random import randint
import random
import torch
import numpy as np
from matplotlib import pyplot as plt

from custom_src.utils.file_io import PathManager
from custom_src.models.vit_models import ViT
from custom_utils.launch import default_argument_parser
from custom_src.configs.config import get_cfg
from custom_src.models.build_model import build_model
from custom_src.engine.trainer import Trainer
from server import PromptGenerator
from custom_utils.loader import prepare_caltech, prepare_domainnet


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.DIST_INIT_PATH = 'env://'
    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    output_folder = os.path.join(
    cfg.DATA.NAME, f"seed_{cfg.SEED}_client_lr{lr}_wd{wd}_server_lr{args.lr}_wd{args.weight_decay}")
    count = 1
    while count <= cfg.RUN_N_TIMES:
        output_path = os.path.join(output_dir, output_folder, f"run{count}")
        # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
        sleep(randint(3, 30))
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)
            cfg.OUTPUT_DIR = output_path
            break
        else:
            count += 1
    if count > cfg.RUN_N_TIMES:
        raise ValueError(
            f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")
    cfg.freeze()
    return cfg


def train(cfg, args):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(0)
    assert torch.cuda.is_available()
    device = f'cuda:{args.device}'
    # DataLoader
    assert cfg.DATA.NAME in ["OfficeCaltech10", "DomainNet10"]
    if cfg.DATA.NAME == "OfficeCaltech10":
        site, train_loaders, val_loaders, test_loaders = prepare_caltech(cfg) # B 3 224 224 
    elif cfg.DATA.NAME == "DomainNet10":
        site, train_loaders, val_loaders, test_loaders = prepare_domainnet(cfg) # B 3 224 224 
    
    # exit()
    num_clients = len(train_loaders)
    # Each Client have vpt
    clients = [build_model(cfg, device) for _ in range(num_clients)] # Freezed except prompt, head
    # Setting for Train
    clients = [Trainer(cfg=cfg, model=clients[i], device=device, 
                       index=i, client_save_path=os.path.join(cfg.OUTPUT_DIR, f'client_{site[i]}'))
                       for i in range(num_clients)]
    
    server_epochs = args.server_epoch

    for server_epoch in range(server_epochs):
        print('$$$'*10)
        print(f'Start server {server_epoch+1} / {server_epochs} training')

        for client_index, client in enumerate(clients):
            print(f'Start client {site[client_index]} training')
            client.train_classifier(train_loader=train_loaders[client_index],
                                    val_loader=val_loaders[client_index],
                                    server_epoch=server_epoch+1)
        
    print('END' * 10)
    print('All Training is ended')
    f = open(os.path.join(cfg.OUTPUT_DIR, 'val_test_acc.txt'), 'w')

    for i, client in enumerate(clients):
        test_acc = client.eval_classifier(test_loaders[i], test=True)
        plt.subplot(2, len(clients)//2, i+1)
        plt.plot(range(len(client.train_loss_list)), client.train_loss_list, label='train')
        plt.plot(range(len(client.val_loss_list)), client.val_loss_list, label='valid')
        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.title(f'{site[i]}')
        plt.ylim([0,6])
        f.write(f'client_{site[i]} val: {client.best_metric["acc"]} \n')
        f.write(f'client_{site[i]} best val at epoch: {client.best_metric["epoch"]} \n')
        f.write(f'client_{site[i]} test: {test_acc} \n')
        f.write('==='*10 + '\n')
        f.write(f'Train ratio: {cfg.DATA.TRAIN_RATIO}')
        print(f'client_{site[i]}: {test_acc}')
    
    plt.savefig(os.path.join(cfg.OUTPUT_DIR,'loss.png'))
    f.write(f'seed: {cfg.SEED} \n')
    f.close()
    

def main(args):
    cfg = setup(args)
    train(cfg, args)
    

if __name__ == '__main__':
    args = default_argument_parser().parse_args()  
    main(args)