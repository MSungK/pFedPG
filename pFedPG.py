import os
from time import sleep
from random import randint
import random
import torch
import numpy as np

from custom_src.utils.file_io import PathManager
from custom_src.models.vit_models import ViT
from custom_utils.launch import default_argument_parser, logging_train_setup
from custom_src.configs.config import get_cfg
from custom_src.models.build_model import build_model
from custom_src.engine.trainer import Trainer
from server import PromptGenerator
from custom_utils.loader import prepare_data


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.DIST_INIT_PATH = 'env://'
    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    output_folder = os.path.join(
    cfg.DATA.NAME, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}")
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
    train_loaders, val_loaders, test_loaders = prepare_data(cfg) # B 3 224 224 
    # for loader in train_loaders:
    #     print(len(loader.dataset))
    # exit()
    num_clients = len(train_loaders)
    # Server 
    server = PromptGenerator(num_clients=num_clients, config=cfg).to(device)
    # Each Client have vpt
    clients = [build_model(cfg, device) for _ in range(num_clients)] # Freezed except prompt, head
    # Setting for Train
    clients = [Trainer(cfg=cfg, model=clients[i], device=device, index=i) for i in range(num_clients)]
    
    server_epochs = 5
    server_optimizer = torch.optim.AdamW(params=server.parameters(),lr=1e-3, weight_decay=1e-4)

    for server_epoch in range(server_epochs):
        server_optimizer.zero_grad()
        server.train()
        client_prompts = server.forward() # 1 num_clients embedded_dims

        client_deltas = list()

        # SOLVER.TOTAL_EPOCH : communication round
        for client_index, client in enumerate(clients):
            client.initialize_prompt(client_prompts[0, client_index*10:(client_index+1)*10, :]) # Server gives client-specific client client
            print(f'Start client {client_index} training')
            client.train_classifier(train_loaders[client_index],val_loaders[client_index],test_loaders[client_index])
            client_deltas.append(client.calculate_delta_prompt())
        
        print('***' * 10)
        print(f'Server Epoch: {server_epoch + 1}')
        client_deltas = torch.cat(client_deltas, dim = 1)
        assert client_prompts.shape == client_deltas.shape
        client_prompts.backward(client_deltas) # upstream gradient: client_deltas
        server_optimizer.step()


def main(args):
    cfg = setup(args)
    train(cfg, args)
    

if __name__ == '__main__':
    args = default_argument_parser().parse_args()   
    main(args)