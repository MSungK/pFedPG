import os
from time import sleep
from random import randint
import random
import torch
import numpy as np

from src.utils.file_io import PathManager
from src.models.vit_models import ViT
from custom_utils.launch import default_argument_parser, logging_train_setup
from src.configs.config import get_cfg
from src.models.build_model import build_model
from server import PromptGenerator


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
    model, cur_device = build_model(cfg)
    server = PromptGenerator(num_clients=4, config=cfg)
    print(server)


def main(args):
    cfg = setup(args)
    train(cfg, args)
    

if __name__ == '__main__':
    args = default_argument_parser().parse_args()   
    main(args)