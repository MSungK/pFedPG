#!/usr/bin/env python3
"""
a trainer class
"""
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import os

from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import logging
from ..utils.train_utils import AverageMeter, gpu_mem_usage

logger = logging.get_logger("visual_prompt")


class Trainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """
    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        device: torch.device,
        index : int,
        client_save_path : str,
        use_val : bool
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        self.index = index # Identify which client
        self.train_loss_list = list()
        self.val_loss_list = list()
        self.val_acc_list = list()
        self.max_val_loss = 1e5
        self.best_metric = dict()
        self.best_metric['acc'] = 0
        self.client_save_path = client_save_path
        self.use_val = use_val
        os.makedirs(self.client_save_path, exist_ok=True)

        # solver related
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if key not in ["head.last_layer.bias",  "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)

        self.cpu_device = torch.device("cpu")

    def forward_one_batch(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            # if loss == float('inf'):
            #     logger.info(
            #         "encountered infinite loss, skip gradient updating for this batch!"
            #     )
            #     return -1, -1
            # elif torch.isnan(loss).any():
            #     logger.info(
            #         "encountered nan loss, skip gradient updating for this batch!"
            #     )
            #     return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, outputs

    # def get_input(self, data):
    #     if not isinstance(data["image"], torch.Tensor):
    #         for k, v in data.items():
    #             data[k] = torch.from_numpy(v)

    #     inputs = data["image"].float()
    #     labels = data["label"]
    #     return inputs, labels

    def train_classifier(self, train_loader, server_epoch, val_loader=None,):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        self.model.eval()

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader.dataset)
        best_epoch = -1
        # log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        # batch_time = AverageMeter('Time', ':6.3f')
        # data_time = AverageMeter('Data', ':6.3f')
        # TODO, Consider the F.crossentropy cls weight 
        self.cls_weights = None
        # print(self.cfg)
        # exit()
        # self.cls_weights = train_loader.dataset.get_class_weights(
        #     self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        print('==='*10)
        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            # batch_time.reset()
            # data_time.reset()

            lr = self.scheduler.get_last_lr()
            print(f'Training {epoch+1} / {total_epoch} epoch, with client {self.index}')

            # Enable training mode
            self.model.train()

            for images, labels in tqdm(train_loader):

                train_loss, _ = self.forward_one_batch(images, labels, is_train=True)
                losses.update(train_loss.item(), images.shape[0])
            
            print(f'Training {epoch+1} loss: {losses.avg:3f}')
            
            self.train_loss_list.append(losses.avg)

            self.scheduler.step()
            # Enable eval mode
            self.model.eval()

            # eval at each epoch for single gpu training
            if self.use_val:
                val_loss, val_acc = self.eval_classifier(val_loader, test=False)

                if val_acc >= self.best_metric['acc']:
                    best_epoch = epoch + 1
                    self.best_metric['acc'] = val_acc
                    self.best_metric['epoch'] = server_epoch

                    print(f'Best epoch {best_epoch}: best metric: {self.best_metric["acc"]:.3f}')

                    patience = 0
                    # if self.cfg.MODEL.SAVE_CKPT:
                    #     out_path = os.path.join(
                    #         self.cfg.OUTPUT_DIR, f"best_for_client{self.index}.pth")
                    #     torch.save(self.model.parameters(), out_path)
                    self.save_client_param()
                else:
                    patience += 1
                
                if patience >= self.cfg.SOLVER.PATIENCE:
                    print("No improvement. Breaking out of loop. for client {self.index}")
                    break

        # save the last checkpoints
        # if self.cfg.MODEL.SAVE_CKPT:
        #     Checkpointer(
        #         self.model,
        #         save_dir=self.cfg.OUTPUT_DIR,
        #         save_to_disk=True
        #     ).save("last_model")

    @torch.no_grad()
    def save_prompt(self):
        # only save the prompt embed if below conditions are satisfied
        if self.cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH:
            if self.cfg.MODEL.TYPE == "vit" and "prompt" in self.cfg.MODEL.TRANSFER_TYPE:
                prompt_embds = self.model.enc.transformer.prompt_embeddings.cpu().numpy()
                out = {"shallow_prompt": prompt_embds}
                if self.cfg.MODEL.PROMPT.DEEP:
                    deep_embds = self.model.enc.transformer.deep_prompt_embeddings.cpu().numpy()
                    out["deep_prompt"] = deep_embds
                torch.save(out, os.path.join(
                    self.cfg.OUTPUT_DIR, f"prompt_{self.index}.pth"))

    @torch.no_grad()
    def eval_classifier(self, data_loader, test=False):
        """evaluate classifier"""
        # batch_time = AverageMeter('Time', ':6.3f')
        # data_time = AverageMeter('Data', ':6.3f')
        if test and self.use_val:
            self.model.load_state_dict(torch.load(os.path.join(self.client_save_path, 'best_val.pth')))

        losses = AverageMeter('Loss', ':.4e')

        total = len(data_loader.dataset)

        # initialize features and target
        total_logits = []
        total_targets = []

        acc = 0
        total = 0

        for images, labels in data_loader:

            loss, outputs = self.forward_one_batch(images, labels, False)
            
            losses.update(loss.item(), images.shape[0])
            acc += self.calculate_accuracy(logits=outputs, target=labels)
            total += images.shape[0]
        
        val_acc = acc / total

        if test:
            return val_acc

        self.val_acc_list.append(val_acc)
        self.val_loss_list.append(losses.avg)
        
        # # save the probs and targets
        # if save and self.cfg.MODEL.SAVE_CKPT:
        #     out = {"targets": total_targets}
        #     out_path = os.path.join(
        #         self.cfg.OUTPUT_DIR, f"best_for_client{self.index}.pth")
        #     torch.save(out, out_path)
            
        return self.val_loss_list[-1], self.val_acc_list[-1]
    
    @torch.no_grad()
    def calculate_accuracy(self, logits : torch.tensor, target : torch.tensor):
        target = target.cpu()
        logits = logits.cpu()
        logits = torch.argmax(logits, dim=1)
        return torch.sum(logits == target).item()
    
    def initialize_prompt(self, client_specific_prompt : torch.tensor):
        cnt = 0
        self.initial_prompt = None
        client_specific_prompt = torch.unsqueeze(client_specific_prompt, dim=0)

        for key, param in self.model.named_parameters():
            if 'prompt' in key:
                assert param.shape == client_specific_prompt.shape
                param = client_specific_prompt
                self.initial_prompt = param.clone()
                print('~~~'*10)
                print(f'Initialize client {self.index} prompt with server')
                cnt+=1
        assert cnt == 1

    def calculate_delta_prompt(self):
        cnt = 0
        delta = None
        
        for key, param in self.model.named_parameters():
            if 'prompt' in key:
                delta = param - self.initial_prompt
                cnt += 1
        assert cnt == 1
        
        return delta
    
    def save_client_param(self):
        torch.save(self.model.state_dict(), os.path.join(self.client_save_path, 'best_val.pth'))
        