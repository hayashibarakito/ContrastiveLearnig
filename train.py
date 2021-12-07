import os
import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F

from dataset import *
from loss import *
from model import *

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(model, loader, criterion, optimizer, scheduler, epoch):
    
    losses = AverageMeter()
    model.train()
    for batch_idx, batch in enumerate(loader):
        # 参考
        # https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py#L209
        images = torch.cat([batch["image"][0], batch["image"][1]], dim=0)
        images = images.to(DEVICE)
        target = batch["target"].to(DEVICE)
        bsz = target.shape[0]

        optimizer.zero_grad()

        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, target)

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), bsz)
        
    return losses.avg

if __name__ == '__main__':

    #パラメーター
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 100
    BATCH = 64

    dic = make_datapath_dic("train")
    transform = ImageTransform(300)
    train_dataset = SupConDataset(dic, transform=transform, phase="train")

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    
    model = SupConModel(base_name="resnet18",pretrained=True,feat_dim=128)
    model = model.to(DEVICE)

    criterion = SupConLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-03, weight_decay=1.0e-02)
    scheduler = lr_scheduler.OneCycleLR(optimizer, epochs=EPOCHS, steps_per_epoch=len(train_loader),
                                        max_lr=1.0e-3, pct_start=0.1, anneal_strategy='cos',
                                        div_factor=1.0e+3, final_div_factor=1.0e+3
                                        )
    x_epoch_data = []
    y_train_loss_data = []
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, epoch)
        x_epoch_data.append(epoch)
        print("epoch:",epoch,"train_loss",train_loss)

    model_name = str(y_train_loss_data[-1]) + '.pth'
    torch.save(model.state_dict(), model_name)
    print(f'Saved model as {model_name}')
