# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     train
   Description :
   Author :       haxu
   date：          2019/3/29
-------------------------------------------------
   Change Activity:
                   2019/3/29:
-------------------------------------------------
"""
__author__ = 'haxu'

import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from models import FaceBoxes
from utils import PriorBox, adjust_learning_rate
from data import VOCDetection, detection_collate
from torch.utils.data import DataLoader
from losses import MultiBoxLoss
from tqdm import tqdm

device = torch.device('cpu')
cudnn.benchmark = True

net = FaceBoxes(phase='train').to(device)
optimizer = SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
criterion = MultiBoxLoss(device, 0.35, True, 0, True, 7, 0.35, False)


def train(net):
    net.train()
    priorbox = PriorBox()
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.to(device)

    dataloader = DataLoader(VOCDetection(), batch_size=2, collate_fn=detection_collate, num_workers=12)

    for epoch in range(1000):
        loss_ls, loss_cs = [], []
        load_t0 = time.time()
        if epoch > 500:
            adjust_learning_rate(optimizer, 1e-4)

        for images, targets in dataloader:
            images = images.to(device)
            targets = [anno.to(device) for anno in targets]
            out = net(images)
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, priors, targets)

            loss = 2 * loss_l + loss_c
            loss.backward()
            optimizer.step()
            loss_cs.append(loss_c.item())
            loss_ls.append(loss_l.item())
        load_t1 = time.time()

        print(f'{np.mean(loss_cs)}, {np.mean(loss_ls)} time:{load_t1-load_t0}')
        torch.save(net.state_dict(), 'Final_FaceBoxes.pth')


def test(net):
    net.eval()


def main():
    train(net)


if __name__ == '__main__':
    main()
