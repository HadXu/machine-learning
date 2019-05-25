# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main
   Description :
   Author :       haxu
   date：          2019-05-25
-------------------------------------------------
   Change Activity:
                   2019-05-25:
-------------------------------------------------
"""
__author__ = 'haxu'

from operator import __or__
from itertools import cycle
from torch.nn import functional as F
from functools import reduce
import numpy as np
import torch
from argparse import Namespace
from torch import nn
from torch.optim import SGD
from torchvision.datasets import CIFAR10
from torch.nn.utils import weight_norm
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.backends.cudnn as cudnn
from tqdm import tqdm

device = None

np.random.seed(42)
torch.manual_seed(42)
torch.random.manual_seed(42)
cudnn.benchmark = True


def load_data(labels_per_class=100, valid_labels_per_class=500, batch_size=100, workers=1):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=2),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_data = CIFAR10('../data/', train=True, transform=train_transform, download=True)
    test_data = CIFAR10('../data/', train=False, transform=test_transform, download=True)
    num_classes = 10

    n_labels = num_classes

    def get_sampler(labels, n=None, n_valid=None):
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
        np.random.shuffle(indices)

        indices_valid = np.hstack(
            [list(filter(lambda idx: labels[idx] == i, indices))[:n_valid] for i in range(n_labels)])
        indices_train = np.hstack(
            [list(filter(lambda idx: labels[idx] == i, indices))[n_valid:n_valid + n] for i in range(n_labels)])
        indices_unlabelled = np.hstack(
            [list(filter(lambda idx: labels[idx] == i, indices))[n_valid:] for i in range(n_labels)])
        indices_train = torch.from_numpy(indices_train)
        indices_valid = torch.from_numpy(indices_valid)
        indices_unlabelled = torch.from_numpy(indices_unlabelled)
        sampler_train = SubsetRandomSampler(indices_train)
        sampler_valid = SubsetRandomSampler(indices_valid)
        sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
        return sampler_train, sampler_valid, sampler_unlabelled

    train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.targets, labels_per_class,
                                                                   valid_labels_per_class)

    labelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=workers, pin_memory=True)
    validation = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler,
                                             num_workers=workers, pin_memory=True)
    unlabelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=unlabelled_sampler,
                                             num_workers=workers, pin_memory=True)
    test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers,
                                       pin_memory=True)

    return labelled, validation, unlabelled, test, num_classes


def mixup_data(x, y, alpha):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class ICTModel(nn.Module):
    def __init__(self, num_classes=10, dropout=0.0):
        super(ICTModel, self).__init__()

        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1 = nn.Dropout(dropout)

        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2 = nn.Dropout(dropout)

        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)

        self.fc1 = weight_norm(nn.Linear(128, num_classes))

    def forward(self, x):
        out = x
        ## layer 1-a###
        out = self.conv1a(out)
        out = self.bn1a(out)
        out = self.activation(out)

        ## layer 1-b###
        out = self.conv1b(out)
        out = self.bn1b(out)
        out = self.activation(out)

        ## layer 1-c###
        out = self.conv1c(out)
        out = self.bn1c(out)
        out = self.activation(out)

        out = self.mp1(out)
        out = self.drop1(out)

        ## layer 2-a###
        out = self.conv2a(out)
        out = self.bn2a(out)
        out = self.activation(out)

        ## layer 2-b###
        out = self.conv2b(out)
        out = self.bn2b(out)
        out = self.activation(out)

        ## layer 2-c###
        out = self.conv2c(out)
        out = self.bn2c(out)
        out = self.activation(out)

        out = self.mp2(out)
        out = self.drop2(out)

        ## layer 3-a###
        out = self.conv3a(out)
        out = self.bn3a(out)
        out = self.activation(out)

        ## layer 3-b###
        out = self.conv3b(out)
        out = self.bn3b(out)
        out = self.activation(out)

        ## layer 3-c###
        out = self.conv3c(out)
        out = self.bn3c(out)
        out = self.activation(out)

        out = self.ap3(out)

        out = out.view(-1, 128)
        out = self.fc1(out)
        return out


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(model, ema_model, train_loader, unlabeled_loader, optimizer, class_loss_fn, mixup_consistency_loss):
    model.train()
    ema_model.train()

    losses = []

    for (input, target), (u1, _), (u2, _) in tqdm(zip(cycle(train_loader), unlabeled_loader, unlabeled_loader)):
        if input.shape[0] != u1.shape[0]:
            bt_size = np.minimum(input.shape[0], u1.shape[0])
            input = input[0:bt_size]
            target = target[0:bt_size]
            u1 = u1[0:bt_size]
            u2 = u2[0:bt_size]

        inv_idx = torch.arange(u2.size(0) - 1, -1, -1).long()
        u2 = u2.index_select(0, inv_idx)
        u2 = u2[inv_idx]

        input, target, u1, u2 = input.to(device), target.to(device), u1.to(device), u2.to(device)

        # 分类
        output = model(input)
        class_loss = class_loss_fn(output, target)

        # mixup
        lam = np.random.beta(1, 1)
        lam = torch.tensor(lam).to(device)
        mixup_xy = lam * u1 + (1. - lam) * u2

        mixup_output = model(mixup_xy)

        ema_output1 = ema_model(u1)
        ema_output2 = ema_model(u2)

        ema_output = lam * ema_output1 + (1. - lam) * ema_output2

        mixup_loss = mixup_consistency_loss(ema_output, mixup_output)

        loss = class_loss + mixup_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    minibatch_size = len(target)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / minibatch_size))
    return res


def val(model, loader):
    prec1s, prec5s = [], []
    for i, (input, target) in enumerate(loader):
        input, target = input.to(device), target.to(device)
        output = model(input)
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        prec1s.append(prec1)
        prec5s.append(prec5)

    return np.mean(prec1s), np.mean(prec5s)


def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes


def main(args):
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data(labels_per_class=100,
                                                                                    valid_labels_per_class=100,
                                                                                    batch_size=args.bs, workers=32)

    model = ICTModel().to(device)
    ema_model = ICTModel().to(device)
    for param in ema_model.parameters():
        param.detach_()

    optim = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    loss_class_fn = nn.CrossEntropyLoss()

    for e in range(10):
        print(f'epoch {e}........')

        train_loss = train(model, ema_model, trainloader, unlabelledloader, optim, loss_class_fn, softmax_mse_loss)
        update_ema_variables(model, ema_model, 0.999, e + 1)

        print(f'train loss {train_loss}')

        prec1, prec5 = val(model, validloader)

        print(f'val acc1{prec1} acc5{prec5}...')


if __name__ == '__main__':
    device = torch.device('cpu')
    args = Namespace(
        bs=32,
        lr=0.01
    )
    main(args)
