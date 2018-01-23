# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main
   Description :
   Author :       haxu
   date：          2018/1/22
-------------------------------------------------
   Change Activity:
                   2018/1/22:
-------------------------------------------------
"""
__author__ = 'haxu'

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import utils

from models import NetD, NetG
from Config import Config

from tensorboardX import SummaryWriter

writer = SummaryWriter()

opt = Config()

transform = transforms.Compose([
    transforms.Scale(opt.image_size),
    transforms.CenterCrop(opt.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageFolder(opt.data_path, transform=transform)

dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, drop_last=True)

netd = NetD(opt)
netg = NetG(opt)

if opt.netd_path:
    netd.load_state_dict(torch.load(opt.netd_path, map_location=lambda storage, loc: storage))
if opt.netg_path:
    netg.load_state_dict(torch.load(opt.netg_path, map_location=lambda storage, loc: storage))

optimizer_g = Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
optimizer_d = Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))

criterion = nn.BCELoss()

true_labels = Variable(torch.ones(opt.batch_size))
fake_labels = Variable(torch.zeros(opt.batch_size))
fix_noises = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1))
noises = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1))

if opt.use_gpu:
    netd.cuda()
    netg.cuda()
    criterion.cuda()
    true_labels, fake_labels = true_labels.cuda(), fake_labels.cuda()
    fix_noises, noises = fix_noises.cuda(), noises.cuda()

for epoch in range(opt.max_epoch):
    for ii, (img, _) in enumerate(dataloader):
        real_img = Variable(img)

        if opt.use_gpu:
            real_img = real_img.cuda()

        if (ii + 1) % opt.d_every == 0:
            optimizer_d.zero_grad()
            output = netd(real_img)
            error_d_real = criterion(output, true_labels)
            error_d_real.backward()
            noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))

            fake_img = netg(noises).detach()
            fake_output = netd(fake_img)
            error_d_fake = criterion(fake_output, fake_labels)
            error_d_fake.backward()

            error = error_d_real + error_d_fake

            print('error_d:', error.data[0])
            writer.add_scalar('data/error_d', error_d_fake.data[0], ii)

            optimizer_d.step()

        if (ii + 1) % opt.g_every == 0:
            optimizer_g.zero_grad()

            noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
            fake_img = netg(noises)
            fake_output = netd(fake_img)
            error_g = criterion(fake_output, true_labels)

            print('error_g:,', error_g.data[0])
            writer.add_scalar('data/error_g', error_g.data[0], ii)

            error_g.backward()
            optimizer_g.step()

        if (ii + 1) % opt.plot_every == 0:
            fix_fake_imgs = netg(fix_noises)

            fake = fix_fake_imgs[:64] * 0.5 + 0.5
            real = real_img[:64] * 0.5 + 0.5

            writer.add_image('image/fake_Image', fake, ii)
            writer.add_image('image/real_Image', real, ii)

            print('epoch[{}:{}],ii[{}:{}]'.format(epoch, opt.max_epoch, ii, len(dataloader)))

        if (epoch + 1) % opt.decay_every == 0:
            utils.save_image(fix_fake_imgs.data[:64], '%s/%s.png' % (opt.save_path, epoch), normalize=True,
                             range=(-1, 1))
            torch.save(netd.state_dict(), 'checkpoints/netd_%s.pth' % epoch)
            torch.save(netg.state_dict(), 'checkpoints/netg_%s.pth' % epoch)
            optimizer_g = torch.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
            optimizer_d = torch.optim.Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
