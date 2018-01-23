# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     Config
   Description :
   Author :       haxu
   date：          2018/1/22
-------------------------------------------------
   Change Activity:
                   2018/1/22:
-------------------------------------------------
"""
__author__ = 'haxu'


class Config(object):
    data_path = 'data/'
    num_workers = 4
    image_size = 96
    batch_size = 256
    max_epoch = 200
    lr1 = 2e-4
    lr2 = 2e-4
    beta1 = 0.5
    use_gpu = False
    nz = 100
    ngf = 64
    ndf = 64
    save_path = 'imgs/'

    plot_every = 1

    d_every = 1
    g_every = 1
    decay_every = 1

    netd_path = ''
    netg_path = ''

    gen_img = 'result.png'

    gen_num = 64
    gen_search_num = 512
    gen_mean = 0
    gen_std = 1
