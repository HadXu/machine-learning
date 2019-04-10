# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     utils
   Description :
   Author :       haxu
   date：          2019/3/29
-------------------------------------------------
   Change Activity:
                   2019/3/29:
-------------------------------------------------
"""
__author__ = 'haxu'

import torch
from itertools import product as product
import numpy as np

cfg = {
    'name': 'FaceBoxes',
    'feature_maps': [[32, 32], [16, 16], [8, 8]],
    'min_dim': 1024,
    'steps': [32, 64, 128],
    'min_sizes': [[32, 64, 128], [256], [512]],
    'loc_weight': 2.0,
    'gpu_train': True,
}


def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


class PriorBox(object):
    def __init__(self, cfg=cfg, box_dimension=None, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        if phase == 'train':
            self.image_size = (cfg['min_dim'], cfg['min_dim'])
            self.feature_maps = cfg['feature_maps']
        elif phase == 'test':
            self.feature_maps = box_dimension.cpu().numpy().astype(np.int)
            self.image_size = image_size

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if min_size == 32:
                        dense_cx = [x * self.steps[k] / self.image_size[1] for x in
                                    [j + 0, j + 0.25, j + 0.5, j + 0.75]]
                        dense_cy = [y * self.steps[k] / self.image_size[0] for y in
                                    [i + 0, i + 0.25, i + 0.5, i + 0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            mean += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0, j + 0.5]]
                        dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0, i + 0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            mean += [cx, cy, s_kx, s_ky]
                    else:
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        mean += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        return output


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    prior = PriorBox()

    print(prior.forward().size())
