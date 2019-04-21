# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main
   Description :
   Author :       haxu
   date：          2019/4/12
-------------------------------------------------
   Change Activity:
                   2019/4/12:
-------------------------------------------------
"""
__author__ = 'haxu'

import torch
from torch import nn

if __name__ == '__main__':
    rnn = nn.GRU(input_size=300, hidden_size=100)
    x = torch.randn(3, 8, 300)

    a, b = rnn(x)

    batch_size, seq_size, feat_size = a.shape
    a = a.contiguous().view(batch_size * seq_size, feat_size)

    print(a.size())
