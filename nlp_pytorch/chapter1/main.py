# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main
   Description :
   Author :       haxu
   date：          2019/4/10
-------------------------------------------------
   Change Activity:
                   2019/4/10:
-------------------------------------------------
"""
__author__ = 'haxu'

import torch

if __name__ == '__main__':
    a = torch.rand(3, 4, 5)
    b = torch.rand(5, 4)
    b = b.unsqueeze(0).expand(a.size(0), *b.size())

    print(torch.bmm(a, b).size())
