# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data
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
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from augments import preproc


def detection_collate(batch):
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return torch.stack(imgs, 0), targets


def AnnotationTransform(target):
    assert isinstance(target, ET.Element)

    WIDER_CLASSES = ('__background__', 'face')
    class_to_ind = dict(zip(WIDER_CLASSES, range(len(WIDER_CLASSES))))

    res = np.empty((0, 5))
    for obj in target.iter('object'):
        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text)
            bndbox.append(cur_pt)
        label_idx = class_to_ind[name]
        bndbox.append(label_idx)
        res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
    return res


class VOCDetection(Dataset):
    def __init__(self, mode='train'):
        self._annopath = f'../data/WIDER_{mode}/annotations/%s'
        self._imgpath = f'../data/WIDER_{mode}/images/%s'
        self.preproc = preproc(1024, (104, 117, 123))
        self.ids = list()
        with open('../data/WIDER_train/img_list.txt', 'r') as f:
            self.ids = [tuple(line.split()) for line in f]

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id[1]).getroot()
        img = cv2.imread(self._imgpath % img_id[0], cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        target = AnnotationTransform(target)
        img, target = self.preproc(img, target)
        return torch.from_numpy(img), target

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    datasets = VOCDetection()
    """
    x:(bs,3,1024,1024)
    y:(bs,num_obj, 5) # (cx,cy,w,h,1) #last idx is the label
    """
    dataloader = DataLoader(datasets, batch_size=30, collate_fn=detection_collate)

    for x, y in dataloader:
        print(x.size())
        print(y)
        break
