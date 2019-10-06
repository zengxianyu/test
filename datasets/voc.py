import os
import numpy as np
from PIL import Image, ImageDraw
import PIL
import torch
import pdb
if __name__ != "__main__":
    from .base_data import _BaseData
else:
    from base_data import _BaseData
    from tqdm import tqdm
import random
import math
import xml.etree.ElementTree as ET


palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


index2color = labelcolormap(21)
index2color = [list(hh) for hh in index2color]
index2name = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
              'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor')


class VOCClass(_BaseData):
    """
    saliency dataset
        return image (0, 1), mask {0,1} is salient object mask
    """
    def __init__(self, path_image, path_ann, size=256, crop=None, rotate=None,  
                 flip=False):
        super(VOCClass, self).__init__(size=size, crop=crop, rotate=rotate, flip=flip, mean=None, std=None)
        name_ann = os.listdir(path_ann)
        list_ann = list(map(lambda x: os.path.join(path_ann, x), name_ann))
        list_image = list(map(lambda x: 
            os.path.join(path_image, ".".join(x.split(".")[:-1])+'.jpg'), name_ann))
        self.list_image = list_image
        self.list_ann = list_ann

    def __len__(self):
        return len(self.list_ann)

    def __getitem__(self, index):
        path_img = self.list_image[index]
        path_ann = self.list_ann[index]
        img = Image.open(path_img).convert("RGB")
        if self.crop:
            img, = self.random_crop(img)
        if self.rotate:
            img, = self.random_rotate(img)
        if self.flip:
            img, = self.random_flip(img)
        img = img.resize(self.size)
        img = np.array(img, dtype=np.float64) / 255.0
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        ann = ET.parse(path_ann)
        classes = [index2name.index(obj.find("name").text) for obj in ann.findall("object")]
        classes = list(set(classes))+[0]
        classes = np.array(classes)[None, ...] # skip background
        rg = np.arange(len(index2name))[..., None]
        onehot = (classes==rg)
        onehot = onehot.sum(1)
        onehot = torch.from_numpy(onehot).float()
        return img, onehot




if __name__ == "__main__":
    dataset = VOCClass('../data/VOC12/VOCdevkit/VOC2012/JPEGImages', 
            '../data/VOC12/VOCdevkit/VOC2012/Annotations', crop=0.9, flip=True)
    for i in range(len(dataset)):
        sb = dataset.__getitem__(i)
        pdb.set_trace()
