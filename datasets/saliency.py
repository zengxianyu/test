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


class SaliencyMask(_BaseData):
    """
    saliency dataset
        return image (0, 1), mask {0,1} is salient object mask
    """
    def __init__(self, path_image, path_mask, size=256, crop=None, rotate=None,  
                 flip=False):
        super(SaliencyMask, self).__init__(size=size, crop=crop, rotate=rotate, flip=flip, mean=None, std=None)
        name_mask = os.listdir(path_mask)
        list_mask = list(map(lambda x: os.path.join(path_mask, x), name_mask))
        list_image = list(map(lambda x: 
            os.path.join(path_image, ".".join(x.split(".")[:-1])+'.jpg'), name_mask))
        self.list_image = list_image
        self.list_mask = list_mask

    def __len__(self):
        return len(self.list_mask)

    def __getitem__(self, index):
        path_img = self.list_image[index]
        path_mask = self.list_mask[index]
        img = Image.open(path_img).convert("RGB")
        msk = Image.open(path_mask).convert("L")
        if self.crop:
            img, msk = self.random_crop(img, msk)
        if self.rotate:
            img, msk = self.random_rotate(img, msk)
        if self.flip:
            img, msk = self.random_flip(img, msk)
        img = img.resize(self.size)
        msk = msk.resize(self.size)
        img = np.array(img, dtype=np.float64) / 255.0
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        msk = np.array(msk)
        msk = (msk>0).astype(np.float)
        msk = torch.from_numpy(msk).float()
        return img, msk




if __name__ == "__main__":
    dataset = SaliencyMask('../data/ECSSD/images', '../data/ECSSD/masks', crop=0.9, flip=True)
    sb = dataset.__getitem__(0)
    pdb.set_trace()
