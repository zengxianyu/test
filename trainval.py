import torch
import torch.nn.functional as F
import torchvision
import os
from networks import MTFCN
from datasets import SaliencyMask, VOCClass
from datasets.voc import index2color
from logger import Logger
import pdb

path_sal_image = "data/DUTS-TR/DUTS-TR-Image"
path_sal_ann = "data/DUTS-TR/DUTS-TR-Mask"
path_voc_image = "data/VOC12/VOCdevkit/VOC2012/JPEGImages"
path_voc_ann = "data/VOC12/VOCdevkit/VOC2012/Annotations"
bsize=64
image_size =128
train_iters = 100000

log_writer = Logger("logs", clear=True)

def train():
    train_loader_sal = torch.utils.data.DataLoader(
    SaliencyMask(path_sal_image, path_sal_ann, size=image_size,
                 crop=0.9, flip=True),
    batch_size=bsize, shuffle=True, num_workers=8, pin_memory=True,drop_last=True)
    train_loader_voc = torch.utils.data.DataLoader(
    VOCClass(path_voc_image, path_voc_ann, size=image_size,
                 crop=0.9, flip=True),
    batch_size=bsize, shuffle=True, num_workers=9, pin_memory=True,drop_last=True)

    train_iter_sal = iter(train_loader_sal)
    it_sal = 0
    train_iter_voc = iter(train_loader_voc)
    it_voc = 0

    net = MTFCN(c_output=21)
    net = net.cuda()

    optimizer = torch.optim.Adam([
                {'params': net.parameters(), 'lr': 1e-4},
                    ])

    for i in range(train_iters):
        # read voc images and one-hot class labels
        if it_voc >= len(train_loader_voc):
            train_iter_voc = iter(train_loader_voc)
            it_voc = 0
        image_voc, class_voc = train_iter_voc.next()
        it_voc += 1
        # read saliency image and ground-truth masks
        if it_sal >= len(train_loader_sal):
            train_iter_sal = iter(train_loader_sal)
            it_sal = 0
        image_sal, mask_sal = train_iter_sal.next()
        it_sal += 1

        optimizer.zero_grad()

        pred, _, pred_cls, pred_cls0 = net(image_voc.cuda())
        ##############
        ##############

        _, pred_sal, _, _  = net(image_sal.cuda())
        ##############
        ##############
        
        loss_sal.backward()

        optimizer.step()

        if i % 50 == 0:
            # visualize training samples
            _, pred_index = pred.detach().cpu().max(1)
            _b = pred_index.size(0)
            # colorize class label maps for vis
            msk = torch.Tensor(_b, image_size, image_size, 3)
            for j, color in enumerate(index2color):
                if (pred_index == j).sum() > 0:
                    msk[pred_index == j, :] = torch.Tensor(color)
            msk = torch.transpose(msk, 2, 3)
            msk = torch.transpose(msk, 1, 2)
            log_writer.add_scalar("class loss", loss_voc.item(), i)
            log_writer.add_scalar("sal loss", loss_sal.item(), i)
            log_writer.add_image("voc image", torchvision.utils.make_grid(image_voc[:4]), i)
            log_writer.add_image('prediction', torchvision.utils.make_grid(msk[:4] / 255), i)

            log_writer.add_image("saliency image", torchvision.utils.make_grid(image_sal[:4]), i)
            log_writer.add_image('saliency', 
                    torchvision.utils.make_grid(pred_sal[:4, 1:].expand(-1, 3, -1,-1)).detach(), 
                    i)
            log_writer.write_html()


if __name__ == "__main__":
    train()
