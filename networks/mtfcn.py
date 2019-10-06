import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd.variable import Variable

from .resnet import resnet50

import numpy as np
import sys
thismodule = sys.modules[__name__]
import pdb

dim_dict = {
    'resnet101': [512, 1024, 2048],
    'resnet152': [512, 1024, 2048],
    'resnet50': [512, 1024, 2048],
    'resnet34': [128, 256, 512],
    'resnet18': [128, 256, 512],
    'densenet121': [256, 512, 1024],
    'densenet161': [384, 1056, 2208],
    'densenet169': [256, 640, 1664],
    'densenet201': [256, 896, 1920],
    'vgg': [256, 512, 512]
}


def proc_resnet(model):
    def hook(module, input, output):
        model.feats[output.device.index] += [output]
    model.layer3[-1].reluo.register_forward_hook(hook)
    model.layer2[-1].reluo.register_forward_hook(hook)
    return model

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class MTFCN(nn.Module):
    def __init__(self, c_output=21, n_hidden=64):
        super(MTFCN, self).__init__()
        dims = dim_dict['resnet50']
        dims = dims[::-1]
        self.cweight = nn.Sequential(
                nn.Linear(dims[0], n_hidden), nn.ReLU(), 
                nn.Linear(n_hidden, c_output), nn.Sigmoid())
        self.preds = nn.ModuleList([nn.Conv2d(d, c_output, kernel_size=1) for d in dims])
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.fill_(0)
        self.feature = resnet50(pretrained=True)
        self.feature.feats = {}
        self.feature = proc_resnet(self.feature)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad=False 

    def forward(self, x):
        self.feature.feats[x.device.index] = []
        x32 = self.feature(x)
        feats = self.feature.feats[x.device.index]
        x8, x16 = feats

        pred = self.preds[0](x32)
        pred_cls0 = F.softmax(pred, 1).mean(3).mean(2)
        pred = F.interpolate(pred, scale_factor=2, mode="bilinear")

        pred = self.preds[1](x16) + pred
        pred = F.interpolate(pred, scale_factor=2, mode="bilinear")

        pred = self.preds[2](x8) + pred
        pred = F.interpolate(pred, scale_factor=8, mode="bilinear")
        pred = F.softmax(pred, 1)
        pred_cls = pred.mean(3).mean(2)

        cweight = self.cweight(
            x32.mean(3).mean(2))[..., None, None]
        pred_sal = (pred*cweight)[:, 1:].sum(1, keepdim=True)
        return pred, pred_sal, pred_cls, pred_cls0


if __name__ == "__main__":
    fcn = MTFCN(c_output=21).cuda()
    x = torch.Tensor(2, 3, 256, 256).cuda()
    sb = fcn(Variable(x))
    pdb.set_trace()
