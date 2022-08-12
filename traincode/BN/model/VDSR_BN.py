import torch
import torch.nn as nn
from model import common
import torch.nn.functional as F

def make_model(args, parent=False):
    return VDSR_BN(args)

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = common.CustomBN2D(64)
        
    def forward(self, xt):
        mu, var, x, l, use_runstat = xt[0], xt[1], xt[2], xt[3], xt[4]
        x = self.conv(x)
        xt = self.bn((mu, var, x, l, use_runstat))
        mu, var, x, l, use_runstat = xt[0], xt[1], xt[2], xt[3], xt[4]
        return mu, var, self.relu(x), l, use_runstat
        
class VDSR_BN(nn.Module):
    def __init__(self, args):
        super(VDSR_BN, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=args.n_colors, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.output = nn.Conv2d(in_channels=64, out_channels=args.n_colors, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Upsample(scale_factor=int(args.scale), mode='bicubic', align_corners=False)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, xt):

        mu, var, x, l, use_runstat = xt[0], xt[1], xt[2], xt[3], xt[4]
        x = self.up(x).clamp(min=0, max=1.0)

        residual = x
        x = self.relu(self.input(x))
        xt = (mu, var, x, l, use_runstat)
        xt = self.residual_layer(xt)
        mu, var, x, l, use_runstat = xt[0], xt[1], xt[2], xt[3], xt[4]
        x = self.output(x)
        x = torch.add(x, residual)

        return mu, var, x, l, use_runstat