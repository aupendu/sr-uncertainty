import torch
import torch.nn as nn
from model import common
import torch.nn.functional as F

def make_model(args, parent=False):
    return VDSR_BN(args)

class INConv_ReLU_Block(nn.Module):
    def __init__(self):
        super(INConv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = common.CustomBN2D(128)
        
    def forward(self, xt):
        mu, var, x, l, use_runstat = xt[0], xt[1], xt[2], xt[3], xt[4]
        x = self.conv(x)
        xt = self.bn((mu, var, x, l, use_runstat))
        mu, var, x, l, use_runstat = xt[0], xt[1], xt[2], xt[3], xt[4]
        return mu, var, self.relu(x), l, use_runstat

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = common.CustomBN2D(128)
        
    def forward(self, xt):
        mu, var, x, l, use_runstat = xt[0], xt[1], xt[2], xt[3], xt[4]
        x = self.conv(x)
        xt = self.bn((mu, var, x, l, use_runstat))
        mu, var, x, l, use_runstat = xt[0], xt[1], xt[2], xt[3], xt[4]
        return mu, var, self.relu(x), l, use_runstat
        
class VDSR_BN(nn.Module):
    def __init__(self, args):
        super(VDSR_BN, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 11)
        self.input = INConv_ReLU_Block()
        self.relu = nn.ReLU(inplace=True)
        r = int(args.scale)
        
        if r == 2 or r == 3:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128*r*r, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(r),
                nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True),
            )
        elif r == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128*4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.Conv2d(in_channels=128, out_channels=128*4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True),
            )
        elif r == 8:
            self.upscale = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128*4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.Conv2d(in_channels=128, out_channels=128*4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.Conv2d(in_channels=128, out_channels=128*4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True),
            )

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, xt):

        xt = self.input(xt)
        xt = self.residual_layer(xt)
        mu, var, x, l, use_runstat = xt[0], xt[1], xt[2], xt[3], xt[4]
        x = self.relu(self.upscale(x))

        return mu, var, x, l, use_runstat