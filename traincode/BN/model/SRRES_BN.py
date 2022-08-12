import torch
import torch.nn as nn
from model import common
import torch.nn.functional as F
import math

def make_model(args, parent=False):
	return SRRES_BN(args)


class MeanShift(nn.Conv2d):
	def __init__(self, rgb_mean, sign):
		super(MeanShift, self).__init__(3, 3, kernel_size=1)
		self.weight.data = torch.eye(3).view(3, 3, 1, 1)
		self.bias.data = float(sign) * torch.Tensor(rgb_mean)

		# Freeze the MeanShift layer
		for params in self.parameters():
			params.requires_grad = False

class _Residual_Block(nn.Module):
	def __init__(self):
		super(_Residual_Block, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = common.CustomBN2D(64)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = common.CustomBN2D(64)

	def forward(self, xt):
		mu, var, x, l, use_runstat = xt[0], xt[1], xt[2], xt[3], xt[4]
		identity_data = x

		x = self.conv1(x)
		xt = self.bn1((mu, var, x, l, use_runstat))
		mu, var, x, l, use_runstat = xt[0], xt[1], xt[2], xt[3], xt[4]

		x = self.conv2(self.relu(x))
		xt = self.bn2((mu, var, x, l, use_runstat))
		mu, var, x, l, use_runstat = xt[0], xt[1], xt[2], xt[3], xt[4]

		output = torch.add(x,identity_data)
		return mu, var, output, l, use_runstat 

class SRRES_BN(nn.Module):
	def __init__(self, args):
		super(SRRES_BN, self).__init__()

		r = int(args.scale)

		rgb_mean = (0.4488, 0.4371, 0.4040)
		self.sub_mean = MeanShift(rgb_mean, -1)
		self.add_mean = MeanShift(rgb_mean, 1)

		self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
		self.relu = nn.ReLU(inplace=True)
		
		self.residual = self.make_layer(_Residual_Block, 16)

		self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

		if r == 2 or r == 3:
			self.upscale = nn.Sequential(
				nn.Conv2d(in_channels=64, out_channels=64*r*r, kernel_size=3, stride=1, padding=1, bias=False),
				nn.PixelShuffle(r),
				nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
			)
		elif r == 4:
			self.upscale = nn.Sequential(
				nn.Conv2d(in_channels=64, out_channels=64*4, kernel_size=3, stride=1, padding=1, bias=False),
				nn.PixelShuffle(2),
				nn.Conv2d(in_channels=64, out_channels=64*4, kernel_size=3, stride=1, padding=1, bias=False),
				nn.PixelShuffle(2),
				nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
			)
		elif r == 8:
			self.upscale = nn.Sequential(
				nn.Conv2d(in_channels=64, out_channels=64*4, kernel_size=3, stride=1, padding=1, bias=False),
				nn.PixelShuffle(2),
				nn.Conv2d(in_channels=64, out_channels=64*4, kernel_size=3, stride=1, padding=1, bias=False),
				nn.PixelShuffle(2),
				nn.Conv2d(in_channels=64, out_channels=64*4, kernel_size=3, stride=1, padding=1, bias=False),
				nn.PixelShuffle(2),
				nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
			)
		

	def make_layer(self, block, num_of_layer):
		layers = []
		for _ in range(num_of_layer):
			layers.append(block())
		return nn.Sequential(*layers)

	def forward(self, xt):
		mu, var, x, l, use_runstat = xt[0], xt[1], xt[2], xt[3], xt[4]

		x = self.sub_mean(x)
		out = self.relu(self.conv_input(x))
		residual = out

		xt = (mu, var, out, l, use_runstat)
		xt = self.residual(xt)
		mu, var, out, l, use_runstat = xt[0], xt[1], xt[2], xt[3], xt[4]
		out = self.relu(self.conv_mid(out))
		out = torch.add(out,residual)
		
		out = self.upscale(out)
		out = self.add_mean(out)
		return mu, var, out, l, use_runstat