import os
from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        print('Making model...')

        self.device = torch.device('cuda')
        module = import_module('model.' + args.model)
        self.model = module.make_model(args).to(self.device)

    def forward(self, x):
        return self.model(x)
