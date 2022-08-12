import torch
import model
from option import args
from trainer import Trainer
from dataloader import Data
import numpy as np
import random
import time

#torch.backends.cudnn.deterministic=True
#torch.backends.cudnn.benchmark = True

torch.manual_seed(args.seed)
#torch.cuda.manual_seed(args.seed)
#np.random.seed(args.seed)
#random.seed(args.seed)

def main():
	global model
	loader = Data(args)
	_model = model.Model(args)
	_optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
	_loss = torch.nn.MSELoss()
	t = Trainer(args, loader, _model, _loss, _optimizer)
	if args.test_only:
		t.valid()
		t.test(ibatch=4, itimes=25)
	else:
		if not args.only_MV:
			while not t.terminate():
				t.train()
				t.valid()
		if args.save_MV:
			print('Mean-Var Calculation Started !')
			t.get_meanvar()

if __name__ == '__main__':
	main()
