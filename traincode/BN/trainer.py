import os
import sys
from math import exp
import torch
import torchvision
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pickle
from PIL import Image
import math
import random
import numpy as np
from common import Image_Quality_Metric
import time
from decimal import Decimal
from scipy import io
import imageio

class Trainer():
	def __init__(self, args, my_loader, my_model, my_loss, my_optimizer):
		self.args = args
		if not args.test_only: self.loader_train = my_loader.loader_train
		self.loader_valid = my_loader.loader_valid
		self.model = my_model
		self.loss = my_loss
		self.optimizer = my_optimizer
		self.quality = Image_Quality_Metric()
		self.bestvalpsnr = 0
		self.bestepoch = 0
		self.decay = args.decay.split('+')
		self.decay = [int(i) for i in self.decay] 
		self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.decay, gamma=args.gamma)
		if args.resume or args.test_only:
			checkpoint = torch.load(args.trained_model)
			self.model.load_state_dict(checkpoint['model'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			self.scheduler.load_state_dict(checkpoint['scheduler'])
			self.bestvalpsnr = checkpoint['psnr']
			self.bestepoch = checkpoint['epoch']
			print(self.scheduler.last_epoch+1)
		if args.pre_train:
			checkpoint = torch.load(args.trained_model)
			pretrained_dict = checkpoint['model']
			new_state_dict = self.model.state_dict()
			for name, param in pretrained_dict.items():
				if name in new_state_dict:
					input_param = new_state_dict[name]
					if input_param.shape == param.shape:
						input_param.copy_(param)

	def train(self):
		self.model.train()
		train_loss = 0
		lr = self.optimizer.param_groups[0]['lr']
		print('===========================================\n')
		print('[Epoch: %d] [Learning Rate: %.2e]'%(self.scheduler.last_epoch+1, Decimal(lr)))
		startepoch = time.time()
		for i_batch, (HRbatch, LRbatch, _) in enumerate(self.loader_train):

			HRbatch, LRbatch = Variable(HRbatch.cuda()), Variable(LRbatch.cuda())
			imean = []
			ivar = []
			_input = (imean, ivar, LRbatch/(255.0/self.args.rgb_range), 0, True)
			self.optimizer.zero_grad()
			_output = self.model(_input)
			SRbatch = _output[2]
			loss = self.loss(SRbatch, HRbatch/(255.0/self.args.rgb_range))
			loss.backward()
			self.optimizer.step()
			train_loss += loss.data.cpu()

			if (i_batch+1)%100==0:
				train_loss = train_loss/100
				total_time = time.time()-startepoch
				startepoch = time.time()
				print('[Batch: %d] [Train Loss: %f] [Time: %f s]'%(i_batch+1, train_loss, total_time))
				train_loss = 0
		self.scheduler.step()

	def get_meanvar(self):
		self.model.train()
		totalmean = []
		totalvar = []
		for i_batch, (HRbatch, LRbatch, _) in enumerate(self.loader_train):

			HRbatch, LRbatch = HRbatch.cuda(), LRbatch.cuda()
			imean = []
			ivar = []
			_input = (imean, ivar, LRbatch/(255.0/self.args.rgb_range), 0, True)
			with torch.no_grad():
				_output = self.model(_input)
			totalmean.append(_output[0])
			totalvar.append(_output[1])
		with open('meanvar_dir/'+self.args.model+'_'+self.args.scale+'_var.pickle', 'wb') as handle:
			pickle.dump(totalvar, handle, protocol=pickle.HIGHEST_PROTOCOL)
		with open('meanvar_dir/'+self.args.model+'_'+self.args.scale+'_mean.pickle', 'wb') as handle:
			pickle.dump(totalmean, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def valid(self):
		print('\n')
		torch.set_grad_enabled(False)
		self.model.eval()
		if self.args.test_only:
			if not os.path.exists(os.path.join('OurResults', self.args.model, 'BNSR', self.args.scale, self.args.data_val)):
				os.makedirs(os.path.join('OurResults', self.args.model, 'BNSR', self.args.scale, self.args.data_val))
		val_psnr = 0
		for i_batch, (HRimage, LRimage, filename) in enumerate(self.loader_valid):
			sys.stdout.write("[Scale: %s] BNSR Calculation in Progress: %d/ %d  \r" % (self.args.scale, i_batch+1, len(self.loader_valid)) )
			sys.stdout.flush()
			HRimage = self.modcrop(HRimage, int(self.args.scale))
			HRimage, LRimage = HRimage.cuda(), LRimage.cuda()
			imean = []
			ivar = []
			_input = (imean, ivar, LRimage/(255.0/self.args.rgb_range), 0, True)
			torch.cuda.empty_cache()
			with torch.no_grad():
				_output = self.model(_input)
				SRimage = _output[2]
			SRimage = torch.round((255.0/self.args.rgb_range)*torch.clamp(SRimage, 0, self.args.rgb_range))
			if self.args.test_only:
				imageio.imwrite(os.path.join('OurResults', self.args.model, 'BNSR', self.args.scale, self.args.data_val)+'/'+filename[0]+'png', \
							 np.uint8(torch.squeeze(SRimage).cpu().numpy().transpose((1, 2, 0))))
			val_psnr += self.quality.psnr(SRimage, HRimage, int(self.args.scale))
		torch.set_grad_enabled(True)
		if not self.args.test_only:
			if self.bestvalpsnr<val_psnr/len(self.loader_valid):
				self.bestepoch = self.scheduler.last_epoch
				self.bestvalpsnr = val_psnr/len(self.loader_valid)
				torch.save({'model': self.model.state_dict(), 'psnr': self.bestvalpsnr, 'epoch': self.bestepoch,
							'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
							}, 'model_dir/'+self.args.model+'_'+self.args.scale+'_bestPSNR.pth.tar')
			torch.save({'model': self.model.state_dict(), 'psnr': self.bestvalpsnr, 'epoch': self.bestepoch,
						 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
						}, 'model_dir/'+self.args.model+'_'+self.args.scale+'_checkpoint.pth.tar')

		print('\n\nTest PSNR:    %f     (Best PSNR: %f at %d)'%(val_psnr/len(self.loader_valid), self.bestvalpsnr, self.bestepoch))

	def test(self, ibatch, itimes):
		print('\n')
		with open('meanvar_dir/'+self.args.model+'_'+self.args.scale+'_var.pickle', 'rb') as handle:
			totalvar = pickle.load(handle)
		with open('meanvar_dir/'+self.args.model+'_'+self.args.scale+'_mean.pickle', 'rb') as handle:
			totalmean = pickle.load(handle)
		torch.set_grad_enabled(False)
		self.model.eval()
		if not os.path.exists(os.path.join('OurResults', self.args.model, 'BayesSR', self.args.scale, self.args.data_val)):
			os.makedirs(os.path.join('OurResults', self.args.model, 'BayesSR', self.args.scale, self.args.data_val))
		val_psnr = 0
		for i_batch, (HRimage, LRimage, filename) in enumerate(self.loader_valid):

			sys.stdout.write("[Scale: %s] BayesSR Calculation in Progress: %d/ %d  \r" % (self.args.scale, i_batch+1, len(self.loader_valid)) )
			sys.stdout.flush()

			imv = list(zip(totalmean, totalvar))
			random.shuffle(imv)
			totalmean, totalvar = zip(*imv)

			HRimage = self.modcrop(HRimage, int(self.args.scale))
			HRimage, LRimage = HRimage.cuda(), LRimage.cuda()

			BLRimage = torch.cat([LRimage]*ibatch)
			BSRimage = []

			for loop in range(itimes):
				imean = []
				ivar = []
				for j in range(len(totalmean[0])):
					ls = [torch.unsqueeze(totalmean[i][j], 0) for i in range(loop*ibatch, (loop+1)*ibatch)]
					imean.append(torch.cat(ls, dim=0).cuda())
					ls = [torch.unsqueeze(totalvar[i][j], 0) for i in range(loop*ibatch, (loop+1)*ibatch)]
					ivar.append(torch.cat(ls, dim=0).cuda())
				_input = (imean, ivar, BLRimage/(255.0/self.args.rgb_range), 0, False)
				torch.cuda.empty_cache()
				with torch.no_grad():
					_output = self.model(_input)
					SRimage = _output[2]
					#SRimage = torch.round((255.0/self.args.rgb_range)*torch.clamp(SRimage, 0, self.args.rgb_range))
					SRimage = torch.clamp(SRimage, 0, self.args.rgb_range)
					BSRimage.append(SRimage.cpu())
			#BSRimage = np.uint8(torch.cat(BSRimage, dim=0).numpy())
			#io.savemat(os.path.join('OurResults', 'BayesSR', self.args.scale, self.args.data_val)+'/'+filename[0]+'mat', {"SRimage": BSRimage})
			BSRimage = torch.cat(BSRimage, dim=0)
			torch.save(BSRimage, os.path.join('OurResults', self.args.model, 'BayesSR', self.args.scale, self.args.data_val)+'/'+filename[0]+'pt')

	def modcrop(self, img, scale):
		sz1 = img.shape[2] - img.shape[2]%scale
		sz2 = img.shape[3] - img.shape[3]%scale
		img = img[:, :, 0:sz1, 0:sz2]
		return img

	def terminate(self):
		if self.args.test_only:
			self.test()
			return True
		else:
			epoch = self.scheduler.last_epoch
			return epoch >= self.args.epochs


