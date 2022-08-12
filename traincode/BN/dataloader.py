import os
import sys
import glob
import torch
import torchvision
import time
import random
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms, utils
import imageio
import pickle
import torch.nn.functional as F

class CustomDataset(Dataset):
	def __init__(self, args, is_train):

		self.factor = int(args.scale)  
		self.is_train = is_train
		self.args = args
		self.lrfldr='BICUBIC' if args.BI_input  else 'LR'
		if self.is_train:
			self.data = args.data_train
			self.lr_patch = args.lr_patch_size
			self.batch_size = args.batch_size
			self.test_every = args.test_every
		else:
			self.data = args.data_val
			self.batch_size = args.batch_size_val
			self.test_every = 1
		
		self.HR_paths = os.path.join(args.HRpath, self.data)
		self.LR_paths = os.path.join(args.LRpath, self.lrfldr, self.data, 'X'+"%0.2f"%self.factor)
		self.path_bin = os.path.join(args.LRpath, 'bin', self.data)
		
		os.makedirs(self.path_bin, exist_ok=True)
		self.files = os.listdir(self.HR_paths)
		self.HR_paths, self.LR_paths = self._save()
		self.repeat = (self.batch_size*self.test_every)//len(self.files)

	def _save(self):
		path_HR = os.path.join(self.path_bin, 'HR')
		os.makedirs(path_HR, exist_ok=True)

		path_LR = os.path.join(self.path_bin, self.lrfldr)
		os.makedirs(path_LR, exist_ok=True)

		path_LR = os.path.join(path_LR, 'X'+"%0.2f"%self.factor)
		os.makedirs(path_LR, exist_ok=True)

		for i in range(len(self.files)):
			if os.path.exists(path_LR+'/'+self.files[i][:-4]+'.pt')==False:
				print('Making file:  '+self.files[i])
				_LR = imageio.imread(self.LR_paths+'/'+self.files[i])
				if len(_LR.shape)<3: _LR = np.stack([_LR for _ in range(3)], axis=2)
				with open(path_LR+'/'+self.files[i][:-4]+'.pt', 'wb') as _f:
					pickle.dump(_LR, _f)

			if os.path.exists(path_HR+'/'+self.files[i][:-4]+'.pt')==False:
				_HR = imageio.imread(self.HR_paths+'/'+self.files[i])
				if len(_HR.shape)<3: _HR = np.stack([_HR for _ in range(3)], axis=2)
				with open(path_HR+'/'+self.files[i][:-4]+'.pt', 'wb') as _f:
					pickle.dump(_HR, _f)
		return glob.glob(path_HR+'/*.pt'), glob.glob(path_LR+'/*.pt')
	
	def _crop(self, iHR, iLR, psize, f):
		ih, iw = iLR.shape[:2]
		x0 = random.randrange(0, ih-psize+1)
		x1 = random.randrange(0, iw-psize+1)
		if self.lrfldr=='LR':
			HRbatch = iHR[f*x0:f*x0+f*psize, f*x1:f*x1+f*psize, :]
		else:
			HRbatch = iHR[x0:x0+psize, x1:x1+psize, :]
		LRbatch = iLR[x0:x0+psize, x1:x1+psize, :]
		HRbatch, LRbatch = self._augment(HRbatch, LRbatch)
		return self._np2Tensor(HRbatch), self._np2Tensor(LRbatch)

	def _augment(self, imgHR, imgLR, is_aug=True):
		hflip = is_aug and random.random() < 0.5
		vflip = is_aug and random.random() < 0.5
		rot90 = is_aug and random.random() < 0.5
		if hflip: 
			imgHR = imgHR[:, ::-1, :]
			imgLR = imgLR[:, ::-1, :]
		if vflip: 
			imgHR = imgHR[::-1, :, :]
			imgLR = imgLR[::-1, :, :]
		if rot90: 
			imgHR = imgHR.transpose(1, 0, 2)
			imgLR = imgLR.transpose(1, 0, 2)
		return imgHR, imgLR

	def _np2Tensor(self, img):
		np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
		tensor = torch.from_numpy(np_transpose).float()
		return tensor

	def __getitem__(self, index):
		if self.is_train: index = self._get_index(index)
		with open(self.HR_paths[index], 'rb') as _f:
			HR = pickle.load(_f)
		with open(self.LR_paths[index], 'rb') as _f:
			LR = pickle.load(_f)
		if self.is_train:
			HRbatch, LRbatch = self._crop(HR, LR, self.lr_patch, self.factor)
		else:
			HRbatch, LRbatch = self._np2Tensor(HR), self._np2Tensor(LR)
		head_tail = os.path.split(self.HR_paths[index]) 
		return HRbatch, LRbatch, head_tail[1][:-2]

	def __len__(self): 
		if self.is_train:
			return len(self.HR_paths)*self.repeat
		else:
			return len(self.HR_paths)

	def _get_index(self, index):
		return index % len(self.files)


class Data:
	def __init__(self, args):
		val_dataset = CustomDataset(args, is_train=False)
		self.loader_valid = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=args.n_threads)
		if not args.test_only:
			train_dataset = CustomDataset(args, is_train=True)
			self.loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
		

