import torch
import torch.nn as nn

class CustomBN2D(nn.Module):
	def __init__(self, feat, eps=1e-5, momentum=0.1):
		super(CustomBN2D, self).__init__()
		self.feat = feat
		self.eps = eps
		self.momentum = momentum

		self.weight = torch.nn.Parameter(data=torch.Tensor(self.feat, 1, 1), requires_grad=True)        
		self.bias = torch.nn.Parameter(data=torch.Tensor(self.feat, 1, 1), requires_grad=True) 
		torch.nn.init.ones_(self.weight)
		torch.nn.init.zeros_(self.bias)      

		self.register_buffer('running_mean', torch.zeros(self.feat))
		self.register_buffer('running_var', torch.ones(self.feat))   
		self.running_mean.zero_()
		self.running_var.fill_(1)   
	
	def forward(self, xtuple):
		xmu, xvar, x, l, use_runstat = xtuple[0], xtuple[1], xtuple[2], xtuple[3], xtuple[4]

		if self.training:
			x_ = torch.transpose(x, 0, 1)
			x_ = x_.reshape(x_.shape[0], x_.shape[1]*x_.shape[2]*x_.shape[3])

			mu = torch.mean(x_, dim=1)
			var = torch.var(x_, dim=1)
			self.running_mean = (1-self.momentum)*self.running_mean+self.momentum*mu
			self.running_var  = (1-self.momentum)*self.running_var+self.momentum*var

			xmu.append(mu.data.cpu())
			xvar.append(var.data.cpu())
			mu = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(mu, 1), 1), 0).expand_as(x)
			var = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(var, 1), 1), 0).expand_as(x)
		else:
			if use_runstat:
				mu, var = self.running_mean, self.running_var
				mu = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(mu, 1), 1), 0).expand_as(x)
				var = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(var, 1), 1), 0).expand_as(x)
			else:
				mu, var = xmu[l], xvar[l] 
				mu = torch.unsqueeze(torch.unsqueeze(mu, 2), 2).expand_as(x)
				var = torch.unsqueeze(torch.unsqueeze(var, 2), 2).expand_as(x)
			 
		x_norm = torch.div(torch.add(x, torch.neg(mu)), torch.sqrt(torch.add(var, self.eps)))
		out = torch.mul(self.weight.expand_as(x), x_norm)
		out = torch.add(out, self.bias.expand_as(x))
		return xmu, xvar, out, l+1, use_runstat
