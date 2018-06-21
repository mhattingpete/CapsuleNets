import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

def softmax(x,dim=1):
	xt = x.transpose(dim,len(x.size())-1)
	out = F.softmax(xt.contiguous().view(-1,xt.size(-1)),dim=-1)
	return out.view(*xt.size()).transpose(dim,len(x.size())-1)

class CapsLayer(nn.Module):
	def __init__(self,in_channels,out_channels,num_capsules,num_route_nodes=None,kernel_size=None,stride=None,num_route_iters=3):
		super().__init__()
		# parameters for routing
		self.num_route_nodes = num_route_nodes
		self.num_route_iters = num_route_iters

		# parameters for layer
		self.num_capsules = num_capsules

		if self.num_route_nodes:
			self.route_weights = nn.Parameter(torch.randn(num_capsules,num_route_nodes,in_channels,out_channels))
		else:
			self.capsules = nn.ModuleList([nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=0) for _ in range(num_capsules)])

	def squash(self,tensor,dim=-1):
		normsq = (tensor**2).sum(dim=dim,keepdim=True)
		scale = normsq/(1.+normsq)
		return scale*tensor/torch.sqrt(normsq)

	def forward(self,x):
		if self.num_route_nodes:
			priors = x[None,:,:,None,:] @ self.route_weights[:,None,:,:,:]
			logits = Variable(torch.zeros(*priors.size()))
			if use_cuda:
				logits = logits.cuda()
			for i in range(self.num_route_iters):
				probs = softmax(logits,dim=2)
				outputs = self.squash((probs*priors).sum(dim=2,keepdim=True))
				if i != self.num_route_iters-1:
					delta_logits = (priors*outputs).sum(dim=-1,keepdim=True)
					logits = logits + delta_logits
		else:
			outputs = [cap(x).view(x.size(0),-1,1) for cap in self.capsules]
			outputs = torch.cat(outputs,dim=-1)
			outputs = self.squash(outputs)
		return outputs