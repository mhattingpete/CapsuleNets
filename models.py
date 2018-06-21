import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from layers import CapsLayer

class CapsuleNet(nn.Module):
	def __init__(self,input_size,output_size):
		super().__init__()
		self.output_size = output_size
		self.conv = nn.Conv2d(in_channels=input_size,out_channels=256,kernel_size=9,stride=1)
		self.primaryCaps = CapsLayer(in_channels=256,out_channels=32,num_capsules=8,kernel_size=9,stride=2)
		self.digitCaps = CapsLayer(in_channels=8,out_channels=16,num_capsules=output_size,num_route_nodes=32*6*6)
		self.decoder = nn.Sequential(
			nn.Linear(16*output_size,512),
			nn.ELU(),
			nn.Linear(512,1024),
			nn.ELU(),
			nn.Linear(1024,784),
			nn.Sigmoid()
			)
		self.elu = nn.ELU()
		self.softmax = nn.LogSoftmax(dim=-1)

	def forward(self,x,y=None):
		x = self.elu(self.conv(x))
		x = self.primaryCaps(x)
		x = self.digitCaps(x).squeeze().transpose(0,1)
		out = (x**2).sum(dim=-1)**0.5
		out = self.softmax(out)
		if y is None:
			# get most active capsule
			_,max_index = out.max(dim=1)
			max_index = max_index.to(torch.device("cpu"))
			y = Variable(torch.eye(self.output_size)).index_select(dim=0,index=max_index.data)
			y = y.to(torch.device("cuda"))
		rec = self.decoder((x*y[:,:,None]).view(x.size(0),-1))
		return out,rec