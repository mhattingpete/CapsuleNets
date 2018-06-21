import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from torch.autograd import Variable
from torch import optim
from torchvision import datasets, transforms

from models import CapsuleNet
from utils import train,test,augmentation


if __name__ == '__main__':
	print(torch.cuda.is_available())
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)

	# Training dataset
	train_loader = torch.utils.data.DataLoader(datasets.MNIST(root='.',train=True,transform=transforms.Compose([
	    transforms.ToTensor(),augmentation
	]),download=True),batch_size=64,shuffle=True)
	# Test dataset
	test_loader = torch.utils.data.DataLoader(datasets.MNIST(root='.',train=False,transform=transforms.Compose([
	    transforms.ToTensor(),augmentation
	])),batch_size=64,shuffle=True)

	model = CapsuleNet(input_size=1,output_size=10).to(device)

	optimizer = optim.Adam(model.parameters())

	print("# parameters:", sum(param.numel() for param in model.parameters()))

	epochs = 30
	for epoch in range(1,epochs+1):
		train_loss = train(model,device,train_loader,optimizer,epoch,log_interval=10)
		test_loss = test(model,device,test_loader)
	torch.save(model.state_dict(),"./Saved_models/CapsuleNetwork.pt")