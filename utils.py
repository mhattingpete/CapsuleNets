import torch.nn.functional as F
import torch
import numpy as np

def augmentation(x,max_shift=2):
	x = x.unsqueeze(1).float()/255.0
	_,_,height,width = x.size()
	h_shift, w_shift = np.random.randint(-max_shift,max_shift+1,size=2)
	source_height_slice = slice(max(0,h_shift),h_shift+height)
	source_width_slice = slice(max(0,w_shift),w_shift+width)
	target_height_slice = slice(max(0,-h_shift),-h_shift+height)
	target_width_slice = slice(max(0,-w_shift),-w_shift+width)
	shifted_image = torch.zeros(*x.size())
	shifted_image[:,:,source_height_slice,source_width_slice] = x[:,:,target_height_slice,target_width_slice]
	return shifted_image.float()

def capsule_loss(images,labels,predictions,reconstructions):
	left = F.relu(0.9-predictions)**2
	right = F.relu(predictions-0.1)**2
	margin_loss = (labels*left+0.5*(1.-labels)*right).sum()
	images = images.view(reconstructions.size()[0],-1)
	recon_loss = F.mse_loss(reconstructions,images,size_average=False)
	return (margin_loss+0.0005*recon_loss)/images.size(0)

def train(model,device,train_loader,optimizer,epoch,log_interval):
	model.train()
	train_loss = 0
	for batch_idx,(data,target) in enumerate(train_loader):
		data,target = data.to(device),target.to(device)
		target = torch.eye(10).index_select(dim=0,index=target)
		optimizer.zero_grad()
		data = data.squeeze(1)
		output,reconstruction = model(data,target)
		loss = capsule_loss(data,target,output,reconstruction)
		train_loss += loss.item()
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch,batch_idx*len(data),len(train_loader.dataset),100.*batch_idx/len(train_loader),loss.item()))
	return train_loss / len(train_loader.dataset)

def test(model,device,test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data,target in test_loader:
			data,target = data.to(device),target.to(device)
			data = data.squeeze(1)
			output,reconstruction = model(data)
			test_loss += capsule_loss(data,target,output,reconstruction).item() # sum up batch loss
			pred = output.max(1,keepdim=True)[1] # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
	test_loss /= len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss,correct,len(test_loader.dataset),100.*correct/len(test_loader.dataset)))
	return test_loss