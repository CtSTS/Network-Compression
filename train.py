import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data.dataloader as DataLoader
import torch.optim as optim
import numpy as np
import math
import os
from torch.autograd import Variable

from model import *

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k)
	return res

def train(model,train_loader,test_loader,criterion,optimizer,epoch,device):
	model.train()
	avg_loss = 0.
	train_acc_1 = 0.
	train_acc_5 = 0.
	for batch_idx, (data, target) in enumerate(train_loader):
		data = Variable(data)
		target = Variable(target)
		if device == 'cuda':
			data = data.cuda()
			target = target.cuda()
		optimizer.zero_grad()
		output,_ = model(data)
		loss = criterion(output, target)
		avg_loss += loss.data
		#pred = output.data.max(1, keepdim=True)[1]	
		#train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
		prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
		train_acc_1 += prec1
		train_acc_5 += prec5
		loss.backward()
		optimizer.step()
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data))
	return (avg_loss/len(train_loader)),(100.*train_acc_1/len(train_loader.dataset)),(100.*train_acc_5/len(train_loader.dataset))

def train_slimming(model,train_loader,test_loader,criterion,optimizer,warmup_scheduler,warm,s,epoch,device):
	model.train()
	avg_loss = 0.
	train_acc_1 = 0.
	train_acc_5 = 0.
	for batch_idx, (data, target) in enumerate(train_loader):
		if epoch <= warm:
			warmup_scheduler.step()
		data = Variable(data)
		target = Variable(target)
		if device == 'cuda':
			data = data.cuda()
			target = target.cuda()

		optimizer.zero_grad()
		output,_ = model(data)
		loss = criterion(output, target)
		avg_loss += loss.data
		#pred = output.data.max(1, keepdim=True)[1]
		#train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
		prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
		train_acc_1 += prec1
		train_acc_5 += prec5
		loss.backward()
		for m in model.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.weight.grad.data.add_(s*torch.sign(m.weight.data))

		optimizer.step()
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data))
			print(optimizer.param_groups[0]['lr'])
	return (avg_loss/len(train_loader)),(100.*train_acc_1/len(train_loader.dataset)),(100.*train_acc_5/len(train_loader.dataset))



def Attention_loss(t,s):
	t = torch.abs(t)
	s = torch.abs(s)
	t = torch.sum(t,1)
	s = torch.sum(s,1)
	loss_sum = torch.mul((t - s),(t - s))
	loss = torch.sum(loss_sum)/t.shape[0]
	return math.sqrt(loss)




def train_kd(model_s,train_loader,test_loader,criterion,optimizer,model_t,T,alpha,beta,epoch,device):
	model_s.train()
	avg_loss = 0.
	train_acc_1 = 0.
	train_acc_5 = 0.
	for batch_idx, (data, target) in enumerate(train_loader):
		data = Variable(data)
		target = Variable(target)
		if device == 'cuda':
			data = data.cuda()
			target = target.cuda()
		optimizer.zero_grad()
		output_t,feature_t = model_t(data)
		output_s,feature_s = model_s(data)

		loss = (1-alpha)*criterion(output_s, target)+nn.KLDivLoss()(F.log_softmax(output_s/T), F.softmax(output_t/T)) * (T*T * 2.0 * alpha)
		for t,s in zip(feature_t,feature_s):
			loss = loss + beta*0.5*Attention_loss(t,s)
		avg_loss += loss.data
		prec1, prec5 = accuracy(output_s.data, target, topk=(1, 5))
		train_acc_1 += prec1
		train_acc_5 += prec5
		loss.backward()
		optimizer.step()
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data))
	return (avg_loss/len(train_loader)),(100.*train_acc_1/len(train_loader.dataset)),(100.*train_acc_5/len(train_loader.dataset))

def test(model,test_loader,criterion,device):
	model.eval()
	with torch.no_grad():
		test_loss = 0.
		correct_1 = 0.
		correct_5 = 0.
		for batch_idx, (data, target) in enumerate(test_loader):
			data = Variable(data)
			target = Variable(target)
			if device == 'cuda':
				data = data.cuda()
				target = target.cuda()
			output,_ = model(data)
			test_loss += criterion(output, target).data
			#pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
			#correct += pred.eq(target.data.view_as(pred)).cpu().sum()
			if model.dataset == 'imagenet':
				target = target + 1
			prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
			correct_1 += prec1
			correct_5 += prec5
		test_loss /= len(test_loader)
		print('\nTest set: Average loss: {:.4f}, Acc_top1: {}/{} ({:.1f}%), Acc_top5: {}/{} ({:.1f}%)\n'.format(
			test_loss, correct_1, len(test_loader.dataset),
			100. * correct_1 / len(test_loader.dataset), correct_5, len(test_loader.dataset),
			100. * correct_5 / len(test_loader.dataset)))
		return test_loss,(100. *correct_1 / float(len(test_loader.dataset))),(100. *correct_5 / float(len(test_loader.dataset)))
		
def test_simple(model,test_loader,criterion,device):
	with torch.no_grad():
		test_loss = 0.
		correct_1 = 0.
		correct_5 = 0.
		for batch_idx, (data, target) in enumerate(test_loader):
			data = Variable(data)
			target = Variable(target)
			if batch_idx > 1.0*len(test_loader)/4.0:
				break
			if device == 'cuda':
				data = data.cuda()
				target = target.cuda()
			output,_ = model(data)
			test_loss += criterion(output, target).data
			#pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
			#correct += pred.eq(target.data.view_as(pred)).cpu().sum()
			if model.dataset == 'imagenet':
				target = target + 1
			prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
			correct_1 += prec1
			correct_5 += prec5
		test_loss /= (len(test_loader)/4.0)
		print('\nTest set: Average loss: {:.4f}, Acc_top1: {}/{} ({:.1f}%), Acc_top5: {}/{} ({:.1f}%)\n'.format(
			test_loss, correct_1, 0.25*len(test_loader.dataset),
			400. * correct_1 / len(test_loader.dataset), correct_5, 0.25*len(test_loader.dataset),
			400. * correct_5 / len(test_loader.dataset)))
		return test_loss,(400. *correct_1 / float(len(test_loader.dataset))),(400. *correct_5 / float(len(test_loader.dataset)))