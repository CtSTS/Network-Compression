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

#Block 

class Basic_block(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Basic_block, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.conv = nn.Conv2d(in_channels,out_channels, kernel_size=3,stride=1, padding=1, bias=False)
		self.bn = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
	def forward(self,x):
		out = self.conv(x)
		out = self.bn(out)
		out = self.relu(out)
		return out

class Res_block(nn.Module):
	def __init__(self, in_channels,mid_channels, out_channels, stride=1, downsample=None):
		super(Res_block, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.stride = stride
		self.left=nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
			#nn.BatchNorm2d(out_channels)
			)
		self.shortcut=nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False))
		self.bn_slim = nn.BatchNorm2d(out_channels)
		if stride == 1:
			m = self.shortcut[0]
			m.weight.data = torch.zeros(out_channels,in_channels,1,1)
			for i in range(min(in_channels,out_channels)):
				m.weight.data[i,i,:,:] = torch.full((1,1),1.0)
			for i in m.parameters():
				i.requires_grad=False
	def forward(self, x):
		out = self.left(x)
		out += self.shortcut(x)
		out = self.bn_slim(out)
		out = F.relu(out)
		return out


class Mobile_block(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Mobile_block, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
		self.bn1 = nn.BatchNorm2d(in_channels)
		self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=1, padding=0, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
	def forward(self, x):
		out = self.relu(self.bn1(self.conv1(x)))
		out = self.relu(self.bn2(self.conv2(out)))
		return out

class VGG(nn.Module):
	def __init__(self,cfg,dataset='cifar10',classifier_type='simple'):
		super(VGG,self).__init__()
		self.dataset =dataset
		if dataset == 'cifar10':
			self.num_classes = 10
			self.h = 32
			self.w = 32
		elif dataset == 'cifar100':
			self.num_classes = 100
			self.h = 32
			self.w = 32
		elif dataset == 'intel':
			self.num_classes = 6
			self.h = 150
			self.w = 150
		elif dataset == 'imagenet':
			self.num_classes = 1000
			self.h = 224
			self.w = 224
		self.cfg = cfg
		self.type = 'vgg'
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.feature = self.make_layers(cfg,64)
		if cfg[-1]=='A' or cfg[-1]=='M':
			self.classifier = nn.Sequential(nn.Linear(cfg[-2]*self.h*self.w,self.num_classes))
		else:
			self.classifier = nn.Sequential(nn.Linear(cfg[-1]*self.h*self.w,self.num_classes))
		


	def make_layers(self,cfg,in_channels):
		layers = []
		for v in cfg:
			if v == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
				self.h = self.h//2
				self.w = self.w//2
			elif v == 'A':
				layers += [nn.AvgPool2d(4)]
				self.h = self.h//4
				self.w = self.w//4
			else:
				layers += [Basic_block(in_channels,v)]
				in_channels = v
		return nn.Sequential(*layers)

	def forward(self,x):
		feature = []
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		for block in self.feature:
			x = block(x)
			feature.append(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x,feature

	def model_info(self):
		return [self.dataset,self.cfg,self.type]

class MobileNet(nn.Module):
	def __init__(self,cfg,dataset='cifar10',classifier_type='simple'):
		super(MobileNet,self).__init__()
		self.dataset =dataset
		if dataset == 'cifar10':
			self.num_classes = 10
			self.h=32
			self.w=32
		elif dataset == 'cifar100':
			self.num_classes = 100
			self.h=32
			self.w=32
		elif dataset == 'intel':
			self.num_classes = 6
			self.h = 150
			self.w = 150
		elif dataset == 'imagenet':
			self.num_classes = 1000
			self.h = 224
			self.w = 224
		self.cfg = cfg
		self.type = 'mobile'
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.feature = self.make_layers(cfg,64)
		if cfg[-1]=='A' or cfg[-1]=='M':
			self.classifier = nn.Sequential(nn.Linear(cfg[-2]*self.h*self.w,self.num_classes))
		else:
			self.classifier = nn.Sequential(nn.Linear(cfg[-1]*self.h*self.w, self.num_classes))
	def make_layers(self,cfg,in_channels):
		layers = []
		for v in cfg:
			if v == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
				self.h = self.h//2
				self.w = self.w//2
			elif v == 'A':
				layers += [nn.AvgPool2d(4)]
				self.h = self.h//4
				self.w = self.w//4		
			else:
				layers += [Mobile_block(in_channels,v)]
				in_channels = v
		return nn.Sequential(*layers)
	def forward(self,x):
		feature = []
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		for block in self.feature:
			x = block(x)
			feature.append(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x,feature
	def model_info(self):
		return [self.dataset,self.cfg,self.type]

class ResNet(nn.Module):
	def __init__(self,cfg,dataset='cifar10',classifier_type='simple',cfg_before_slim=None):
		super(ResNet,self).__init__()
		self.dataset =dataset
		if dataset == 'cifar10':
			self.num_classes = 10
			self.h=32
			self.w=32
		elif dataset == 'cifar100':
			self.num_classes = 100
			self.h=32
			self.w=32
		elif dataset == 'intel':
			self.num_classes = 6
			self.h = 150
			self.w = 150
		self.cfg = cfg
		self.cfg_before_slim = cfg_before_slim
		self.type = 'res'
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.feature = self.make_layers(cfg,64,cfg_before_slim)
		if isinstance(cfg[-1],int):
			self.classifier = nn.Sequential(nn.Linear(cfg[-1]*self.h*self.w,self.num_classes))
		elif cfg[-1]=='A' or cfg[-1]=='M':
			if isinstance(cfg[-2],int):
				self.classifier = nn.Sequential(nn.Linear(cfg[-2]*self.h*self.w,self.num_classes))
			else:
				self.classifier = nn.Sequential(nn.Linear(int(cfg[-2][0:-1])*self.h*self.w,self.num_classes))
		else:
			self.classifier = nn.Sequential(nn.Linear(int(cfg[-1][0:-1])*self.h*self.w,self.num_classes))

	def make_layers(self,cfg,in_channels,cfg_before_slim):
		layers = []
		i = 0
		for v in cfg:
			if isinstance(v,int):
				layers += [Basic_block(in_channels,v)]
				in_channels = v
			elif v[-1] == 'r':
				if cfg_before_slim == None:
					layers += [Res_block(in_channels,int(v[0:-1]),int(v[0:-1]))]
					in_channels = int(v[0:-1])
				else:
					layers += [Res_block(in_channels,int(cfg_before_slim[i][0:-1]),int(v[0:-1]))]
					in_channels = int(v[0:-1])
			elif v == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
				self.h = self.h//2
				self.w = self.w//2
			elif v == 'A':
				layers += [nn.AvgPool2d(4)]
				self.h = self.h//4
				self.w = self.w//4

			elif v[-2]=='r':
				if cfg_before_slim == None:
					layers += [Res_block(in_channels,int(v[0:-2]),int(v[0:-2]),stride=int(v[-1]))]
					in_channels = int(v[0:-2])
				else:
					layers += [Res_block(in_channels,int(cfg_before_slim[i][0:-2]),int(v[0:-2]),stride=int(v[-1]))]
					in_channels = int(v[0:-2])
				self.h = self.h//2
				self.w = self.w//2
				
			i = i + 1

		return nn.Sequential(*layers)
	def forward(self,x):
		feature = []
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		for block in self.feature:
			x = block(x)
			feature.append(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x,feature
	def model_info(self):
		return [self.cfg_before_slim,self.dataset,self.cfg,self.type]


if __name__ == '__main__':
	cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
	net = ResNet(cfg)
	x = Variable(torch.FloatTensor(16, 3, 40, 40))
	y,feature = net(x)
	print(y.data.shape)
	print(feature[0].data.shape)
	print(feature[1].data.shape)

