import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os

from model import *
from train import *
from thop import profile

def measure_for_model(model,device='cuda'):
	if model.dataset == 'cifar10' or model.dataset == 'cifar100':
		input_tensor = torch.randn(1,3,32,32).to(device)
	elif model.dataset == 'imagenet':
		input_tensor = torch.randn(1,3,224,224).to(device)
	flops,params = profile(model.to(device), inputs=(input_tensor, ))
	return flops,params

def measure_for_block(block,input_size,device='cuda'):
	input_tensor = torch.randn(1,block.in_channels,input_size,input_size).to(device)
	flops,params = profile(block.to(device), inputs=(input_tensor, ))
	return flops,params

#get_left_bound 返回对于一个待剪枝网络每一层的最小剪枝率:bn.weigth<1e-5
def get_left_bound(model,precent):
	model_class = model.model_info()[-1]
	if model_class=='vgg':
		return get_left_bound_vgg(model,precent)
	elif model_class=='res':
		return get_left_bound_res(model,precent)
	elif model_class=='mobile':
		return get_left_bound_mobile(model,precent)

def get_left_bound_vgg(model,precent):
	left_bound = []
	total = 0
	for block in model.feature.modules():
		if isinstance(block,Basic_block):
			m = block.bn		
			total += m.weight.data.shape[0]
	bn = torch.zeros(total)
	index = 0
	for block in model.feature.modules():
		if isinstance(block,Basic_block):
			m = block.bn
			size = m.weight.data.shape[0]
			bn[index:(index+size)] = m.weight.data.abs().clone()
			index += size
	y, i = torch.sort(bn)
	thre_index = int(total * precent)
	thre = y[thre_index]
	for block in model.feature.modules():
		if isinstance(block,Basic_block):
			total = block.bn.weight.shape[0]
			weight_copy = block.bn.weight.data.abs().clone()
			mask = weight_copy.gt(thre).float()
			left_bound.append(1.0-1.0*torch.sum(mask)/total)
	return left_bound

def get_left_bound_res(model,precent):
	left_bound = []
	total = 0
	for block in model.feature.modules():
		if isinstance(block,Basic_block):
			m = block.bn
			total += m.weight.data.shape[0]
		elif isinstance(block,Res_block):
			m = block.bn_slim
			total += m.weight.data.shape[0]
	bn = torch.zeros(total)
	index = 0
	for block in model.feature.modules():
		if isinstance(block,Basic_block):
			m = block.bn
			size = m.weight.data.shape[0]
			bn[index:(index+size)] = m.weight.data.abs().clone()
			index += size
		elif isinstance(block,Res_block):
			m = block.bn_slim
			size = m.weight.data.shape[0]
			bn[index:(index+size)] = m.weight.data.abs().clone()
			index += size

	y, i = torch.sort(bn)
	thre_index = int(total * precent)
	thre = y[thre_index]
	for block in model.feature.modules():
		if isinstance(block,Basic_block):
			total = block.bn.weight.shape[0]
			weight_copy = block.bn.weight.data.abs().clone()
			mask = weight_copy.gt(thre).float()
			left_bound.append(1.0-1.0*torch.sum(mask)/total)
		elif isinstance(block,Res_block):
			total = block.bn_slim.weight.shape[0]
			weight_copy = block.bn_slim.weight.data.abs().clone()
			mask = weight_copy.gt(thre).float()
			left_bound.append(1.0-1.0*torch.sum(mask)/total)
	return left_bound

def get_left_bound_mobile(model,precent):
	left_bound = []
	total = 0
	for block in model.feature.modules():
		if isinstance(block, Mobile_block):
			m = block.bn2
			total += m.weight.data.shape[0]
	bn = torch.zeros(total)
	index = 0
	for block in model.feature.modules():
		if isinstance(block, Mobile_block):
			m = block.bn2
			size = m.weight.data.shape[0]
			bn[index:(index+size)] = m.weight.data.abs().clone()
			index += size
	y, i = torch.sort(bn)
	thre_index = int(total * precent)
	thre = y[thre_index]
	for block in model.feature.modules():
		if isinstance(block,Basic_block):
			total = block.bn.weight.shape[0]
			weight_copy = block.bn.weight.data.abs().clone()
			mask = weight_copy.gt(thre).float()
			left_bound.append(1.0-1.0*torch.sum(mask)/total)
		elif isinstance(block,Mobile_block):
			total = block.bn2.weight.shape[0]
			weight_copy = block.bn2.weight.data.abs().clone()
			mask = weight_copy.gt(thre).float()
			left_bound.append(1.0-1.0*torch.sum(mask)/total)
	return left_bound

def to_numpy(var):
	use_cuda = torch.cuda.is_available()
	return var.cpu().data.numpy() if use_cuda else var.data.numpy()

def to_tensor(ndarray, requires_grad=False):  # return a float tensor by default
	tensor = torch.from_numpy(ndarray).float()  # by default does not require grad
	if requires_grad:
		tensor.requires_grad_()
	return tensor.cuda() if torch.cuda.is_available() else tensor
