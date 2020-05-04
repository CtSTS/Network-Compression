import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os

from model import *
from train import *


def pruning_by_action_list(model,action_list):
	model_class = model.model_info()[-1]
	if model_class=='vgg':
		return pruning_by_action_list_vgg(model,action_list)
	elif model_class=='res':
		return pruning_by_action_list_res(model,action_list)
	elif model_class=='mobile':
		return pruning_by_action_list_mobile(model,action_list)

def pruning_by_action_list_vgg(model,action_list):
	thre_list = []
	count = 0
	for m in model.feature.modules():
		if isinstance(m,Basic_block):
			total = m.bn.weight.shape[0]
			bn = m.bn.weight.data.abs()
			y, i = torch.sort(bn)
			thre_index = int(total * (action_list[count]))
			if thre_index <= 0 :
				thre = -1
			else:
				thre = y[thre_index]
			thre_list.append(thre)
			count = count + 1
	cfg = []
	cfg_mask = []
	i = 0
	for block in model.feature.modules():
		if isinstance(block,Basic_block):
			for m in block.modules():
				if isinstance(m,nn.BatchNorm2d):
					if count == 1:
						weight_copy = m.weight.data.abs().clone()
						mask = weight_copy.gt(-1).float()
						cfg.append(int(torch.sum(mask)))
						cfg_mask.append(mask.clone())
					else:
						weight_copy = m.weight.data.abs().clone()
						mask = weight_copy.gt(thre_list[i]).float()
						cfg.append(int(torch.sum(mask)))
						cfg_mask.append(mask.clone())
					count = count - 1
					i = i + 1
		elif isinstance(block, nn.MaxPool2d):
			cfg.append('M')
		elif isinstance(block,nn.AvgPool2d):
			cfg.append('A')
	print(cfg)
	vgg_new = VGG(cfg,dataset = model.model_info()[0])
	source_feature = model.feature

	BN_id = 0
	start_mask = torch.ones(64)   #64 is one super-parameter,it can be change in VGG class
	end_mask = cfg_mask[BN_id]
	for m_s,m_t in zip(source_feature.modules(),vgg_new.feature.modules()):
		if isinstance(m_s,nn.BatchNorm2d):
			idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
			m_t.weight.data = m_s.weight.data[idx1.tolist()].clone()
			m_t.bias.data = m_s.bias.data[idx1.tolist()].clone()
			m_t.running_mean = m_s.running_mean[idx1.tolist()].clone()
			m_t.running_var = m_s.running_var[idx1.tolist()].clone()
			BN_id += 1
			start_mask = end_mask.clone()
			if not BN_id == len(cfg_mask):
				end_mask = cfg_mask[BN_id]
		elif isinstance(m_s,nn.Conv2d):
			idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
			idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
			w = m_s.weight.data[:, idx0.tolist(), :, :].clone()
			w = w[idx1.tolist(), :, :, :].clone()
			m_t.weight.data = w.clone()
	vgg_new.conv1 = model.conv1
	vgg_new.bn1 = model.bn1
	vgg_new.classifier = model.classifier
	return vgg_new

def pruning_by_action_list_res(model,action_list):
	total = 0
	count = 0
	thre_list = []
	if model.model_info()[0] == None:
		cfg_before_slim = model.model_info()[2]
	else:
		cfg_before_slim = model.model_info()[0]
	for block in model.feature.modules():
		if isinstance(block,Basic_block):
			total = block.bn.weight.shape[0]
			bn = block.bn.weight.data.abs()
			y, i = torch.sort(bn)
			thre_index = int(total * action_list[count])
			if thre_index <= 0 :
				thre = -1
			else:
				thre = y[thre_index]
			thre_list.append(thre)
			count = count + 1
		elif isinstance(block,Res_block):
			total = block.bn_slim.weight.shape[0]
			bn = block.bn_slim.weight.data.abs()
			y, i = torch.sort(bn)
			thre_index = int(total * action_list[count])
			if thre_index <= 0 :
				thre = -1
			else:
				thre = y[thre_index]
			thre_list.append(thre)
			count = count + 1

	cfg = []
	cfg_mask = []
	i = 0
	for block in model.feature.modules():
		if isinstance(block, nn.MaxPool2d):
			cfg.append('M')
		elif isinstance(block,nn.AvgPool2d):
			cfg.append('A')
		else:
			if isinstance(block,Res_block):
				m = block.bn_slim
				if count == 1:
					weight_copy = m.weight.data.abs().clone()
					mask = weight_copy.gt(-1).float()

					if block.left[0].stride[0]==1:
						cfg.append(str(int(torch.sum(mask)))+'r')
					else:
						cfg.append(str(int(torch.sum(mask)))+'r'+str(block.left[0].stride[0]))
					cfg_mask.append(mask.clone())
				else:
					weight_copy = m.weight.data.abs().clone()
					mask = weight_copy.gt(thre_list[i]).float()
					if block.left[0].stride[0]==1:
						cfg.append(str(int(torch.sum(mask)))+'r')
					else:
						cfg.append(str(int(torch.sum(mask)))+'r'+str(block.left[0].stride[0]))
					cfg_mask.append(mask.clone())
				i = i + 1
				count = count - 1
			elif isinstance(block,Basic_block):
				m = block.bn
				if count == 1:
					weight_copy = m.weight.data.abs().clone()
					mask = weight_copy.gt(-1).float()
					cfg.append(int(torch.sum(mask)))
					cfg_mask.append(mask.clone())
				else:
					weight_copy = m.weight.data.abs().clone()
					mask = weight_copy.gt(thre_list[i]).float()

					cfg.append(int(torch.sum(mask)))
					cfg_mask.append(mask.clone())
				i = i + 1
				count = count - 1
	print(cfg)
	res_new = ResNet(cfg,cfg_before_slim=cfg_before_slim,dataset=model.dataset)
	source_feature = model.feature

	BN_id = 0
	start_mask = torch.ones(64)   #64 is one super-parameter,it can be change in VGG class
	end_mask = cfg_mask[BN_id]
	for s,t in zip(source_feature.children(),res_new.feature.children()):
		if isinstance(s,nn.MaxPool2d):
			continue
		elif isinstance(s,nn.AvgPool2d):
			continue
		elif isinstance(s,Res_block):
			idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
			idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))

			m_s = s.left[0]
			m_t = t.left[0]
			m_t.weight.data = m_s.weight.data[:,idx0.tolist(),:,:].clone()

			t.left[1] = s.left[1]

			m_s = s.left[3]
			m_t = t.left[3]
			m_t.weight.data = m_s.weight.data[idx1.tolist(),:,:,:].clone()

			
			m_s = s.shortcut[0]
			m_t = t.shortcut[0]
			m_t.weight.data = torch.zeros([len(idx1.tolist()),len(idx0.tolist()),1,1]).to(m_s.weight.data.device)
			w = m_s.weight.data[:,idx0.tolist(),:,:].clone()
			m_t.weight.data += w[idx1.tolist(),:,:,:].clone()
			
			
				

			m_s = s.bn_slim
			m_t = t.bn_slim
			m_t.weight.data = m_s.weight.data[idx1.tolist()].clone()
			m_t.bias.data = m_s.bias.data[idx1.tolist()].clone()
			m_t.running_mean = m_s.running_mean[idx1.tolist()].clone()
			m_t.running_var = m_s.running_var[idx1.tolist()].clone()


			BN_id = BN_id + 1
			if BN_id<len(cfg_mask):
				start_mask = end_mask
				end_mask = cfg_mask[BN_id]

		elif isinstance(s,Basic_block):
			m_s = s.bn
			m_t = t.bn
			idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
			m_t.weight.data = m_s.weight.data[idx1.tolist()].clone()
			m_t.bias.data = m_s.bias.data[idx1.tolist()].clone()
			m_t.running_mean = m_s.running_mean[idx1.tolist()].clone()
			m_t.running_var = m_s.running_var[idx1.tolist()].clone()
			
			m_s = s.conv
			m_t = t.conv

			idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
			idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
			w = m_s.weight.data[:, idx0.tolist(), :, :].clone()
			w = w[idx1.tolist(), :, :, :].clone()
			m_t.weight.data = w.clone()

			BN_id = BN_id + 1
			if BN_id<len(cfg_mask):
				start_mask = end_mask
				end_mask = cfg_mask[BN_id]

	res_new.conv1 = model.conv1
	res_new.bn1 = model.bn1
	res_new.classifier = model.classifier
	return res_new

def pruning_by_action_list_mobile(model,action_list):
	total = 0
	count = 0
	thre_list = []
	for block in model.feature.modules():
		if isinstance(block,Mobile_block):
			m = block.bn2
			total = m.weight.data.shape[0]
			bn = m.weight.data.abs()
			y, i = torch.sort(bn)
			thre_index = int(total * action_list[count])
			if thre_index <= 0 :
				thre = -1
			else:
				thre = y[thre_index]
			thre_list.append(thre)
			count = count + 1
		elif isinstance(block,Basic_block):
			total = block.bn.weight.shape[0]
			bn = block.bn.weight.data.abs()
			y, i = torch.sort(bn)
			thre_index = int(total * (action_list[count]))
			if thre_index <= 0 :
				thre = -1
			else:
				thre = y[thre_index]
			thre_list.append(thre)
			count = count + 1
	cfg = []
	cfg_mask = []
	i = 0
	for block in model.feature.modules():
		if isinstance(block, nn.MaxPool2d):
			cfg.append('M')
		elif isinstance(block,nn.AvgPool2d):
			cfg.append('A')
		elif isinstance(block,Mobile_block):
			m = block.bn2
			if count == 1:
				weight_copy = m.weight.data.abs().clone()
				mask = weight_copy.gt(-1).float()
				cfg.append(int(torch.sum(mask)))
				cfg_mask.append(mask.clone())
			else:
				weight_copy = m.weight.data.abs().clone()
				mask = weight_copy.gt(thre_list[i]).float()
				cfg.append(int(torch.sum(mask)))
				cfg_mask.append(mask.clone())
			count = count - 1
			i = i + 1
		elif isinstance(block,Basic_block):
			for m in block.modules():
				if isinstance(m,nn.BatchNorm2d):
					if count == 1:
						weight_copy = m.weight.data.abs().clone()
						mask = weight_copy.gt(-1).float()
						cfg.append(int(torch.sum(mask)))
						cfg_mask.append(mask.clone())
					else:
						weight_copy = m.weight.data.abs().clone()
						mask = weight_copy.gt(thre_list[i]).float()
						cfg.append(int(torch.sum(mask)))
						cfg_mask.append(mask.clone())
					count = count - 1
					i = i + 1
	print(cfg)
	mobile_new = MobileNet(cfg,dataset = model.model_info()[0])
	source_feature = model.feature

	BN_id = 0
	start_mask = torch.ones(64)   #64 is one super-parameter,it can be change in VGG class
	end_mask = cfg_mask[BN_id]

	for s,t in zip(source_feature.children(),mobile_new.feature.children()):
		if isinstance(s,nn.MaxPool2d):
			continue
		elif isinstance(s,nn.AvgPool2d):
			continue
		elif isinstance(s,Mobile_block):
			idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
			idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))

			m_s = s.conv1
			m_t = t.conv1
			m_t.weight.data = m_s.weight.data[idx0.tolist(),:,:,:].clone()

			m_s = s.bn1
			m_t = t.bn1
			m_t.weight.data = m_s.weight.data[idx0.tolist()].clone()
			m_t.bias.data = m_s.bias.data[idx0.tolist()].clone()
			m_t.running_mean = m_s.running_mean[idx0.tolist()].clone()
			m_t.running_var = m_s.running_var[idx0.tolist()].clone()

			m_s = s.conv2
			m_t = t.conv2
			w = m_s.weight.data[:,idx0.tolist(),:,:].clone()
			m_t.weight.data = w[idx1.tolist(),:,:,:].clone()

			m_s = s.bn2
			m_t = t.bn2
			m_t.weight.data = m_s.weight.data[idx1.tolist()].clone()
			m_t.bias.data = m_s.bias.data[idx1.tolist()].clone()
			m_t.running_mean = m_s.running_mean[idx1.tolist()].clone()
			m_t.running_var = m_s.running_var[idx1.tolist()].clone()

			BN_id = BN_id + 1
			if BN_id<len(cfg_mask):
				start_mask = end_mask
				end_mask = cfg_mask[BN_id]

		elif isinstance(s,Basic_block):
			idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
			idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
			
			m_s = s.conv
			m_t = t.conv
			w = m_s.weight.data[:, idx0.tolist(), :, :].clone()
			w = w[idx1.tolist(), :, :, :].clone()
			m_t.weight.data = w.clone()

			m_s = s.bn
			m_t = t.bn
			m_t.weight.data = m_s.weight.data[idx1.tolist()].clone()
			m_t.bias.data = m_s.bias.data[idx1.tolist()].clone()
			m_t.running_mean = m_s.running_mean[idx1.tolist()].clone()
			m_t.running_var = m_s.running_var[idx1.tolist()].clone()

			BN_id = BN_id + 1
			if BN_id<len(cfg_mask):
				start_mask = end_mask
				end_mask = cfg_mask[BN_id]

	mobile_new.conv1 = model.conv1
	mobile_new.bn1 = model.bn1
	mobile_new.classifier = model.classifier
	return mobile_new

def pruning_by_global_rate(model,pruning_rate):
	model_class = model.model_info()[-1]
	if model_class=='vgg':
		return pruning_by_global_rate_vgg(model,pruning_rate)
	elif model_class=='res':
		return pruning_by_global_rate_res(model,pruning_rate)
	elif model_class=='mobile':
		return pruning_by_global_rate_mobile(model,pruning_rate)

def pruning_by_global_rate_vgg(model,pruning_rate):
	total = 0
	count = 0
	for block in model.feature.modules():
		if isinstance(block,Basic_block):
			m = block.bn		
			total += m.weight.data.shape[0]
			count = count + 1
	bn = torch.zeros(total)
	index = 0
	for block in model.feature.modules():
		if isinstance(block,Basic_block):
			m = block.bn
			size = m.weight.data.shape[0]
			bn[index:(index+size)] = m.weight.data.abs().clone()
			index += size
	y, i = torch.sort(bn)
	thre_index = int(total * pruning_rate)
	thre = y[thre_index]
			
	cfg = []
	cfg_mask = []
	for block in model.feature.modules():
		if isinstance(block,Basic_block):
			m = block.bn
			if count == 1:
				weight_copy = m.weight.data.abs().clone()
				mask = weight_copy.gt(-1).float()
				cfg.append(int(torch.sum(mask)))
				cfg_mask.append(mask.clone())
			else:
				weight_copy = m.weight.data.abs().clone()
				mask = weight_copy.gt(thre).float()
				cfg.append(int(torch.sum(mask)))
				cfg_mask.append(mask.clone())
			count = count - 1
		elif isinstance(block, nn.MaxPool2d):
			cfg.append('M')
		elif isinstance(block,nn.AvgPool2d):
			cfg.append('A')
	print(cfg)
	vgg_new = VGG(cfg,dataset = model.model_info()[0])
	source_feature = model.feature

	BN_id = 0
	start_mask = torch.ones(64)   #64 is one super-parameter,it can be change in VGG class
	end_mask = cfg_mask[BN_id]
	for s,t in zip(source_feature.modules(),vgg_new.feature.modules()):
		if isinstance(s,Basic_block):
			idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
			idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))

			m_s = s.conv
			m_t = t.conv
			w = m_s.weight.data[:, idx0.tolist(), :, :].clone()
			w = w[idx1.tolist(), :, :, :].clone()
			m_t.weight.data = w.clone()

			m_s = s.bn
			m_t = t.bn
			m_t.weight.data = m_s.weight.data[idx1.tolist()].clone()
			m_t.bias.data = m_s.bias.data[idx1.tolist()].clone()
			m_t.running_mean = m_s.running_mean[idx1.tolist()].clone()
			m_t.running_var = m_s.running_var[idx1.tolist()].clone()

			BN_id += 1
			start_mask = end_mask.clone()
			if not BN_id == len(cfg_mask):
				end_mask = cfg_mask[BN_id]

	vgg_new.conv1 = model.conv1
	vgg_new.bn1 = model.bn1
	vgg_new.classifier = model.classifier
	return vgg_new

def pruning_by_global_rate_res(model,pruning_rate):
	total = 0
	count = 0
	if model.model_info()[0] == None:
		cfg_before_slim = model.model_info()[2]
	else:
		cfg_before_slim = model.model_info()[0]
	for block in model.feature.modules():
		if isinstance(block,Basic_block):
			m = block.bn
			total += m.weight.data.shape[0]
			count +=1
		elif isinstance(block,Res_block):
			m = block.bn_slim
			total += m.weight.data.shape[0]
			count +=1
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
	thre_index = int(total * pruning_rate)
	thre = y[thre_index]
	cfg = []
	cfg_mask = []
	for block in model.feature.modules():
		if isinstance(block, nn.MaxPool2d):
			cfg.append('M')
		elif isinstance(block,nn.AvgPool2d):
			cfg.append('A')
		elif isinstance(block,Res_block):
			m = block.bn_slim
			if count == 1:
				weight_copy = m.weight.data.abs().clone()
				mask = weight_copy.gt(-1).float()
				if block.left[0].stride[0]==1:
					cfg.append(str(int(torch.sum(mask)))+'r')
					cfg_mask.append(mask.clone())
				else:
					cfg.append(str(int(torch.sum(mask)))+'r'+str(block.left[0].stride[0]))
					cfg_mask.append(mask.clone())
			else:
				weight_copy = m.weight.data.abs().clone()
				mask = weight_copy.gt(thre).float()
				if block.left[0].stride[0]==1:
					cfg.append(str(int(torch.sum(mask)))+'r')
					cfg_mask.append(mask.clone())
				else:
					cfg.append(str(int(torch.sum(mask)))+'r'+str(block.left[0].stride[0]))
					cfg_mask.append(mask.clone())
			count = count - 1
		elif isinstance(block,Basic_block):
			m = block.bn
			if count == 1:
				weight_copy = m.weight.data.abs().clone()
				mask = weight_copy.gt(-1).float()
				cfg.append(int(torch.sum(mask)))
				cfg_mask.append(mask.clone())
			else:
				weight_copy = m.weight.data.abs().clone()
				mask = weight_copy.gt(thre).float()
				cfg.append(int(torch.sum(mask)))
				cfg_mask.append(mask.clone())
			count = count - 1
	print(cfg)
	res_new = ResNet(cfg,cfg_before_slim=cfg_before_slim,dataset=model.dataset)
	source_feature = model.feature
	BN_id = 0
	start_mask = torch.ones(64)   #64 is one super-parameter,it can be change in VGG class
	end_mask = cfg_mask[BN_id]
	for s,t in zip(source_feature.modules(),res_new.feature.modules()):

		if isinstance(s,nn.MaxPool2d):
			continue
		elif isinstance(s,nn.AvgPool2d):
			continue
		elif isinstance(s,Res_block):
			idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
			idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
			m_s = s.left[0]
			m_t = t.left[0]
			#m_t = nn.Conv2d(m_s.weight.data.shape[1],m_s.weight.data.shape[0],kernel_size=3,stride=1,padding=1,bias=False)
			m_t.weight.data = m_s.weight.data[:,idx0.tolist(),:,:].clone()

			t.left[1] = s.left[1]

			m_s = s.left[3]
			m_t = t.left[3]
			m_t.weight.data = m_s.weight.data[idx1.tolist(),:,:,:].clone()

			
			m_s = s.shortcut[0]
			m_t = t.shortcut[0]
			m_t.weight.data = torch.zeros([len(idx1.tolist()),len(idx0.tolist()),1,1]).to(m_s.weight.data.device)
			w = m_s.weight.data[:,idx0.tolist(),:,:].clone()
			m_t.weight.data += w[idx1.tolist(),:,:,:].clone()
			

			m_s = s.bn_slim
			m_t = t.bn_slim
			m_t.weight.data = m_s.weight.data[idx1.tolist()].clone()
			m_t.bias.data = m_s.bias.data[idx1.tolist()].clone()
			m_t.running_mean = m_s.running_mean[idx1.tolist()].clone()
			m_t.running_var = m_s.running_var[idx1.tolist()].clone()


			BN_id = BN_id + 1
			if BN_id<len(cfg_mask):
				start_mask = end_mask
				end_mask = cfg_mask[BN_id]

		elif isinstance(s,Basic_block):
			m_s = s.bn
			m_t = t.bn
			idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
			m_t.weight.data = m_s.weight.data[idx1.tolist()].clone()
			m_t.bias.data = m_s.bias.data[idx1.tolist()].clone()
			m_t.running_mean = m_s.running_mean[idx1.tolist()].clone()
			m_t.running_var = m_s.running_var[idx1.tolist()].clone()
			
			m_s = s.conv
			m_t = t.conv

			idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
			idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
			w = m_s.weight.data[:, idx0.tolist(), :, :].clone()
			w = w[idx1.tolist(), :, :, :].clone()
			m_t.weight.data = w.clone()

			BN_id = BN_id + 1
			if BN_id<len(cfg_mask):
				start_mask = end_mask
				end_mask = cfg_mask[BN_id]
			
	res_new.conv1 = model.conv1
	res_new.bn1 = model.bn1
	res_new.classifier = model.classifier

	return res_new

def pruning_by_global_rate_mobile(model,pruning_rate):
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
	thre_index = int(total * pruning_rate)
	thre = y[thre_index]
	cfg = []
	cfg_mask = []
	for block in model.feature.modules():
		if isinstance(block, nn.MaxPool2d):
			cfg.append('M')
		elif isinstance(block,nn.AvgPool2d):
			cfg.append('A')
		elif isinstance(block, Mobile_block):
			m = block.bn2
			if count == 1:
				weight_copy = m.weight.data.abs().clone()
				mask = weight_copy.gt(-1).float()
				cfg.append(int(torch.sum(mask)))
				cfg_mask.append(mask.clone())
			else:
				weight_copy = m.weight.data.abs().clone()
				mask = weight_copy.gt(thre).float()
				cfg.append(int(torch.sum(mask)))
				cfg_mask.append(mask.clone())
			count = count - 1
	print(cfg)
	mobile_new = MobileNet(cfg,dataset = model.model_info()[0])
	source_feature = model.feature

	BN_id = 0
	start_mask = torch.ones(64)   #64 is one super-parameter,it can be change in VGG class
	end_mask = cfg_mask[BN_id]
	for s,t in zip(source_feature.modules(),mobile_new.feature.modules()):
		if isinstance(s,Mobile_block):
			idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
			idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))

			m_s = s.conv1
			m_t = t.conv1
			m_t.weight.data = m_s.weight.data[idx0.tolist(),:,:,:].clone()

			m_s = s.bn1
			m_t = t.bn1
			m_t.weight.data = m_s.weight.data[idx0.tolist()].clone()
			m_t.bias.data = m_s.bias.data[idx0.tolist()].clone()
			m_t.running_mean = m_s.running_mean[idx0.tolist()].clone()
			m_t.running_var = m_s.running_var[idx0.tolist()].clone()

			m_s = s.conv2
			m_t = t.conv2
			w = m_s.weight.data[:,idx0.tolist(),:,:].clone()
			m_t.weight.data = w[idx1.tolist(),:,:,:].clone()

			m_s = s.bn2
			m_t = t.bn2
			m_t.weight.data = m_s.weight.data[idx1.tolist()].clone()
			m_t.bias.data = m_s.bias.data[idx1.tolist()].clone()
			m_t.running_mean = m_s.running_mean[idx1.tolist()].clone()
			m_t.running_var = m_s.running_var[idx1.tolist()].clone()
			BN_id = BN_id + 1
			if BN_id<len(cfg_mask):
				start_mask = end_mask
				end_mask = cfg_mask[BN_id]

	mobile_new.conv1 = model.conv1
	mobile_new.bn1 = model.bn1
	mobile_new.classifier = model.classifier
	return mobile_new
