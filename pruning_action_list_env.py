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
from copy import deepcopy

from pruning import *
from model import *
from train import *
from utils import *
class Pruning_Env():
	def __init__(self,model,test_loader,criterion,pruning_rate,left_bound,right_bound,device):
		self.prunable_layer_types = [Basic_block, Res_block,Mobile_block]
		self.model = model
		self.test_loader = test_loader
		self.criterion = criterion
		self.pruning_rate = pruning_rate
		self.device = device
		self.n_calibration_batches = 60
		self.n_points_per_layer = 10
		self.channel_round = 8

		self.model_type = self.model.model_info()[-1]
		self.cfg_old = self.model.model_info()[-2]

		self.left_bound = left_bound
		self.right_bound = right_bound
		self.action_list = []
		if model.dataset == 'cifar10' or model.dataset == 'cifar100':
			self.hw = 32
		elif model.dataset == 'imagenet':
			self.hw = 224

		#assert self.pruning_rate > self.lbound, 'Error! You can make achieve pruning_rate smaller than lbound!'
		
		self._build_index()
		self._extract_layer_information()
		self.org_flops = sum(self.flops_list)
		self.n_prunable_layer = len(self.prunable_idx)
		self._build_state_embedding()	
		self.reset()
		_,self.org_acc,_ = test(self.model,self.test_loader,self.criterion,self.device)
		print('=> original acc: {:.3f}%'.format(self.org_acc))
		self.org_model_size = sum(self.wsize_list)
		print('=> original weight size: {:.4f} M param'.format(self.org_model_size * 1. / 1e6))
		self.org_flops = sum(self.flops_list)
		self.pruning_comp = self.pruning_rate * self.org_flops
		self.org_w_size = sum(self.wsize_list)
		print('=> FLOPs:')
		print([self.layer_info_dict[idx]['flops']/1e6 for idx in sorted(self.layer_info_dict.keys())])
		print('=> original FLOPs: {:.4f} M'.format(self.org_flops * 1. / 1e6))
	
	def _build_index(self):
		self.prunable_idx = []
		self.prunable_ops = []
		self.layer_type_dict = {}
		self.strategy_dict = {}
		self.org_channels = []
		# build index and the min strategy dict
		for i, m in enumerate(self.model.modules()):
			if type(m) in self.prunable_layer_types:  
				self.prunable_idx.append(i)
				self.prunable_ops.append(m)
				self.layer_type_dict[i] = type(m)
				if type(m) == Basic_block:
					self.org_channels.append(m.conv.weight.data.shape[1])
				elif type(m) == Res_block:
					self.org_channels.append(m.shortcut[0].weight.data.shape[1])
				elif type(m) == Mobile_block:
					self.org_channels.append(m.conv1.weight.data.shape[1])
				


	
	

		print('=> Prunable layer idx: {}'.format(self.prunable_idx))


		# added for supporting residual connections during pruning

		
		self.best_reward = -math.inf
		self.best_strategy = None
		self.best_d_prime_list = None
		
	def _build_state_embedding(self):
		# build the static part of the state embedding
		layer_embedding = []
		module_list = list(self.model.modules())
		for i, ind in enumerate(self.prunable_idx):
			m = module_list[ind]
			this_state = []
			this_state.append(i)  # index
			if type(m) == Basic_block:
				this_state.append(m.conv.weight.data.shape[1])
				this_state.append(m.conv.weight.data.shape[0])
				this_state.append(1.)
				this_state.append(np.prod(m.conv.weight.size()))
			elif type(m) == Res_block:
				this_state.append(m.shortcut[0].weight.data.shape[1])
				this_state.append(m.shortcut[0].weight.data.shape[0])
				this_state.append(m.shortcut[0].stride[0])
				this_state.append(np.prod(m.left[0].weight.size())*np.prod(m.left[3].weight.size()))
			elif type(m) == Mobile_block:
				this_state.append(m.conv1.weight.data.shape[1])
				this_state.append(m.conv2.weight.data.shape[0])
				this_state.append(1.)
				this_state.append(np.prod(m.conv1.weight.size())*np.prod(m.conv2.weight.size()))

			# this 3 features need to be changed later
			this_state.append(0.)  # reduced
			this_state.append(0.)  # rest
			this_state.append(0.)  # a_{t-1}
			layer_embedding.append(np.array(this_state))

		# normalize the state
		layer_embedding = np.array(layer_embedding, 'float')
		print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
		assert len(layer_embedding.shape) == 2, layer_embedding.shape
		for i in range(layer_embedding.shape[1]):
			fmin = min(layer_embedding[:, i])
			fmax = max(layer_embedding[:, i])
			if fmax - fmin > 0:
				layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

		self.layer_embedding = layer_embedding

	def reset(self):
		self.true_action_list = []
		self.cur_ind = 0
		self.d_prime_list = []
		# reset layer embeddings
		self.layer_embedding[:, -1] = 1.
		self.layer_embedding[:, -2] = 0.
		self.layer_embedding[:, -3] = 0.
		obs = self.layer_embedding[0].copy()
		obs[-2] = sum(self.flops_list[1:]) * 1. / self.org_flops
		self.extract_time = 0
		self.fit_time = 0
		self.val_time = 0
		self.action_list = [0.]*self.layer_embedding.shape[0]
		return obs

	def _extract_layer_information(self):
		self.wsize_list = []
		self.flops_list = []
		self.layer_info_dict = dict()
		input_hw = self.hw
		i = 0
		for m in self.model.modules():
			if isinstance(m,nn.MaxPool2d):
				input_hw = input_hw // 2
			elif isinstance(m,nn.AvgPool2d):
				input_hw = input_hw // 4
			elif type(m) in self.prunable_layer_types:
				in_channels = m.in_channels
				macs,params=measure_for_block(m,input_hw,device=self.device)
				self.wsize_list.append(params)
				self.flops_list.append(macs)
				idx = self.prunable_idx[i]
				self.layer_info_dict[idx] = dict()
				self.layer_info_dict[idx]['params'] = params
				self.layer_info_dict[idx]['flops'] = macs
				i = i + 1
				if type(m) == Res_block:
					if m.stride == 2:
						input_hw = input_hw//2

	def step(self, action):
		action = self._action_wall(action)  # percentage to preserve
		self.action_list[self.cur_ind]=action
		# all the actions are made
		if self._is_final_layer():
			model_new=pruning_by_action_list(self.model,self.action_list)
			_,acc,_ = test(model_new,self.test_loader,self.criterion,self.device)
			pruning_ratio = self._cur_pruning_flops() * 1. / self.org_flops
			info_set = {'pruning_ratio': pruning_ratio, 'accuracy': acc, 'action_list': self.action_list.copy()}
			#reward = -0.01*(100.0-acc)*np.log(self.org_flops-self._cur_pruning_flops())
			reward = 0.01*acc
			if reward > self.best_reward:
				self.best_action_list = self.action_list
				self.best_reward = reward
				print('New best reward: {:.4f}, acc: {:.4f}'.format(self.best_reward, acc))
				print('New best action_list: {}'.format(self.action_list))


			obs = self.layer_embedding[self.cur_ind+1, :].copy()  # actually the same as the last state
			done = True
			return obs, reward, done, info_set,action
		info_set = None
		reward = 0
		done = False
		self.cur_ind += 1  # the index of next layer
		# build next state (in-place modify)
		self.layer_embedding[self.cur_ind][-3] = self._cur_pruning_flops() * 1. / self.org_flops  # reduced
		self.layer_embedding[self.cur_ind][-2] = sum(self.flops_list[self.cur_ind + 1:]) * 1. / self.org_flops  # rest
		self.layer_embedding[self.cur_ind][-1] = self.action_list[self.cur_ind-1]
		obs = self.layer_embedding[self.cur_ind, :].copy()

		return obs, reward, done, info_set,action

	def _is_final_layer(self):
		return self.cur_ind == len(self.prunable_idx) - 2


	def _action_wall(self, action):
		action = float(action)
		action = max(action,0)
		action = min(action,1)
		other_comp = 0.
		this_comp_min = 0.
		this_comp_max = 0.
		max_left_comp = 0.
		min_left_comp = 0.
		for i, idx in enumerate(self.prunable_idx):
			flop = self.layer_info_dict[idx]['flops']
			if i == self.cur_ind:
				if i == 0:
					this_comp_min = flop*1.0
					this_comp_max = flop*1.0
				else:
					this_comp_min = flop *1.0*max(self.action_list[i-1],0.01)
					this_comp_max = flop *1.0*max(self.action_list[i-1],0.01)
			elif i == 0:
				other_comp += flop*1.0*self.action_list[i]
			elif i <= self.cur_ind - 1: 
				other_comp += flop * (1.0-(1-self.action_list[i])*(1.0-self.action_list[i-1]))
			elif i == self.cur_ind + 1:
				if i == len(self.prunable_idx) - 1:
					pass
				else:
					this_comp_min += flop*self.left_bound[i]*1.0
					this_comp_max += flop*self.right_bound[i]*1.0
			elif i < len(self.prunable_idx)-1:
				max_left_comp+=flop*(1-(1-self.right_bound[i-1])*(1-self.right_bound[i]))
				min_left_comp+=flop*(1-(1-self.left_bound[i-1])*(1-self.left_bound[i]))
			elif i == len(self.prunable_idx) -1 :
				max_left_comp+=flop*(1-(1-self.right_bound[i-1])*1.0)
				min_left_comp+=flop*(1-(1-self.left_bound[i-1])*1.0)	

		min_pruning_rate = (self.pruning_comp - other_comp - max_left_comp) * 1. / this_comp_max
		max_pruning_rate = (self.pruning_comp - other_comp - min_left_comp) * 1. / this_comp_min
		action = np.minimum(action, max_pruning_rate)
		action = np.maximum(action,min_pruning_rate)
		action = max(action,self.left_bound[self.cur_ind])
		action = min(action,self.right_bound[self.cur_ind])
		#print(min_pruning_rate)
		#print(action)
		#print(max_pruning_rate)
		return action

	def _cur_pruning_flops(self):
		flops = 0
		for i, idx in enumerate(self.prunable_idx):
			if i > 0:
				c = self.action_list[i-1]
			else:
				c = 1.0
			n = self.action_list[i]
			flops += self.layer_info_dict[idx]['flops'] *(1-(1-c)*(1-n))
		return flops

	def return_action_list(self):
		action_list = []
		for i in range(len(self.flops_list)):
			action_list.append(0.)
		return action_list

	def action_wall(self,action_list,cur_ind):
		other_comp = 0.
		this_comp_min = 0.
		this_comp_max = 0.
		max_left_comp = 0.
		min_left_comp = 0.
		for i, idx in enumerate(self.prunable_idx):
			flop = self.layer_info_dict[idx]['flops']
			if i == cur_ind:
				if i == 0:
					this_comp_min = flop*1.0
					this_comp_max = flop*1.0
				else:
					this_comp_min = flop *1.0*max(action_list[i-1],0.01)
					this_comp_max = flop *1.0*max(action_list[i-1],0.01)
			elif i == 0:
				other_comp += flop*1.0*action_list[i]
			elif i <= cur_ind - 1: 
				other_comp += flop * (1.0-(1-action_list[i])*(1.0-action_list[i-1]))
			elif i == cur_ind + 1:
				if i == len(self.prunable_idx) - 1:
					pass
				else:
					this_comp_min += flop*self.left_bound[i]*1.0
					this_comp_max += flop*self.right_bound[i]*1.0
			elif i < len(self.prunable_idx)-1:
				max_left_comp+=flop*(1-(1-self.right_bound[i-1])*(1-self.right_bound[i]))
				min_left_comp+=flop*(1-(1-self.left_bound[i-1])*(1-self.left_bound[i]))
			elif i == len(self.prunable_idx) -1 :
				max_left_comp+=flop*(1-(1-self.right_bound[i-1])*1.0)
				min_left_comp+=flop*(1-(1-self.left_bound[i-1])*1.0)	

		min_pruning_rate = (self.pruning_comp - other_comp - max_left_comp) * 1. / this_comp_max
		max_pruning_rate = (self.pruning_comp - other_comp - min_left_comp) * 1. / this_comp_min
		
		return min_pruning_rate,max_pruning_rate