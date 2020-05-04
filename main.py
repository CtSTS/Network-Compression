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
from DDPG_agent import *
from pruning_action_list_env import *
from train import *
from DDPG_memory import *
from model import *
from pruning import *
from utils import *
import argparse
from tensorboardX import SummaryWriter
from kmeans import *
import logging
import time
from torch.optim.lr_scheduler import _LRScheduler
def parse_args():
	parser = argparse.ArgumentParser(description='model compression')

	parser.add_argument('--model_cfg', default=None, type=str, help='cfg for build one new model( "[64,64,"M",128,128,"A"]")')
	parser.add_argument('--model_type', default=None, type=str, help='type for build one new model(vgg/res/mobile)')
	parser.add_argument('--classifier_type', default='simple', type=str, help='model classifier_type')
	parser.add_argument('--model_path', default=None, type=str, help='path for load one old model')
	parser.add_argument('--train', default='yes', type=str, help='model need train again or not')
	
	parser.add_argument('--pruning_rate', default=0.5, type=float, help='pruning_rate of the model')
	parser.add_argument('--lbound', default=0.0, type=float, help='minimum pruning rate for a layer')
	parser.add_argument('--rbound', default=0.95, type=float, help='maximum pruning rate for a layer')
	parser.add_argument('--pruning_method', default='global', type=str, help='method to prune (global/search/DDPG)')
	parser.add_argument('--alpha', default=0.3, type=float, help='param to balance accuracy and pruning rate ')
	#search
	parser.add_argument('--pruning_time', default=1, type=int, help='time for pruning and retrain')
	# ddpg
	parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
	parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
	parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for critic')
	parser.add_argument('--lr_a', default=1e-4, type=float, help='learning rate for actor')
	parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
	parser.add_argument('--train_episode', default=800, type=int, help='train iters each timestep')
	parser.add_argument('--ddpg_path', default='./DDPG_model', type=str, help='DDPG model save path')
	parser.add_argument('--nb_states', default=8, type=int, help='number for state length')
	parser.add_argument('--nb_actions', default=1, type=int, help='number for action length')
	
	# datasets
	parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to use (cifar10/cifar100/imagenet)')
	parser.add_argument('--data_root', default='./data', type=str, help='dataset path')
	parser.add_argument('--n_gpu', default=1, type=int, help='number of gpu to use')
	parser.add_argument('--n_worker', default=2, type=int, help='number of data loader worker')
	parser.add_argument('--data_bsize', default=64, type=int, help='number of data batch size')
	
	# training
	parser.add_argument('--lr', default=0.1, type=float, help='learning rate for train')
	parser.add_argument('--s', default=0.01, type=float, help='slimming rate for train')
	parser.add_argument('--kd_T', default=0.5, type=float, help='T for kd_train')
	parser.add_argument('--kd_alpha', default=0.5, type=float, help='alpha for kd_train')
	parser.add_argument('--kd_beta', default=0.01, type=float, help='beta for kd_train')
	parser.add_argument('--device', default='cuda', type=str, help='device for train')
	parser.add_argument('--warm', default=0, type=int, help='warm for train')
	parser.add_argument('--milestones', default='[60,120,160,200]', type=str, help='learning rate for train')
	# k-means
	parser.add_argument('--bits', default=8, type=int, help='bits for k-means compression')

	parser.add_argument('--output_path', default='./output_model', type=str, help='dir for output model')
	parser.add_argument('--output_name', default='output', type=str, help='name for outpur model')
	return parser.parse_args()

def pruning_global_rate(model,test_loader,criterion,pruning_rate,device='cuda'):
	model_new=pruning_by_global_rate(model,pruning_rate)
	return model_new

def pruning_search(model,test_loader,criterion,pruning_rate,alpha = 0.3,lbound=0,rbound=0.8,device='cuda'):
	left_bound = get_left_bound(model,pruning_rate/3.0)
	right_bound = []
	for i in range(len(left_bound)):
		right_bound.append(max(min(rbound,float(1.0*rbound/len(left_bound)*(i+1))),pruning_rate/2.0))
	env = Pruning_Env(model,test_loader,criterion,pruning_rate,left_bound,right_bound,device)
	action_list = env.return_action_list()
	best_action_list = action_list
	model_pruning_global = pruning_by_global_rate(model,pruning_rate)
	_,acc_bound,_ = test(model_pruning_global.to(device),test_loader,criterion,device)
	acc_bound = acc_bound - 2.0
	print("acc_bound for layers:{:.2f}".format(acc_bound))
	print("left_bound for layers:{}".format(left_bound))
	print("right_bound for layers:{}".format(right_bound))
	for i in range(len(action_list)-1):
		print("pruning for the {} layer:".format(i))
		min_pruning_ratio,max_pruning_ratio = env.action_wall(action_list,i)
		now_left_bound = max(min_pruning_ratio,left_bound[i])
		now_right_bound = min(max_pruning_ratio,right_bound[i])
		if now_left_bound > now_right_bound:
			action_list[i] = now_right_bound
			continue
		best_reward = 0.
		best_action = 0.
		print("acc_bound:{:.2f}".format(acc_bound))
		print("left_bound:{:.3f}".format(now_left_bound))
		print("right_bound:{:.3f}".format(now_right_bound))
		for action in range(int(now_left_bound*100),int(now_right_bound*100),5):
			print("pruning for {} layer  jump length: 5".format(i))
			action_list[i] = action*0.01
			min_pruning_ratio,max_pruning_ratio = env.action_wall(action_list,i+1)
			now_left_bound_next = max(min_pruning_ratio,left_bound[i+1])
			now_right_bound_next = min(max_pruning_ratio,right_bound[i+1])
			
				
			best_reward_next = 0.
			best_action_next = now_left_bound_next
			if now_left_bound_next >= now_right_bound_next:
				best_action_next = now_right_bound_next
			for action_next in range(int(now_left_bound_next*100),int(now_right_bound_next*100),3):
				print("pruning for {} layer test i+1 layer acc jump length: 5".format(i))
				action_list[i+1] = action_next*0.01
				model_cut = pruning_by_action_list(model,action_list)
				_,acc,_ = test(model_cut.to(device),test_loader,criterion,device)
				if acc < acc_bound:
					break
				reward = acc + alpha * (action_list[i]+action_list[i+1]+action_list[i]*action_list[i+1])
				if reward > best_reward_next:
					best_reward_next = reward
					best_action_next = action_next*0.01
				if reward > best_reward:
					best_reward = reward
					best_action = action*0.01
			
			print("pruning for {} layer test i+1 layer acc jump length: 5 best_action_next:{}".format(i,best_action_next))
			for action_next in range(int(best_action_next*100-3),int(best_action_next*100+2)):
				print("pruning for {} layer test i+1 layer acc jump length: 1".format(i))
				if action_next*0.01<0 or action_next*0.01 >rbound:
					continue
				action_list[i+1] = action_next*0.01
				model_cut = pruning_by_action_list(model,action_list)
				_,acc,_ = test(model_cut.to(device),test_loader,criterion,device)
				if acc < acc_bound:
					break
				reward = acc + alpha * (action_list[i]+action_list[i+1]+action_list[i]*action_list[i+1])		
				if reward > best_reward_next:
					best_reward = reward
					best_action = action*0.01

		for action in range(int(best_action*100-3),int(best_action*100+2)):
			print("pruning for {} layer test jump length: 1".format(i))
			if action*0.01 < 0 or action*0.01 > rbound:
				continue
			action_list[i] = action*0.01
			min_pruning_ratio,max_pruning_ratio = env.action_wall(action_list,i+1)
			now_left_bound_next = max(min_pruning_ratio,left_bound[i+1])
			now_right_bound_next = min(max_pruning_ratio,right_bound[i+1])
			best_reward_next = 0.
			best_action_next = now_left_bound_next
			if now_left_bound_next >= now_right_bound_next:
				 best_action_next = now_right_bound_next
			for action_next in range(int(now_left_bound_next*100),int(now_right_bound_next*100),5):
				print("pruning for {} layer test i+1 layer acc jump length: 5".format(i))
				action_list[i+1] = action_next*0.01
				model_cut = pruning_by_action_list(model,action_list)
				_,acc,_ = test(model_cut.to(device),test_loader,criterion,device)
				if acc < acc_bound:
					break
				reward = acc + alpha * (action_list[i]+action_list[i+1]+action_list[i]*action_list[i+1])
				if reward > best_reward_next:
					best_reward_next = reward
					best_action_next = action_next*0.01
	
			print("pruning for {} layer test i+1 layer acc jump length: 1 best_action_next:{}".format(i,best_action_next))
			for action_next in range(int(best_action_next*100-3),int(best_action_next*100+3)):
				print("pruning for {} layer test i+1 layer acc jump length: 1".format(i))
				if action_next*0.01<0 or action_next*0.01 >rbound:
					continue
				action_list[i+1] = action_next*0.01
				model_cut = pruning_by_action_list(model,action_list)
				_,acc,_ = test(model_cut.to(device),test_loader,criterion,device)
				if acc < acc_bound:
					break
				reward = acc + alpha * (action_list[i]+action_list[i+1]+action_list[i]*action_list[i+1])
				if reward > best_reward:
					best_reward = reward
					best_action = action*0.01
		action_list[i] = best_action	
		action_list[i+1]= 0.
		print("After {} layer pruning action_list:{}".format(i,action_list))
	model_new = pruning_by_action_list(model,action_list)
	return model_new

def pruning_DDPG(model,test_loader,criterion,pruning_rate,num_episode,warmup,lbound=0,rbound=0.8,output='./',nb_states=8,nb_actions=1,hidden1=300,hidden2=300,lr_a=1e-4,lr_c=1e-4,device='cuda'):
	writer = SummaryWriter("./DDPG")
	left_bound = get_left_bound(model,pruning_rate/3.0)
	right_bound = []
	for i in range(len(left_bound)):
		left_bound[i] = float(left_bound[i].cpu().numpy())
		right_bound.append(rbound)
	print("left_bound:",left_bound)
	print("right_bound:",right_bound)
	agent = DDPG(warmup,nb_states, nb_actions,left_bound,right_bound,hidden1,hidden2,lr_a,lr_c)
	agent.is_training = True
	env = Pruning_Env(model,test_loader,criterion,pruning_rate,left_bound,right_bound,device)
	step = episode = episode_steps = 0
	episode_reward = 0.
	observation = None
	T = []  # trajectory
	while episode < num_episode:  # counting based on episode
		# reset if it is the start of episode
		if observation is None:
			observation = deepcopy(env.reset())
			agent.reset(observation)
		# agent pick action ...
		if episode <= warmup:
			action = agent.random_action()
			# action = sample_from_truncated_normal_distribution(lower=0., upper=1., mu=env.pruning_rate, sigma=0.5)
		else:
			action = agent.select_action(observation, episode=episode)	
		# env response with next_observation, reward, terminate_info

		observation2, reward, done, info,action = env.step(action)
		if (episode>warmup):
			writer.add_scalars('Acc_DDPG',{'Acc':100+reward},episode-warmup)
		observation2 = deepcopy(observation2)
		#print(observation)
		T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])
		# fix-length, never reach here
		# if max_episode_length and episode_steps >= max_episode_length - 1:
		#	 done = True
		# [optional] save intermideate model
		if episode % int(num_episode / 3) == 0:
			agent.save_model(output)
		# update
		step += 1
		episode_steps += 1
		episode_reward += reward
		observation = deepcopy(observation2)
		if done:  # end of episode
			print('#{}: episode_reward:{:.4f} acc: {:.4f}, ratio: {:.4f}'.format(episode,episode_reward,info['accuracy'],info['pruning_ratio']))
			final_reward = T[-1][0]
			#if final_reward > best_reward - 5.0:
			# print('final_reward: {}'.format(final_reward))
			# agent observe and update policy
			for i in range(len(T)):
				r_t, s_t, s_t1, a_t, done = T[i]
				agent.observe(final_reward, s_t, s_t1, a_t, done)
				if episode > warmup:
					agent.update_policy()
			#agent.memory.append(
			#	observation,
			#	agent.select_action(observation, episode=episode),
			#	0., False
			#)
			# reset
			observation = None
			episode_steps = 0
			episode_reward = 0.
			episode += 1
			T = []
	print("best_action_list:")
	print(env.best_action_list)
	model_new = pruning_by_action_list(model,env.best_action_list)
	return model_new




def prepare_datasets(args):
	logging.info("dataset: {}".format(args.dataset))
	if args.dataset == 'cifar10':
		transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),  
		transforms.RandomHorizontalFlip(),  
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.data_bsize, shuffle=True, num_workers=args.n_worker)  

		testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=args.data_bsize, shuffle=False, num_workers=args.n_worker)
		
	elif args.dataset == 'cifar100':
		
		transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(15),
		transforms.ToTensor(),
		transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)), #R,G,B
		])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
		])
		trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train) 
		testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.data_bsize, shuffle=True, num_workers=args.n_worker)  	
		test_loader = torch.utils.data.DataLoader(testset, batch_size=args.data_bsize, shuffle=False, num_workers=args.n_worker)
	else:
		raise RuntimeError("No such dataset for compression!")
	return test_loader,train_loader

def prepare_model(args):
	if args.model_cfg == None and args.model_path == None:
		raise RuntimeError("No model exist for compression!")
	if args.model_path == None:
		print(args.model_cfg)
		args.model_cfg = eval(args.model_cfg)
		print(args.model_cfg)
		if args.model_type == 'vgg':
			model = VGG(args.model_cfg,dataset = args.dataset,classifier_type=args.classifier_type)
		elif args.model_type == 'res':
			model = ResNet(args.model_cfg,dataset = args.dataset,classifier_type=args.classifier_type)
		elif args.model_type == 'mobile':
			model = MobileNet(args.model_cfg,dataset = args.dataset,classifier_type=args.classifier_type)
		logging.info("model_type: {}  model_cfg: {}".format(args.model_type,args.model_cfg))
	else:
		model = torch.load(args.model_path)
		logging.info("model_path: {}  model_type: {}  model_cfg: {}".format(args.model_path,model.model_info()[-1],model.model_info()[-2]))
	model = model.to(args.device)
	if args.n_gpu > 1:
		model = torch.nn.DataParallel(model, range(args.n_gpu))
	return model

class WarmUpLR(_LRScheduler):
	"""warmup_training learning rate scheduler
	Args:
		optimizer: optimzier(e.g. SGD)
		total_iters: totoal_iters of warmup phase
	"""
	def __init__(self, optimizer, total_iters, last_epoch=-1):
		
		self.total_iters = total_iters
		super().__init__(optimizer, last_epoch)

	def get_lr(self):
		"""we will use the first m batches, and set the learning
		rate to base_lr * m / total_iters
		"""
		return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def train_model(model,test_loader,train_loader,criterion,writer,args): 
	if args.train == 'no':
		print("model is pretrained , no need for train again")
		print("model before pruning test:")
		_,acc,_ = test(model,test_loader,criterion,args.device)
		logging.info("model_without_training acc : {}".format(acc))
		return
	print("---start training model---")
	args.milestones = eval(args.milestones)
	optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
	train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.2) #learning rate decay
	iter_per_epoch = len(train_loader)
	warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * (args.warm+1))
	print(iter_per_epoch)
	best = 50.0
	for epoch in range(0,args.milestones[-1]):
		if epoch > args.warm:
			train_scheduler.step(epoch)
		train_loss,train_acc,_ = train_slimming(model,train_loader,test_loader,criterion,optimizer,warmup_scheduler,args.warm,args.s,epoch,args.device)
		test_loss,test_acc,_ = test(model,test_loader,criterion,args.device)
		writer.add_scalars('Acc',{'Train_acc':train_acc,'Test_acc':test_acc},epoch)
		writer.add_scalars('Loss',{'Train_loss':train_loss,'Test_loss':test_loss},epoch)
		if epoch>120 and test_acc > best:
			best = test_acc
			torch.save(model,os.path.join(args.output_path,args.output_name+'_float32.pkl'))
	
	model = torch.load(os.path.join(args.output_path,args.output_name+'_float32.pkl'))
	print("---end trianing model---")
	print("model before pruning test:")
	_,acc1,_ = test(model,test_loader,criterion,args.device)
	logging.info("model_after_training acc : {}".format(acc1))
	return

def prune_model(model,test_loader,criterion,args):
	logging.info("pruning_mathod : {}".format(args.pruning_method))
	logging.info("pruning_rate : {}".format(args.pruning_rate))
	if args.pruning_method == 'global':
		return pruning_global_rate(model,test_loader,criterion,args.pruning_rate,device = args.device)
	if args.pruning_method == 'search':
		logging.info("pruning_alpha : {}".format(args.alpha))
		return pruning_search(model,test_loader,criterion,args.pruning_rate,alpha = args.alpha, \
											lbound=args.lbound,rbound=args.rbound,device=args.device)
	if args.pruning_method == 'DDPG':
		logging.info("hidden1 : {}  hidden2 ： {}".format(args.hidden1,args.hidden2))
		logging.info("lr_c : {}  lr_a : {}".format(args.lr_c,args.lr_a))
		logging.info("warmup : {}".format(args.warmup))
		logging.info("train_episode : {}".format(args.train_episode))
		return pruning_DDPG(model,test_loader,criterion,args.pruning_rate,args.train_episode,args.warmup, \
											lbound=args.lbound,rbound=args.rbound,output=args.ddpg_path,nb_states=args.nb_states, \
											nb_actions=args.nb_actions,hidden1=args.hidden1,hidden2=args.hidden2,lr_a=args.lr_a, \
											lr_c=args.lr_c,device=args.device)

def reload_model(args):
	if args.model_cfg == None:
		model = torch.load(args.model_path)
	else:
		model = torch.load(os.path.join(args.output_path,args.output_name+'_float32.pkl'))
	return model

def train_model_after_pruning(model_new,model,test_loader,train_loader,criterion,args):
	model_new = model_new.to(args.device)
	_,acc,_ = test(model_new,test_loader,criterion,args.device)
	logging.info("model_after_prune_before_retrain acc : {}".format(acc))
	print("---start training model after pruning---")
	optimizer = optim.SGD(model_new.parameters(), lr=args.lr*0.01)
	best = 60.0
	for epoch in range(0,25):
		train_loss,train_acc,_ = train_kd(model_new,train_loader,test_loader,criterion,optimizer,model,args.kd_T,args.kd_alpha,args.kd_beta,epoch,args.device)
		test_loss,test_acc,_ = test(model_new,test_loader,criterion,args.device)
		writer.add_scalars('Acc_AfterPruning',{'Train_acc':train_acc,'Test_acc':test_acc},epoch)
		writer.add_scalars('Loss_AfterPruning',{'Train_loss':train_loss,'Test_loss':test_loss},epoch)
	optimizer = optim.SGD(model_new.parameters(), lr=args.lr*0.001)
	for epoch in range(25,40):
		train_loss,train_acc,_ = train_kd(model_new,train_loader,test_loader,criterion,optimizer,model,args.kd_T,args.kd_alpha,args.kd_beta,epoch,args.device)
		test_loss,test_acc,_ = test(model_new,test_loader,criterion,args.device)
		writer.add_scalars('Acc_AfterPruning',{'Train_acc':train_acc,'Test_acc':test_acc},epoch)
		writer.add_scalars('Loss_AfterPruning',{'Train_loss':train_loss,'Test_loss':test_loss},epoch)
		if test_acc > best:
			best = test_acc
			torch.save(model_new,os.path.join(args.output_path,args.output_name+'_AfterPruning_float32.pkl'))
	model_new = torch.load(os.path.join(args.output_path,args.output_name+'_AfterPruning_float32.pkl'))
	print("model_new after pruning after retrain test:")
	_,acc1,_ = test(model_new,test_loader,criterion,args.device)
	logging.info("model_after_prune_after_retrain acc : {}".format(acc1))

def k_means_compression(model_new,args):
	model_int,map_int=apply_weight_kmeans(model_new,args.bits)
	torch.save(model_int.state_dict(),os.path.join(args.output_path,args.output_name+'_int8.pkl'))
	np.save(os.path.join(args.output_path,args.output_name+'_int8.npy'),map_int)

def compare(model,model_new,args):
	print("---before pruning---")
	flops,params = measure_for_model(model,device=args.device)
	print("flops: {}   params:{}".format(flops,params))
	print("---after pruning---")
	flops1,params1 = measure_for_model(model_new,device=args.device)
	print("flops: {}   params:{}".format(flops1,params1))
	time.sleep(5)
	logging.info("model_before_prune flops : {} params : {}".format(flops,params))
	logging.info("model_after_prune flops : {} params : {}".format(flops1,params1))

if __name__ == "__main__":
	criterion = nn.CrossEntropyLoss()
	args = parse_args()
	if not os.path.exists(args.output_path):
		os.mkdir(args.output_path)
	f = open(os.path.join(args.output_path,args.output_name+'.log'),'w')
	f.close()
	logging.basicConfig(filename=os.path.join(args.output_path,args.output_name+'.log'),format='%(asctime)s-%(levelname)s %(message)s', level=logging.DEBUG)
	print("---prepare for model---")
	model = prepare_model(args)
	print(model)
	writer = SummaryWriter(os.path.join(args.output_path,'result'))
	print("---prepare for datasets---")
	test_loader,train_loader = prepare_datasets(args)
	print("---train for model---")
	train_model(model,test_loader,train_loader,criterion,writer,args)
	print("---pruning for model---")
	model_for_pruning = model
	model_new = model
	for i in range(args.pruning_time):
		model_new = prune_model(model_for_pruning,test_loader,criterion,args)
		print("model_new after the {} pruning befor retrain test:".format(i))
		_,_,_ = test(model_new,test_loader,criterion,args.device)
		print("---compare for model and model_new---")
		compare(model,model_new,args)
		model = reload_model(args)
		print("---train for model_new after pruning---")
		train_model_after_pruning(model_new,model,test_loader,train_loader,criterion,args)
		model_for_pruning = model_new
	print("---k-means---")
	k_means_compression(model_new,args)
	model_reassignment = model_reassignment_kmeans(
		os.path.join(args.output_path,args.output_name+'_int8.pkl'),
			os.path.join(args.output_path,args.output_name+'_int8.npy'))
	print("---after k-means---")
	_,acc,_ = test(model_reassignment.to(args.device),test_loader,criterion,args.device)
	logging.info("---after k-means---")
	logging.info("acc : {}".format(acc))


	 
