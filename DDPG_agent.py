import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from DDPG_memory import *
from utils import *
from tensorboardX import SummaryWriter

criterion = nn.MSELoss()
USE_CUDA = torch.cuda.is_available()

#超参数存放点

rmsize = 100*12
window_length = 1
bsize = 64
tau = 0.01
discount = 1
epsilon = 50000
init_delta = 0.1
delta_decay = 0.95


class Actor(nn.Module):
	def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300):
		super(Actor, self).__init__()
		self.fc1 = nn.Linear(nb_states, hidden1)
		self.fc2 = nn.Linear(hidden1, hidden2)
		self.fc3 = nn.Linear(hidden2, nb_actions)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.relu(out)
		out = self.fc3(out)
		out = self.sigmoid(out)
		return out


class Critic(nn.Module):
	def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300):
		super(Critic, self).__init__()
		self.fc11 = nn.Linear(nb_states, hidden1)
		self.fc12 = nn.Linear(nb_actions, hidden1)
		self.fc2 = nn.Linear(hidden1, hidden2)
		self.fc3 = nn.Linear(hidden2, 1)
		self.relu = nn.ReLU()

	def forward(self, xs):
		x, a = xs
		out = self.fc11(x) + self.fc12(a)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.relu(out)
		out = self.fc3(out)
		return out

class DDPG(object):
	def __init__(self,warmup,nb_states,nb_actions,left_bound,right_bound,hidden1 = 300 ,hidden2 = 300,lr_a=1e-4,lr_c=1e-4):
		self.nb_states = int(nb_states)
		self.nb_actions = int(nb_actions)
		# Create Actor and Critic Network
		self.lr_a = lr_a
		self.lr_c = lr_c
		self.actor = Actor(self.nb_states, self.nb_actions, hidden1,hidden2)
		self.actor_target = Actor(self.nb_states, self.nb_actions,  hidden1,hidden2)
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr_a)
		self.critic = Critic(self.nb_states, self.nb_actions,  hidden1,hidden2)
		self.critic_target = Critic(self.nb_states, self.nb_actions,  hidden1,hidden2)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr_c)
		self.hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
		self.hard_update(self.critic_target, self.critic)
		# Create replay buffer
		self.memory = SequentialMemory(limit=rmsize, window_length=window_length)
		self.batch_size = int(bsize)
		self.tau = tau
		self.discount = discount
		self.depsilon = 1.0 / epsilon
		self.left_bound = left_bound
		self.right_bound = right_bound
		# noise
		self.init_delta = init_delta
		self.delta_decay = delta_decay
		self.warmup = int(warmup)
		#
		self.epsilon = 1.0
		# self.s_t = None  # Most recent state
		# self.a_t = None  # Most recent action
		self.is_training = True
		#
		if USE_CUDA: self.cuda()
		# moving average baseline
		self.moving_average = None
		self.moving_alpha = 0.5  # based on batch, so small
		self.train_epoch = 0
		self.cur_id = 0
		self.update_epoch = 0
		self.writer = SummaryWriter("./DDPG")
	def update_policy(self):
		# Sample batch
		state_batch, action_batch, reward_batch, \
		next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
		# normalize the reward
		
		batch_mean_reward = np.mean(reward_batch)
		if self.moving_average is None:
			self.moving_average = batch_mean_reward
		else:
			self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
		reward_batch -= self.moving_average
		with torch.no_grad():
			next_q_values = self.critic_target([
				to_tensor(next_state_batch),
				self.actor_target(to_tensor(next_state_batch)),
			])

		target_q_batch = to_tensor(reward_batch) + \
						 self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values
		# Critic update
		self.critic.zero_grad()

		q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])
		value_loss = criterion(q_batch, target_q_batch)
		#print("q_batch:")
		#print(q_batch)
		#print("target_q_batch:")
		#print(target_q_batch)
		
		value_loss.backward()
		self.critic_optim.step()
		# Actor update
		self.actor.zero_grad()
		policy_loss = -self.critic([
			to_tensor(state_batch),
			self.actor(to_tensor(state_batch))
		])

		policy_loss = policy_loss.mean()
		policy_loss.backward()
		self.writer.add_scalars('Loss_DDPG',{'Critric_loss':value_loss.mean(),'Actor_loss':policy_loss},self.train_epoch)
		self.actor_optim.step()
		# Target update
		self.soft_update(self.actor_target, self.actor)
		self.soft_update(self.critic_target, self.critic)
		self.train_epoch = self.train_epoch + 1
	
	def eval(self):
		self.actor.eval()
		self.actor_target.eval()
		self.critic.eval()
		self.critic_target.eval()

	def cuda(self):
		self.actor.cuda()
		self.actor_target.cuda()
		self.critic.cuda()
		self.critic_target.cuda()

	def observe(self, r_t, s_t, s_t1, a_t, done):
		if self.is_training:
			self.memory.append(s_t, a_t, r_t, done)  # save to memory
			# self.s_t = s_t1

	def random_action(self):
		action = np.random.uniform(self.left_bound[self.cur_id],self.right_bound[self.cur_id], self.nb_actions)
		self.cur_id = (self.cur_id + 1)%len(self.left_bound)
		return action

	def select_action(self, s_t, episode):
		# assert episode >= self.warmup, 'Episode: {} warmup: {}'.format(episode, self.warmup)
		action = to_numpy(self.actor(to_tensor(np.array(s_t).reshape(1, -1)))).squeeze(0)
		#print(action)
		delta = self.init_delta * (self.delta_decay ** (episode - self.warmup))
		# action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
		#action = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=action, sigma=delta)
		action = action + np.random.uniform(-delta,delta,1)
		action = np.clip(action, self.left_bound[self.cur_id], self.right_bound[self.cur_id])
		self.cur_id = (self.cur_id + 1)%len(self.left_bound)

		# self.a_t = action
		return action

	def reset(self, obs):
		pass
		# self.s_t = obs
		# self.random_process.reset_states()

	def load_weights(self, output):
		if output is None: return

		self.actor.load_state_dict(
			torch.load('{}/actor.pkl'.format(output))
		)

		self.critic.load_state_dict(
			torch.load('{}/critic.pkl'.format(output))
		)

	def save_model(self, output):
		torch.save(
			self.actor.state_dict(),
			'{}/actor.pkl'.format(output)
		)
		torch.save(
			self.critic.state_dict(),
			'{}/critic.pkl'.format(output)
		)

	def soft_update(self, target, source):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(
				target_param.data * (1.0 - self.tau) + param.data * self.tau
			)

	def hard_update(self, target, source):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)

	def sample_from_truncated_normal_distribution(self, lower, upper, mu, sigma, size=1):
		from scipy import stats
		return stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=size)


