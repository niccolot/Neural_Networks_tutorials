import torch
import torch.nn as nn
import math
import random
from collections import namedtuple
from collections import deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
  def __init__(self, exp_buffer, policy_net, target_net, device, n_steps=1, noisy_net=False):
    self.exp_buffer = exp_buffer
    self.policy_net = policy_net
    self.target_net = target_net
    self.n_steps = n_steps
    self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=1e-4, amsgrad=True)
    self.device = device
    self.noisy_net = noisy_net
    self.steps_done = 0
    self.eps_start = 0.9
    self.eps_end = 0.05
    self.eps_decay = 1000
    self.gamma = 0.99
    self.tau = 0.005


  def _eps_decay(self):
    return self.eps_end + (self.eps_start - self.eps_end) * \
        math.exp(-1. * self.steps_done / self.eps_decay)


  def _select_action(self, state, env):

    if self.noisy_net:
      # if noisy_net method is used, the networks loaded in the agent must be
      # with noisy layers
      with torch.no_grad():
        # .view(1,1) in order for it to work with the .gather() method
        # used to extract state action values
        return self.policy_net(state).max(1)[1].view(1,1)

    else:
      eps_threshold = self._eps_decay()
    
      if random.random() > eps_threshold:
        with torch.no_grad():
          # .view(1,1) in order for it to work with the .gather() method
          # used to extract state action values
          return self.policy_net(state).max(1)[1].view(1,1)
      else:
        return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)


  def sync_nets(self):
    target_net_state_dict = self.target_net.state_dict()
    policy_net_state_dict = self.policy_net.state_dict()

    for key in policy_net_state_dict:
      target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)

    self.target_net.load_state_dict(target_net_state_dict)


  def explore(self, env, state):
   
    total_reward = 0.0
    state_init = state

    for step in range(self.n_steps):

      action = self._select_action(state, env)
      observation, reward, terminated, truncated, _ = env.step(action.item())
      done = terminated or truncated

      reward *= self.gamma**step
      total_reward += reward

      if terminated:
        next_state = None
        state = next_state
        break

      else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)


      state = next_state


    total_reward = torch.tensor([total_reward], device=self.device)
    self.exp_buffer.push(state_init, action, next_state, total_reward)
    state = next_state
    self.steps_done += 1

    return state, done


  def optimize_model(self, batch_size, double_dqn=False):

    if len(self.exp_buffer) < batch_size:
      return

    transitions = self.exp_buffer.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
    
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])


    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=self.device)
    next_state_acts = torch.zeros(batch_size, device=self.device)
    with torch.no_grad():

      if double_dqn:
          next_state_acts = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(-1)
          next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_state_acts).squeeze(-1)

      else:
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values * (self.gamma**self.n_steps)) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()