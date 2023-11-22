import gymnasium as gym
from matplotlib import pyplot as plt
from itertools import count
import copy
import agent
import models
import torch
import time

env = gym.make("CartPole-v1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
memory_buffer_size = 10000
num_episodes  = 600

episode_durations = []

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

torch.manual_seed(seed)
policy_net = models.DQN(n_observations, n_actions).to(device)
target_net = copy.deepcopy(policy_net)
target_net.to(device)

exp_buffer = agent.ReplayMemory(memory_buffer_size)

agent = agent.Agent(exp_buffer, policy_net, target_net, device)

for i_episode in range(num_episodes):
  state, info = env.reset()
  state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
  t1 = time.time()
  for t in count():

    state, done = agent.explore(env, state)
    agent.optimize_model(batch_size=128)
    agent.sync_nets()

    if done:
      episode_durations.append(t + 1)
      t2 = time.time()
      print("Episode: ", i_episode+1, "duration: ", t2-t1)
      break

print('Complete')
plt.plot(episode_durations)
plt.show()

