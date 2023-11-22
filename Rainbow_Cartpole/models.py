import torch
import torch.nn as nn
import math
import torch.functional as F


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features,
                 sigma_init=0.005, bias=True):
        super(NoisyLinear, self).__init__(
            in_features, out_features, bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", z)

        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)
            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", z)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias

        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * \
                   self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data + \
            self.weight
        
        return F.linear(input, v, bias)


class Noisy_DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = NoisyLinear(n_observations, 128)
        self.layer2 = NoisyLinear(128, 128)
        self.layer3 = NoisyLinear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class Dueling_DQN(nn.Module):
  def __init__(self, n_observations, n_actions):
    super().__init__()

    self.layer1 = NoisyLinear(n_observations, 128)
    self.layer2 = NoisyLinear(128, 128)

    self.advantage = nn.Sequential(
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64, n_actions)
    )

    self.value = nn.Sequential(
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

  def adv_val(self, x):
    return self.advantage(x), self.value(x)

  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    advantage, value = self.adv_val(x)
      
    return value + (advantage - advantage.mean(dim=1, keepdim=True))