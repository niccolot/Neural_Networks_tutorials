# DQN Rainbow extensions for DQN on Cartpole problem

This repo contains the implementations of some of the DQN extensions presented in [this article](https://arxiv.org/abs/1710.02298), in particular:


## Double DQN

In basic DQN the update rule reads as 

$$
Q(s_t, a_t) = r_t + \gamma \max_aQ'(s_{t+1}, a)
$$

with $Q'$ the target network. Double DQN consists in choosing actions
for the next state using the trained network, but taking values of $Q$ from the target network, resulting in the update rule

$$
Q(s_t, a_t) = r_t + \gamma \max_aQ'\Big(s_{t+1}, \textrm{argmax} \\{Q(s_{t+1}, a)\\}_a\Big)
$$

This is implemented as a boolean option in the `optimize_model` method of the agent in `agent.py`

```python
def optimize_model(self, batch_size, double_dqn=False)
```

## Noisy networks

It adds to the networks some gaussian noise in order to be used as an exporation mechanism instead of the $\varepsilon$-greedy approach.

Independent Gaussian noise: for every weight in a fully connected layer, we
have a random value that we draw from the normal distribution. Parameters
of the noise, $\mu$ and $\sigma$ , are stored inside the layer and get trained using
backpropagation in the same way that we train weights of the standard
linear layer. The output of such a "noisy layer" is calculated in the same way as in a linear layer.

This is implemented as a neural network model in the `models.py` file 

```python
class NoisyLinear(nn.Linear)
```

The noisy networks must be loaded as `policy_net` and `target_net` in the `main.py` file and the agent must has the boolean option `noisy_net=True` since it disables the $\varepsilon$-greedy exploration.

## N-steps DQN

It substitutes the usual update rule

$$
Q(s_t, a_t) = r_t + \gamma \max_aQ'(s_{t+1}, a)
$$

by taking more steps before choosing the action, resulting in an update rule (e.g. for n=2 steps)

$$
Q(s_t, a_t) = r_t + \gamma r_{t+1} + \gamma^2 \max_aQ'(s_{t+2}, a)
$$

This is implemented as a boolean option for the agent in `agent.py`

```python
def __init__(self, exp_buffer, policy_net, target_net, device, n_steps=1, noisy_net=False)
```

## Dueling DQN

Dueling DQN consists in defining the state-action value function as

$$
Q(s,a) = V(s) + A(s,a) - \frac{1}{N}\sum_k A(s,k)
$$

for an agent with N actions.

This is implemented as another neural network model in the `models.py` file that has to be selected as `policy_net` and `target_net` in the `main.py` file.

# Contents and details

* **agent.py**: the implementation of the agent and the replay buffer
* **models.py**: the NN models for the state-action functions
* **main.py**: a script for the training itself

The synchronization between the policy and target networks is done with soft update rule presented in [this article](https://arxiv.org/abs/1509.02971) in which the weights of the target net are updated each step with the rule

$$
\theta_{target}' = \tau\theta_{policy} + (1-\tau)\theta_{target}
$$

with $\tau \ll1$.

The methods are implemented with the cartpole problems in order to have a fast and light training, altough it could hide the true effectiveness of the methods due to the simplicity of the environment, and every hyperparameters has not been optimized due to the large amount of time it would take even for an environment as simple as cartpole.
