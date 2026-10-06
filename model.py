"""
Q-network: a small MLP mapping one agent's input vector to one Q-value per action.

Input:  the agent's observation (positions normalized to [0, 1]), plus a one-hot
        agent ID when both agents share the network.
Output: Q(s, a) for the 5 actions (LEFT, DOWN, RIGHT, UP, STAND).

The state is just a handful of coordinates on a small grid, so two hidden layers of
64 units are plenty; the difficulty of the task lies in coordination, not in
representing the state.
"""

import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, in_dim, n_actions=5, hidden=(64, 64)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_actions))  # linear output: Q-values are unbounded
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
