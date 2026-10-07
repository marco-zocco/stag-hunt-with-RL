"""
Q-network: a small MLP mapping one agent's input vector to one Q-value per action.

Input:  agent's observation normalized, plus a one-hot
        agent ID when both agents share the network.
Output: Q(s, a) for the 5 actions (LEFT, DOWN, RIGHT, UP, STAND).

With noisy=True every linear layer is a NoisyLinear (Fortunato et al., 2018,
"Noisy Networks for Exploration"): exploration comes from learned noise on the
weights instead of epsilon-greedy.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    Linear layer with factorised Gaussian noise (Fortunato et al., 2018, Sec. 3.2):

        w = mu_w + sigma_w * eps_w,   eps_w = f(eps_out) f(eps_in)^T,   f(x) = sign(x) sqrt(|x|)
        b = mu_b + sigma_b * eps_b,   eps_b = f(eps_out)

    mu and sigma are learned by gradient descent like ordinary weights, so the
    network itself learns how much to explore (sigma can shrink where it has
    learned enough). Factorised noise needs only in + out random numbers per
    sample instead of in * out.

    In train mode the layer uses the current noise sample (reset_noise() draws a
    new one); in eval mode it uses the mean weights only, which is the greedy
    policy we evaluate.
    """

    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.sigma_b = nn.Parameter(torch.empty(out_features))
        # noise samples are buffers: saved with the model, never trained
        self.register_buffer("eps_in", torch.zeros(in_features))
        self.register_buffer("eps_out", torch.zeros(out_features))

        # initialisation from the paper (factorised case)
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.mu_w, -bound, bound)
        nn.init.uniform_(self.mu_b, -bound, bound)
        nn.init.constant_(self.sigma_w, sigma0 / math.sqrt(in_features))
        nn.init.constant_(self.sigma_b, sigma0 / math.sqrt(in_features))
        self.reset_noise()

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()

    @torch.no_grad()
    def reset_noise(self):
        self.eps_in.copy_(self._f(torch.randn(self.in_features)))
        self.eps_out.copy_(self._f(torch.randn(self.out_features)))

    def forward(self, x):
        if not self.training:
            return F.linear(x, self.mu_w, self.mu_b)
        w = self.mu_w + self.sigma_w * torch.outer(self.eps_out, self.eps_in)
        b = self.mu_b + self.sigma_b * self.eps_out
        return F.linear(x, w, b)


class QNetwork(nn.Module):
    def __init__(self, in_dim, n_actions=5, hidden=(64, 64), noisy=False, sigma0=0.5):
        super().__init__()

        def linear(i, o):
            return NoisyLinear(i, o, sigma0) if noisy else nn.Linear(i, o)

        layers, d = [], in_dim
        for h in hidden:
            layers += [linear(d, h), nn.ReLU()]
            d = h
        layers.append(linear(d, n_actions))  # linear output: Q-values are unbounded
        self.net = nn.Sequential(*layers)
        self.noisy_layers = [m for m in self.net if isinstance(m, NoisyLinear)]

    def forward(self, x):
        return self.net(x)

    # --- noise handling (no-ops for an ordinary network)

    def reset_noise(self):
        for m in self.noisy_layers:
            m.reset_noise()

    def get_noise(self):
        return [(m.eps_in.clone(), m.eps_out.clone()) for m in self.noisy_layers]

    @torch.no_grad()
    def set_noise(self, noise):
        for m, (e_in, e_out) in zip(self.noisy_layers, noise):
            m.eps_in.copy_(e_in)
            m.eps_out.copy_(e_out)
