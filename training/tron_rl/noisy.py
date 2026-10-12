"""Factorized Gaussian NoisyNets linear layer (Fortunato et al., 2017),
copied near-verbatim from summer_challenge_2026/training/rlx/noisy.py -- this
is fully game-agnostic machinery, used here as the exploration mechanism
instead of epsilon-greedy. Lives only in head linears, never in trunk convs.

At eval() this collapses to a plain `W_mu @ x + b_mu`: the numpy submission
exports only weight_mu/bias_mu and ships no noise machinery.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma0 = sigma0

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma0 / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma0 / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size, device, dtype):
        x = torch.randn(size, device=device, dtype=dtype)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        device, dtype = self.weight_mu.device, self.weight_mu.dtype
        eps_in = self._scale_noise(self.in_features, device, dtype)
        eps_out = self._scale_noise(self.out_features, device, dtype)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)
