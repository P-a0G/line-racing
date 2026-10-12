"""Simplified IQN-dueling network for line-racing, adapted from
summer_challenge_2026/training/rlx/net.py.

Kept as-is (generic, game-agnostic BTR machinery): the cylindric (uniform
channels/resolution) CNN trunk with spectral-norm convs + ReZero residual
connections, KataGo-style GlobalPoolingBias context injection, and the IQN
distributional value head (8 training taus, 64 cosine embeddings).

Dropped relative to the rail game's net.py: all size-agnosticism plumbing
(the `valid`/`n_valid` masking, the global-pool's size-scaled-mean term) --
line-racing's board is always exactly WIDTH x HEIGHT, so every cell is
always real and there is nothing to mask. Also dropped: the per-cell dense
Build head and per-region Ink head (that combinatorial action space doesn't
exist here), replaced by a single dense advantage head over the plain
4-way {UP, DOWN, LEFT, RIGHT} action space, dueling-combined and
mean-normalized over the legal moves only -- the real advantage property.

NoisyNets (noisy.py) live only in the head linears, same as the rail game.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from tron_rl.obs import NUM_CHANNELS
from tron_rl.noisy import NoisyLinear

NUM_ACTIONS = 4  # UP, DOWN, LEFT, RIGHT -- see simulator/Engine.hpp's Direction enum
NUM_TAUS_TRAIN = 8
NUM_COSINES = 64
DEFAULT_CHANNELS = 24
DEFAULT_BLOCKS = 2
DEFAULT_GPB_BLOCKS = (0,)  # one GlobalPoolingBias block is plenty at this depth


def _global_pool2(x):
    """x: (N,C,H,W). Returns (N,2C): [mean, max] over all cells -- every
    cell is always real on this fixed-size board, so no masking is needed
    (contrast summer_challenge_2026's 3-term, size-scaled, masked pool)."""
    mean = x.mean(dim=(2, 3))
    mx = x.amax(dim=(2, 3))
    return torch.cat([mean, mx], dim=1)


class GlobalPoolingBias(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Linear(channels * 2, channels)

    def forward(self, x):
        bias = self.fc(_global_pool2(x))
        return x + bias[:, :, None, None]


class ReZeroBlock(nn.Module):
    """Conv3x3 -> ReLU -> [GPB] -> Conv3x3 -> x + alpha*y, alpha learnable
    and zero-initialized (ReZero) -- used instead of BatchNorm for the same
    training-stability-under-off-policy-replay reason as the rail game."""

    def __init__(self, channels, use_gpb):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(channels, channels, 3, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(channels, channels, 3, padding=1, bias=False))
        self.gpb = GlobalPoolingBias(channels) if use_gpb else None
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        if self.gpb is not None:
            y = self.gpb(y)
        y = self.conv2(y)
        return x + self.alpha * y


class Trunk(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, channels=DEFAULT_CHANNELS,
                 blocks=DEFAULT_BLOCKS, gpb_blocks=DEFAULT_GPB_BLOCKS):
        super().__init__()
        self.channels = channels
        self.stem = spectral_norm(nn.Conv2d(in_channels, channels, 1, bias=False))
        self.blocks = nn.ModuleList([
            ReZeroBlock(channels, use_gpb=(i in gpb_blocks)) for i in range(blocks)
        ])

    def forward(self, grid):
        """grid: (B,NUM_CHANNELS,H,W). Returns (B,channels,H,W)."""
        x = self.stem(grid)
        for block in self.blocks:
            x = block(x)
        return x


def cosine_embedding(taus, num_cosines=NUM_COSINES):
    """taus: (B,K) in [0,1]. Returns (B,K,num_cosines)."""
    i = torch.arange(1, num_cosines + 1, device=taus.device, dtype=taus.dtype)
    angles = math.pi * i[None, None, :] * taus[:, :, None]
    return torch.cos(angles)


class IQNEmbedding(nn.Module):
    def __init__(self, channels, num_cosines=NUM_COSINES):
        super().__init__()
        self.fc = nn.Linear(num_cosines, channels)
        self.num_cosines = num_cosines

    def forward(self, taus):
        return F.relu(self.fc(cosine_embedding(taus, self.num_cosines)))  # (B,K,channels)


class DuelingIQNNet(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, channels=DEFAULT_CHANNELS,
                 blocks=DEFAULT_BLOCKS, gpb_blocks=DEFAULT_GPB_BLOCKS,
                 num_cosines=NUM_COSINES, noisy_sigma0=0.5):
        super().__init__()
        self.channels = channels
        self.trunk = Trunk(in_channels, channels, blocks, gpb_blocks)
        self.iqn = IQNEmbedding(channels, num_cosines)

        self.value_fc1 = NoisyLinear(channels * 2, channels, noisy_sigma0)
        self.value_fc2 = NoisyLinear(channels, 1, noisy_sigma0)
        self.adv_fc1 = NoisyLinear(channels * 2, channels, noisy_sigma0)
        self.adv_fc2 = NoisyLinear(channels, NUM_ACTIONS, noisy_sigma0)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, grid, legal_mask, taus):
        """
        grid        (B, NUM_CHANNELS, H, W) float
        legal_mask  (B, NUM_ACTIONS) bool -- this turn's legal moves
        taus        (B, K) float in [0,1] -- IQN quantile fractions

        Returns dict with 'v' (B,K) and 'q' (B,K,NUM_ACTIONS), the latter
        already dueling-combined and mean-normalized over the legal action
        set (a true advantage, not an arbitrary per-action offset), with
        every illegal action driven to -1e9 -- EXCEPT when every action is
        illegal (fully boxed in), where all four instead get a plain
        mean-centered advantage over the full 4 (see the comment at that
        branch for why: this transition still gets trained on, and -1e9
        would poison its TD target/prediction, not just a decision-time
        argmax). Which direction wins that argmax is inconsequential either
        way -- every option is equally fatal.
        """
        B, K = taus.shape
        trunk_out = self.trunk(grid)                                  # (B,C,H,W)
        phi = self.iqn(taus)                                           # (B,K,C)
        psi = trunk_out[:, None] * phi[:, :, :, None, None]             # (B,K,C,H,W)
        psi_flat = psi.reshape(B * K, self.channels, *trunk_out.shape[-2:])

        pool = _global_pool2(psi_flat)                                  # (B*K, 2C)
        v = self.value_fc2(F.relu(self.value_fc1(pool))).reshape(B, K)
        a_raw = self.adv_fc2(F.relu(self.adv_fc1(pool))).reshape(B, K, NUM_ACTIONS)

        legal = legal_mask[:, None, :].expand(B, K, NUM_ACTIONS)
        legal_count = legal_mask.sum(dim=-1).clamp(min=1).to(grid.dtype)  # (B,)
        sum_a = torch.where(legal, a_raw, torch.zeros_like(a_raw)).sum(dim=-1)  # (B,K)
        mean_a = sum_a / legal_count[:, None]
        a_masked = torch.where(legal, a_raw - mean_a[:, :, None], torch.full_like(a_raw, -1e9))

        # When EVERY action is illegal (fully boxed in -- this does happen in
        # real self-play, not just a theoretical edge case), fall back to a
        # plain mean-centered advantage over all 4 instead of -1e9: this
        # transition still gets replayed and its taken action still gets
        # gathered for the TD target (learner.py), so forcing every entry to
        # -1e9 would poison that target/prediction with a sentinel 9 orders
        # of magnitude past any real reward -- which is exactly what caused
        # a catastrophic (10^8) loss blowup in early real-data testing,
        # traced to this one gather. Which direction gets argmax'd here is
        # otherwise inconsequential: every option is equally fatal.
        any_legal = legal_mask.any(dim=-1)[:, None, None]  # (B,1,1)
        a = torch.where(any_legal, a_masked, a_raw - a_raw.mean(dim=-1, keepdim=True))

        q = v[:, :, None] + a
        return {"v": v, "q": q}
