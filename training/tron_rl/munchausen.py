"""Munchausen RL's log-policy term over the plain {UP, DOWN, LEFT, RIGHT}
action space -- the textbook Munchausen-DQN formula, collapsed from
summer_challenge_2026's combinatorial elementary-symmetric-polynomial
machinery (needed there for the knapsack Build action space). Here there are
at most 4 legal actions, so a plain masked logsumexp is exact -- no
truncation or numerical tricks needed at all.

tau_m * ln pi(a|s) with pi = softmax(Q_bar(s,.) / tau_m) over the legal
actions only (V cancels in the softmax, so this only needs the advantages).
"""
import torch

NEG_INF = float("-inf")


def log_partition(advantage, legal, tau_m):
    """advantage, legal: (B, NUM_ACTIONS). Returns log Z (B,) = logsumexp of
    advantage/tau_m over the legal actions only."""
    x = torch.where(legal, advantage / tau_m, torch.full_like(advantage, NEG_INF))
    return torch.logsumexp(x, dim=-1)


def munchausen_log_policy(chosen_advantage, log_z, tau_m):
    """tau_m * ln pi(a_t|s_t). chosen_advantage, log_z: (B,)."""
    return tau_m * (chosen_advantage / tau_m - log_z)


def munchausen_bonus(tau_m_log_pi, alpha=0.9, clip_lo=-1.0, clip_hi=0.0):
    """alpha * clip(tau_m * ln pi(a_t|s_t), clip_lo, clip_hi), added to the
    reward per BTR's own recipe."""
    return alpha * torch.clamp(tau_m_log_pi, min=clip_lo, max=clip_hi)
