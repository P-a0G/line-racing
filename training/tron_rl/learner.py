"""The BTR learner: one gradient step given a replay batch. Ties together
net.py (online + target DuelingIQNNet) and munchausen.py (the log-policy
reward bonus, computed from the TARGET net's own mean-Q per BTR's own
recipe) into the IQN quantile-Huber loss.

Unlike summer_challenge_2026/training/rlx/learner.py, no decode module is
needed here: with a plain 4-way action space the "greedy action" is just a
masked argmax over the target net's own mean-Q, inlined directly rather than
needing a combinatorial knapsack/region decode.

Matches the BTR component list: IQN (no C51), Munchausen (no separate
Double-DQN -- the greedy quantile bootstrap uses the target net's own mean-Q
for both action selection and the per-quantile target), n-step returns, hard
target replace every `target_update_interval` gradient steps, Adam(lr=1e-4,
eps=0.005/batch_size), grad-norm clip 10.
"""
import torch

from tron_rl.net import DuelingIQNNet
from tron_rl.munchausen import log_partition, munchausen_log_policy, munchausen_bonus


def quantile_huber_loss(pred, pred_taus, target, kappa=1.0):
    """pred, pred_taus: (B,K); target: (B,K'). Standard IQN pairwise
    quantile-Huber loss (Dabney et al. 2018), copied verbatim from
    summer_challenge_2026/training/rlx/learner.py -- fully generic."""
    delta = target[:, None, :] - pred[:, :, None]  # (B,K,K')
    abs_delta = delta.abs()
    huber = torch.where(abs_delta <= kappa, 0.5 * delta ** 2, kappa * (abs_delta - 0.5 * kappa))
    indicator = (delta.detach() < 0).to(pred.dtype)
    weight = (pred_taus[:, :, None] - indicator).abs()
    loss_per_pair = weight * huber / kappa
    return loss_per_pair.sum(dim=1).mean(dim=1)


class Learner:
    def __init__(self, net_kwargs=None, lr=1e-4, batch_size=64, grad_clip=10.0,
                 target_update_interval=500, num_taus=8, kappa=1.0,
                 tau_m=0.03, munch_alpha=0.9, munch_clip=(-1.0, 0.0),
                 device="cpu"):
        net_kwargs = net_kwargs or {}
        self.net_kwargs = net_kwargs  # kept so collect.py can build an architecturally-identical scratch net
        self.device = torch.device(device)
        self.online = DuelingIQNNet(**net_kwargs).to(self.device)
        self.target = DuelingIQNNet(**net_kwargs).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr, eps=0.005 / batch_size)
        self.grad_clip = grad_clip
        self.target_update_interval = target_update_interval
        self.num_taus = num_taus
        self.kappa = kappa
        self.tau_m = tau_m
        self.munch_alpha = munch_alpha
        self.munch_clip = munch_clip
        self.grad_steps = 0

    def _to_tensors(self, batch):
        t = {
            "grid_t": torch.from_numpy(batch["grid_t"]).float().to(self.device),
            "grid_tn": torch.from_numpy(batch["grid_tn"]).float().to(self.device),
            "legal_t": torch.from_numpy(batch["legal_t"]).bool().to(self.device),
            "legal_tn": torch.from_numpy(batch["legal_tn"]).bool().to(self.device),
            "action": torch.from_numpy(batch["action"]).long().to(self.device),
            "terminal": torch.from_numpy(batch["terminal"]).bool().to(self.device),
            "n_step_return": torch.from_numpy(batch["n_step_return"]).float().to(self.device),
            "bootstrap_discount": torch.from_numpy(batch["bootstrap_discount"]).float().to(self.device),
            "weights": torch.from_numpy(batch["weights"]).float().to(self.device),
        }
        return t

    def train_step(self, batch):
        b = self._to_tensors(batch)
        B = b["grid_t"].shape[0]

        online_taus = torch.rand(B, self.num_taus, device=self.device)
        out_t = self.online(b["grid_t"], b["legal_t"], online_taus)
        z_pred = torch.gather(out_t["q"], 2, b["action"][:, None, None].expand(B, self.num_taus, 1)).squeeze(-1)

        with torch.no_grad():
            munch_taus = torch.rand(B, self.num_taus, device=self.device)
            out_t_target = self.target(b["grid_t"], b["legal_t"], munch_taus)
            # Using Q (= V + A) instead of extracting the pure advantage (as
            # the rail game's munchausen.py does) is equivalent here: V is
            # the same additive constant across actions for a given sample,
            # so it cancels inside any softmax over actions, and the
            # per-sample constant that adds to log_z cancels right back out
            # in munchausen_log_policy's own subtraction.
            mean_q_t = out_t_target["q"].mean(dim=1)  # (B, NUM_ACTIONS)
            log_z = log_partition(mean_q_t, b["legal_t"], self.tau_m)
            chosen_q_t = torch.gather(mean_q_t, 1, b["action"][:, None]).squeeze(1)
            tau_m_log_pi = munchausen_log_policy(chosen_q_t, log_z, self.tau_m)
            m_bonus = munchausen_bonus(tau_m_log_pi, self.munch_alpha, *self.munch_clip)
            shaped_return = b["n_step_return"] + m_bonus

            target_taus = torch.rand(B, self.num_taus, device=self.device)
            out_tn = self.target(b["grid_tn"], b["legal_tn"], target_taus)
            mean_q_tn = out_tn["q"].mean(dim=1)
            greedy_action = mean_q_tn.argmax(dim=-1)  # target net's own greedy action -- BTR drops Double-DQN
            bootstrap_z = torch.gather(
                out_tn["q"], 2, greedy_action[:, None, None].expand(B, self.num_taus, 1)).squeeze(-1)
            bootstrap_z = bootstrap_z * (~b["terminal"])[:, None]

            z_target = shaped_return[:, None] + b["bootstrap_discount"][:, None] * bootstrap_z

        per_sample_loss = quantile_huber_loss(z_pred, online_taus, z_target, self.kappa)
        loss = (per_sample_loss * b["weights"]).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()
        self.online.reset_noise()
        self.target.reset_noise()

        self.grad_steps += 1
        if self.grad_steps % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        td_error = (z_target.mean(dim=1) - z_pred.mean(dim=1)).detach().abs()
        return {
            "loss": float(loss.item()),
            "td_error": td_error.cpu().numpy(),
            "mean_munchausen_bonus": float(m_bonus.mean().item()),
        }
