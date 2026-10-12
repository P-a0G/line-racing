import torch

from tron_rl.munchausen import log_partition, munchausen_log_policy, munchausen_bonus


def _reference_log_pi(advantage, legal, chosen, tau_m):
    """Direct masked-softmax reference: exact by construction, since this is
    exactly what pi(a|s) = softmax(Q_bar/tau_m) means. Used to check our
    logsumexp-based formula agrees exactly (not approximately -- there's no
    truncation here, unlike the rail game's combinatorial version)."""
    masked = torch.where(legal, advantage / tau_m, torch.full_like(advantage, float("-inf")))
    log_pi = masked - torch.logsumexp(masked, dim=-1, keepdim=True)
    chosen_log_pi = torch.gather(log_pi, 1, chosen[:, None]).squeeze(1)
    return tau_m * chosen_log_pi


def test_matches_direct_softmax_exactly():
    torch.manual_seed(0)
    B, tau_m = 100, 0.03
    advantage = torch.randn(B, 4) * 5
    legal = torch.rand(B, 4) > 0.3
    legal[:, 0] = True  # guarantee at least one legal action per row
    chosen = torch.stack([torch.where(legal[b])[0][0] for b in range(B)])

    log_z = log_partition(advantage, legal, tau_m)
    chosen_advantage = torch.gather(advantage, 1, chosen[:, None]).squeeze(1)
    got = munchausen_log_policy(chosen_advantage, log_z, tau_m)
    expected = _reference_log_pi(advantage, legal, chosen, tau_m)
    assert torch.allclose(got, expected, atol=1e-5)


def test_single_legal_action_gives_log_pi_zero():
    advantage = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    legal = torch.tensor([[False, False, True, False]])
    log_z = log_partition(advantage, legal, tau_m=0.03)
    chosen_advantage = advantage[:, 2]
    log_pi = munchausen_log_policy(chosen_advantage, log_z, tau_m=0.03)
    assert torch.allclose(log_pi, torch.zeros(1), atol=1e-6)  # the only legal action has pi=1


def test_bonus_is_clipped():
    tau_m_log_pi = torch.tensor([-5.0, -0.5, 0.0, 1.0])
    bonus = munchausen_bonus(tau_m_log_pi, alpha=0.9, clip_lo=-1.0, clip_hi=0.0)
    assert torch.allclose(bonus, torch.tensor([-0.9, -0.45, 0.0, 0.0]))
