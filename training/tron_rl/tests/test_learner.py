import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "simulator"))
import tronsim as ts  # noqa: E402

from tron_rl.obs import build_observation, state_to_arrays  # noqa: E402
from tron_rl.reward import potential, reward_from_potentials, terminal_bonus  # noqa: E402
from tron_rl.replay import ReplayBuffer, Trajectory  # noqa: E402
from tron_rl.learner import Learner  # noqa: E402

DIRECTIONS = [ts.UP, ts.DOWN, ts.LEFT, ts.RIGHT]


def collect_one_trajectory(seed, player_id=0):
    """Minimal single-seat collector for test data -- collect.py (a later
    milestone) will have the real, multi-seat, auto-reset version. Each
    iteration here appends the OBSERVATION for this turn and finalizes the
    PREVIOUS turn's action+reward (its dense delta is only known once we see
    the next observation); the dangling final action is closed out after the
    loop, once the terminal state's potential and win/loss are known."""
    rng = ts.Rng(seed)
    s = ts.random_spawn(2, rng)
    traj = Trajectory()
    prev_phi = None
    last_action = None

    while not ts.is_terminal(s):
        p = s.next_to_move
        if p == player_id:
            grid, heads, alive = state_to_arrays(s)
            obs = build_observation(grid, heads, alive, player_id, s.move_count)
            legal = np.array(ts.legal_moves(s, player_id))
            traj.append_observation(obs, legal)
            phi = potential(grid, heads, alive, player_id)
            if prev_phi is not None:
                traj.append_action(last_action, reward_from_potentials(prev_phi, phi))
            prev_phi = phi
            move = ts.heuristic_action(s, p)
            last_action = DIRECTIONS.index(move)
            ts.step_player(s, p, move)
        else:
            ts.step_player(s, p, ts.heuristic_action(s, p))

    assert last_action is not None  # a 2-player game always gives both seats at least one turn
    grid, heads, alive = state_to_arrays(s)
    final_phi = potential(grid, heads, alive, player_id)
    final_obs = build_observation(grid, heads, alive, player_id, s.move_count)
    traj.append_observation(final_obs, np.zeros(4, dtype=bool))  # terminal -- no further decision
    won = ts.winner(s) == player_id
    reward = reward_from_potentials(prev_phi, final_phi) + terminal_bonus(won=won, lost=not won)
    traj.append_action(last_action, reward)
    return traj


def build_fake_batch(batch_size=8):
    buf = ReplayBuffer(capacity_transitions=512, n_step=3, gamma=0.99)
    for seed in range(6):
        traj = collect_one_trajectory(seed)
        if traj.length > 0:
            buf.add_trajectory(traj)
    assert len(buf) >= batch_size
    return buf.sample(batch_size, rng=np.random.default_rng(0))


def make_learner():
    torch.manual_seed(0)
    return Learner(net_kwargs=dict(channels=8, blocks=2, gpb_blocks=(0,)),
                    batch_size=8, target_update_interval=2, num_taus=4)


def test_train_step_returns_finite_metrics():
    learner = make_learner()
    batch = build_fake_batch()
    out = learner.train_step(batch)
    assert np.isfinite(out["loss"])
    assert out["loss"] >= 0.0
    assert np.isfinite(out["td_error"]).all()
    assert out["td_error"].shape == (8,)
    assert np.isfinite(out["mean_munchausen_bonus"])


def test_online_parameters_change_after_a_step():
    learner = make_learner()
    batch = build_fake_batch()
    before = [p.detach().clone() for p in learner.online.parameters()]
    learner.train_step(batch)
    after = list(learner.online.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_target_network_syncs_on_schedule():
    learner = make_learner()  # target_update_interval=2
    batch = build_fake_batch()

    learner.train_step(batch)
    online_sd = learner.online.state_dict()
    target_sd = learner.target.state_dict()
    assert any(not torch.equal(online_sd[k], target_sd[k]) for k in online_sd)

    learner.train_step(batch)  # grad_steps now == 2 -> sync fires
    online_sd = learner.online.state_dict()
    target_sd = learner.target.state_dict()
    assert all(torch.equal(online_sd[k], target_sd[k]) for k in online_sd)


def test_noise_is_resampled_each_step_eval_mode_still_deterministic():
    learner = make_learner()
    batch = build_fake_batch()
    eps_before = learner.online.value_fc1.weight_epsilon.clone()
    learner.train_step(batch)
    eps_after = learner.online.value_fc1.weight_epsilon.clone()
    assert not torch.equal(eps_before, eps_after)
