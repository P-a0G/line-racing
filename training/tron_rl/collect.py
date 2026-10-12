"""Self-play collection.

Unlike summer_challenge_2026/training/rlx/collect.py -- which runs a
lockstep batch of episodes to completion in one call, safe there since every
episode is a fixed, short MAX_TURNS=100 -- a line-racing episode ends the
moment a player is eliminated, at a length that varies per game (and per
seat: only the mover's own observation changes on a given engine step, since
turns are round-robin, not simultaneous). Lockstepping whole episodes here
would waste the batch on already-finished slots waiting out the slowest
game, and would throw away a long game's progress every call.

Collector instead holds `num_slots` parallel in-flight games PERSISTENTLY
across calls to collect(): each tick, it gathers the current mover from
every slot into one batched network forward pass (grouped by policy kind,
same current/snapshot/heuristic split as the rail game's collector), steps
every slot by one sub-turn, and immediately respawns any slot whose game
just finished -- sampling a fresh player count and fresh seat kinds for it --
so the batch stays full and no slot ever sits idle waiting for others.

Each seat of each game is independently assigned a policy kind via
opponents.sample_policy_kind (current/snapshot/heuristic). Every seat's
complete trajectory is always recorded regardless of which kind drove it --
this is off-policy learning, so a transition is valid to learn from no
matter whose policy produced the action (see opponents.py).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "simulator"))
import tronsim as ts  # noqa: E402

from tron_rl.obs import build_observation, state_to_arrays
from tron_rl.reward import potential, reward_from_potentials, terminal_bonus
from tron_rl.replay import Trajectory
from tron_rl.opponents import sample_policy_kind

DIRECTION_ENUM = [ts.UP, ts.DOWN, ts.LEFT, ts.RIGHT]


def _network_decide_batch(net, obs_batch, num_taus, device, generator=None):
    """obs_batch: list of (obs: (C,H,W) float32, legal: (4,) bool). Returns a
    list of int action indices chosen by masked argmax over the mean-Q
    across `num_taus` fresh random quantile samples -- matching how the
    target net's greedy action is chosen in learner.py, just with this net's
    live weights and, if net.training, NoisyNet exploration. No decode
    module is needed here, unlike the rail game's knapsack/region decode:
    the action space is already just 4 masked slots."""
    M = len(obs_batch)
    grids = np.stack([o[0] for o in obs_batch])
    legal = np.stack([o[1] for o in obs_batch])
    grid_t = torch.from_numpy(grids).float().to(device)
    legal_t = torch.from_numpy(legal).to(device)
    taus = torch.rand(M, num_taus, device=device, generator=generator)
    with torch.no_grad():
        out = net(grid_t, legal_t, taus)
        action = out["q"].mean(dim=1).argmax(dim=-1)
    return action.cpu().tolist()


class Collector:
    def __init__(self, num_slots, player_count_mix=(0.5, 0.5)):
        self.num_slots = num_slots
        self.player_count_mix = player_count_mix
        self.games = [None] * num_slots
        self.seat_kinds = [None] * num_slots     # [slot] -> [(kind,payload), ...] per player id
        self.trajectories = [None] * num_slots   # [slot] -> [Trajectory, ...] per player id
        self.prev_phi = [None] * num_slots       # [slot] -> [float or None, ...] per player id
        self.last_action = [None] * num_slots    # [slot] -> [int or None, ...] per player id

    def _respawn_slot(self, slot, cpp_rng, rng, snapshot_pool, mix):
        n = 2 if rng.random() < self.player_count_mix[0] else 3
        self.games[slot] = ts.random_spawn(n, cpp_rng)
        self.seat_kinds[slot] = [sample_policy_kind(rng, snapshot_pool, mix) for _ in range(n)]
        self.trajectories[slot] = [Trajectory() for _ in range(n)]
        self.prev_phi[slot] = [None] * n
        self.last_action[slot] = [None] * n

    def _finalize_slot(self, slot):
        """The game in `slot` just became terminal: close out every seat's
        dangling last action (reward = its dense delta to the terminal state
        plus the win/loss bonus -- see reward.terminal_bonus) and hand back
        whichever trajectories actually got at least one action."""
        state = self.games[slot]
        grid, heads, alive = state_to_arrays(state)
        winner = ts.winner(state)
        finished = []
        for p in range(state.num_players):
            traj = self.trajectories[slot][p]
            final_phi = potential(grid, heads, alive, p)
            final_obs = build_observation(grid, heads, alive, p, state.move_count)
            traj.append_observation(final_obs, np.zeros(4, dtype=bool))
            won = (p == winner)
            reward = reward_from_potentials(self.prev_phi[slot][p], final_phi) + terminal_bonus(won, not won)
            traj.append_action(self.last_action[slot][p], reward)
            if traj.length > 0:
                finished.append(traj)
        return finished, state.move_count

    def collect(self, learner, cpp_rng, snapshot_pool, mix, num_decisions,
                num_taus=8, scratch_net=None, rng=None, train_mode=True):
        """Runs `num_decisions` sub-turn decisions (one batched network
        forward pass per decision, across every slot's current mover).
        Returns (trajectories, stats): trajectories is every Trajectory
        completed during this call (from slots whose game finished at some
        point mid-call); stats has aggregate info for train.py's logging."""
        rng = rng or np.random.default_rng()
        device = learner.device
        learner.online.train(train_mode)
        if scratch_net is None:
            scratch_net = type(learner.online)(**learner.net_kwargs).to(device)
            scratch_net.eval()

        for slot in range(self.num_slots):
            if self.games[slot] is None:
                self._respawn_slot(slot, cpp_rng, rng, snapshot_pool, mix)

        completed = []
        games_finished = 0
        turns_at_finish = []
        player_counts = {2: 0, 3: 0}

        for _ in range(num_decisions):
            movers, obs_list = [], []
            for slot in range(self.num_slots):
                state = self.games[slot]
                p = state.next_to_move
                grid, heads, alive = state_to_arrays(state)
                obs = build_observation(grid, heads, alive, p, state.move_count)
                legal = np.array(ts.legal_moves(state, p))
                self.trajectories[slot][p].append_observation(obs, legal)
                phi = potential(grid, heads, alive, p)
                if self.prev_phi[slot][p] is not None:
                    self.trajectories[slot][p].append_action(
                        self.last_action[slot][p], reward_from_potentials(self.prev_phi[slot][p], phi))
                self.prev_phi[slot][p] = phi
                movers.append(p)
                obs_list.append((obs, legal))

            actions = [None] * self.num_slots
            current_slots, heuristic_slots = [], []
            by_snapshot = {}
            for slot in range(self.num_slots):
                kind, payload = self.seat_kinds[slot][movers[slot]]
                if kind == "current":
                    current_slots.append(slot)
                elif kind == "snapshot":
                    by_snapshot.setdefault(id(payload), (payload, []))[1].append(slot)
                else:
                    heuristic_slots.append(slot)

            if current_slots:
                decided = _network_decide_batch(
                    learner.online, [obs_list[s] for s in current_slots], num_taus, device)
                for slot, a in zip(current_slots, decided):
                    actions[slot] = a

            for state_dict, slots in by_snapshot.values():
                scratch_net.load_state_dict(state_dict)
                decided = _network_decide_batch(scratch_net, [obs_list[s] for s in slots], num_taus, device)
                for slot, a in zip(slots, decided):
                    actions[slot] = a

            for slot in heuristic_slots:
                move = self.seat_kinds[slot][movers[slot]][1](self.games[slot], movers[slot])
                actions[slot] = DIRECTION_ENUM.index(move)

            for slot in range(self.num_slots):
                p = movers[slot]
                self.last_action[slot][p] = actions[slot]
                ts.step_player(self.games[slot], p, DIRECTION_ENUM[actions[slot]])

                if ts.is_terminal(self.games[slot]):
                    player_counts[self.games[slot].num_players] += 1
                    finished, turns = self._finalize_slot(slot)
                    completed.extend(finished)
                    games_finished += 1
                    turns_at_finish.append(turns)
                    self.games[slot] = None

            for slot in range(self.num_slots):
                if self.games[slot] is None:
                    self._respawn_slot(slot, cpp_rng, rng, snapshot_pool, mix)

        stats = {
            "games_finished": games_finished,
            "mean_turns": float(np.mean(turns_at_finish)) if turns_at_finish else None,
            "player_counts": player_counts,
        }
        return completed, stats
