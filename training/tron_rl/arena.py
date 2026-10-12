"""Local arena: head-to-head matches between any mix of policies (a trained
checkpoint, a frozen snapshot slice of one, the C++ flood-fill heuristic, or
uniform-random legal moves) -- a tool summer_challenge_2026's own README
flags as missing for its new pipeline (its old arena.py only worked with the
removed C++ submission binaries). Since this plugs directly into the
pybind11 tronsim module's step_player/legal_moves, it's a thin driver loop,
not a separate simulator, and is cheap to have from iteration 1.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "simulator"))
import tronsim as ts  # noqa: E402

from tron_rl.net import DuelingIQNNet
from tron_rl.obs import build_observation, state_to_arrays

DIRECTION_ENUM = [ts.UP, ts.DOWN, ts.LEFT, ts.RIGHT]
NUM_TAUS_EVAL = 8


def _net_policy(net):
    """Deterministic 8-fixed-midpoint-tau decoding -- matches how
    train.py's evaluate_vs_heuristic and the real CG submission both decode,
    so arena results are comparable to eval_metrics.csv and to the shipped
    bot, not to training's randomly-sampled-tau exploration behavior."""
    midpoint_taus = torch.tensor([[(2 * k - 1) / (2 * NUM_TAUS_EVAL) for k in range(1, NUM_TAUS_EVAL + 1)]])

    def policy(state, player_id):
        grid, heads, alive = state_to_arrays(state)
        obs = build_observation(grid, heads, alive, player_id, state.move_count)
        legal = np.array(ts.legal_moves(state, player_id))
        grid_t = torch.from_numpy(obs).float().unsqueeze(0)
        legal_t = torch.from_numpy(legal).unsqueeze(0)
        with torch.no_grad():
            out = net(grid_t, legal_t, midpoint_taus)
            action = out["q"].mean(dim=1).argmax(dim=-1).item()
        return DIRECTION_ENUM[action]

    return policy


def _heuristic_policy(state, player_id):
    return ts.heuristic_action(state, player_id)


def _random_policy(rng):
    def policy(state, player_id):
        legal = ts.legal_moves(state, player_id)
        options = [d for d, ok in zip(DIRECTION_ENUM, legal) if ok]
        return rng.choice(options) if options else ts.UP
    return policy


def resolve_policy(spec, rng=None):
    """spec is 'heuristic', 'random', 'checkpoint:<path>' (the online net),
    or 'snapshot:<path>:<index>' (a frozen snapshot from that checkpoint's
    pool). Both checkpoint forms need `net_kwargs` saved alongside the
    weights (see train.py's _save_checkpoint) to rebuild the right-shaped
    network before loading them."""
    if spec == "heuristic":
        return _heuristic_policy
    if spec == "random":
        return _random_policy(rng or np.random.default_rng())
    if spec.startswith("checkpoint:"):
        ckpt = torch.load(spec.split(":", 1)[1], map_location="cpu")
        net = DuelingIQNNet(**ckpt["net_kwargs"])
        net.load_state_dict(ckpt["online"])
        net.eval()
        return _net_policy(net)
    if spec.startswith("snapshot:"):
        _, path, index = spec.split(":", 2)
        ckpt = torch.load(path, map_location="cpu")
        net = DuelingIQNNet(**ckpt["net_kwargs"])
        net.load_state_dict(ckpt["snapshots"][int(index)])
        net.eval()
        return _net_policy(net)
    raise ValueError(f"unknown policy spec: {spec!r}")


def play_match(policies, num_games, seed=0):
    """policies: list of callables (state, player_id) -> Direction, one per
    seat for this len(policies)-player match. Each game cyclically rotates
    which policy occupies which seat so no policy is penalized/favored by a
    fixed turn-order position. Returns (wins, mean_turns): wins is a list
    parallel to `policies` (indexed by POLICY, not by seat), mean_turns the
    average game length."""
    num_players = len(policies)
    cpp_rng = ts.Rng(seed)
    wins = [0] * num_players
    turns = []

    for g in range(num_games):
        rotation = [(i + g) % num_players for i in range(num_players)]  # seat i runs policies[rotation[i]]
        state = ts.random_spawn(num_players, cpp_rng)
        while not ts.is_terminal(state):
            p = state.next_to_move
            move = policies[rotation[p]](state, p)
            ts.step_player(state, p, move)
        winner_seat = ts.winner(state)
        if winner_seat != ts.NONE:
            wins[rotation[winner_seat]] += 1
        turns.append(state.move_count)

    return wins, float(np.mean(turns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=str, required=True,
                          help="comma-separated policy specs, one per seat (2 or 3), e.g. "
                               "'checkpoint:run/checkpoint.pt,heuristic'")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    specs = args.policies.split(",")
    policies = [resolve_policy(spec, rng) for spec in specs]
    wins, mean_turns = play_match(policies, args.games, args.seed)

    for spec, w in zip(specs, wins):
        print(f"{spec}: {w}/{args.games} wins ({w / args.games:.1%})")
    print(f"mean game length: {mean_turns:.1f} turns")
