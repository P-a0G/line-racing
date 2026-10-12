"""The BTR training loop entrypoint. Run this to train.

Writes checkpoint.pt/train_log.txt/train_metrics.csv/eval_metrics.csv under
--out-dir (default training/tron_rl/run/), matching summer_challenge_2026's
train.py convention.

Per that project's own documented sequencing: run this first bootstrapped
against the fixed heuristic only (--mix 0,0,1) to confirm the pipeline
actually learns, then switch to the full opponent mix once win rate is
trending up.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "simulator"))
import tronsim as ts  # noqa: E402

from tron_rl.learner import Learner
from tron_rl.replay import ReplayBuffer
from tron_rl.opponents import SnapshotPool
from tron_rl.collect import Collector
from tron_rl.obs import build_observation, state_to_arrays
from tron_rl.net import DEFAULT_CHANNELS, DEFAULT_BLOCKS

DIRECTION_ENUM = [ts.UP, ts.DOWN, ts.LEFT, ts.RIGHT]


def _decide_with_fixed_taus(net, state, player_id, taus, device):
    """Same decode as collect.py's _network_decide_batch, but for a single
    state and caller-supplied taus (fixed midpoints for eval, rather than
    fresh random samples) -- kept separate rather than adding a
    random-vs-fixed branch to the hot collection path."""
    grid, heads, alive = state_to_arrays(state)
    obs = build_observation(grid, heads, alive, player_id, state.move_count)
    legal = np.array(ts.legal_moves(state, player_id))
    grid_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    legal_t = torch.from_numpy(legal).unsqueeze(0).to(device)
    with torch.no_grad():
        out = net(grid_t, legal_t, taus.to(device))
        action = out["q"].mean(dim=1).argmax(dim=-1).item()
    return DIRECTION_ENUM[action]


def evaluate_vs_heuristic(learner, cpp_rng, num_games, num_taus, num_players):
    """Plays `num_games` `num_players`-player games with the online net
    (eval mode: no NoisyNet exploration, deterministic 8-midpoint taus
    matching submission-time inference) occupying one seat against the
    flood-fill heuristic in every other seat, rotating which seat the net
    plays. Returns (win_rate, mean_turns). Tracked separately per player
    count (see train.py's call sites) since a weak third player can
    "kingmake" who wins in a 3-player game, making a single blended win rate
    a noisier progress signal than the rail game's clean 2-player case."""
    learner.online.eval()
    midpoint_taus = torch.tensor([[(2 * k - 1) / (2 * num_taus) for k in range(1, num_taus + 1)]])

    wins, turns = 0, []
    for g in range(num_games):
        net_seat = g % num_players
        state = ts.random_spawn(num_players, cpp_rng)
        while not ts.is_terminal(state):
            p = state.next_to_move
            if p == net_seat:
                move = _decide_with_fixed_taus(learner.online, state, p, midpoint_taus, learner.device)
            else:
                move = ts.heuristic_action(state, p)
            ts.step_player(state, p, move)
        wins += ts.winner(state) == net_seat
        turns.append(state.move_count)
    learner.online.train()
    return wins / num_games, float(np.mean(turns))


def _save_checkpoint(path, learner, pool, it):
    torch.save({
        "online": learner.online.state_dict(), "target": learner.target.state_dict(),
        "optimizer": learner.optimizer.state_dict(), "grad_steps": learner.grad_steps,
        "iter": it, "snapshots": list(pool.snapshots), "net_kwargs": learner.net_kwargs,
    }, path)


def main(hours=None, max_iterations=None, resume=False, num_slots=32, decisions_per_collect=200,
          min_sampling_size=4000, grad_steps_per_collect=100, batch_size=64, mix=(0.5, 0.3, 0.2),
          player_count_mix=(0.5, 0.5), buffer_capacity_transitions=100_000,
          eval_every=20, checkpoint_every=20, snapshot_every=50, eval_games=20,
          net_kwargs=None, out_dir=None, seed=0):
    net_kwargs = net_kwargs or {"channels": DEFAULT_CHANNELS, "blocks": DEFAULT_BLOCKS}
    out_dir = Path(out_dir) if out_dir else Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"
    log_path = out_dir / "train_log.txt"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    learner = Learner(net_kwargs=net_kwargs, batch_size=batch_size)
    buffer = ReplayBuffer(capacity_transitions=buffer_capacity_transitions, n_step=3, gamma=1.0)
    log(f"replay buffer: {buffer.nbytes() / 1e9:.2f} GB for {buffer_capacity_transitions} transitions")
    pool = SnapshotPool(max_size=20)
    collector = Collector(num_slots=num_slots, player_count_mix=player_count_mix)
    cpp_rng = ts.Rng(seed)
    rng = np.random.default_rng(seed)

    start_iter = 0
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=learner.device)
        learner.online.load_state_dict(ckpt["online"])
        learner.target.load_state_dict(ckpt["target"])
        learner.optimizer.load_state_dict(ckpt["optimizer"])
        learner.grad_steps = ckpt["grad_steps"]
        for sd in ckpt.get("snapshots", []):
            pool.add(sd)
        start_iter = ckpt["iter"] + 1
        log(f"resumed from {ckpt_path} at iteration {start_iter}")

    append = resume and start_iter > 0
    train_csv_f = open(out_dir / "train_metrics.csv", "a" if append else "w", newline="")
    train_csv = csv.writer(train_csv_f)
    eval_csv_f = open(out_dir / "eval_metrics.csv", "a" if append else "w", newline="")
    eval_csv = csv.writer(eval_csv_f)
    if not append:
        train_csv.writerow(["iter", "buffer_size", "loss", "mean_munchausen_bonus",
                              "games_finished", "mean_turns", "sec"])
        eval_csv.writerow(["iter", "win_rate_2p", "win_rate_3p", "mean_turns_2p", "mean_turns_3p"])
        train_csv_f.flush()
        eval_csv_f.flush()  # otherwise the header sits unflushed until the first real eval row (eval_every apart)

    t_start = time.time()
    it = start_iter
    stop_iter = None if max_iterations is None else start_iter + max_iterations
    try:
        while ((hours is None or (time.time() - t_start) < hours * 3600)
               and (stop_iter is None or it < stop_iter)):
            t0 = time.time()
            trajectories, stats = collector.collect(
                learner, cpp_rng, pool, mix, decisions_per_collect, num_taus=learner.num_taus, rng=rng)
            for traj in trajectories:
                buffer.add_trajectory(traj)

            losses, bonuses = [], []
            if len(buffer) >= min_sampling_size:
                for _ in range(grad_steps_per_collect):
                    batch = buffer.sample(batch_size, rng=rng)
                    result = learner.train_step(batch)
                    buffer.update_priorities(batch["indices"], result["td_error"])
                    losses.append(result["loss"])
                    bonuses.append(result["mean_munchausen_bonus"])

            sec = time.time() - t0
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            mean_bonus = float(np.mean(bonuses)) if bonuses else float("nan")
            mean_turns = stats["mean_turns"] if stats["mean_turns"] is not None else float("nan")
            log(f"iter {it}: buffer={len(buffer)} loss={mean_loss:.4f} "
                f"games={stats['games_finished']} mean_turns={mean_turns:.1f} sec={sec:.1f}")
            train_csv.writerow([it, len(buffer), mean_loss, mean_bonus,
                                  stats["games_finished"], mean_turns, sec])
            train_csv_f.flush()

            if (it + 1) % eval_every == 0:
                win2, turns2 = evaluate_vs_heuristic(learner, cpp_rng, eval_games, learner.num_taus, 2)
                win3, turns3 = evaluate_vs_heuristic(learner, cpp_rng, eval_games, learner.num_taus, 3)
                log(f"  eval: win_rate_2p={win2:.2f} win_rate_3p={win3:.2f}")
                eval_csv.writerow([it, win2, win3, turns2, turns3])
                eval_csv_f.flush()

            if (it + 1) % snapshot_every == 0:
                pool.add(learner.online.state_dict())
                log(f"  added snapshot to pool (size={len(pool)})")

            if (it + 1) % checkpoint_every == 0:
                _save_checkpoint(ckpt_path, learner, pool, it)
                log(f"  saved checkpoint to {ckpt_path}")

            it += 1
    finally:
        _save_checkpoint(ckpt_path, learner, pool, it - 1)
        log(f"final checkpoint saved to {ckpt_path} at iteration {it - 1}")
        train_csv_f.close()
        eval_csv_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-slots", type=int, default=32, help="parallel in-flight self-play games")
    parser.add_argument("--decisions-per-collect", type=int, default=200,
                          help="batched network decisions (one per slot each) per training iteration")
    parser.add_argument("--min-sampling-size", type=int, default=4000)
    parser.add_argument("--grad-steps-per-collect", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-capacity-transitions", type=int, default=100_000,
                          help="~25KB/transition; default 100k -> ~2.5GB full buffer, see replay.py's nbytes()")
    parser.add_argument("--mix", type=str, default="0.5,0.3,0.2", help="current,snapshot,heuristic")
    parser.add_argument("--player-count-mix", type=str, default="0.5,0.5", help="P(2 players),P(3 players)")
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--snapshot-every", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(hours=args.hours, max_iterations=args.iterations, resume=args.resume,
          num_slots=args.num_slots, decisions_per_collect=args.decisions_per_collect,
          min_sampling_size=args.min_sampling_size, grad_steps_per_collect=args.grad_steps_per_collect,
          batch_size=args.batch_size, buffer_capacity_transitions=args.buffer_capacity_transitions,
          mix=tuple(float(x) for x in args.mix.split(",")),
          player_count_mix=tuple(float(x) for x in args.player_count_mix.split(",")),
          eval_every=args.eval_every, checkpoint_every=args.checkpoint_every,
          snapshot_every=args.snapshot_every, eval_games=args.eval_games,
          net_kwargs={"channels": args.channels, "blocks": args.blocks},
          out_dir=args.out_dir, seed=args.seed)
