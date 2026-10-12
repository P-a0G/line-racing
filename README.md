# line-racing ("Tron"/lumicycle) bot

CodinGame contest: a 30x20 grid, 2 or 3 lumicycles leaving a solid light
trail behind them, playing in round-robin turn order (not simultaneously —
see [rules.md](rules.md), whose own worked example proves this). Last
lumicycle standing wins; there is no draw.

## Where this stands

Two earlier from-scratch attempts (both in the nested `reinforcement_learning/`
directory — a separate git repo pointing at the same remote, kept as a
read-only historical record, not built on) didn't produce a strong bot:

- A **tabular Q-learning/TD(V)** agent (`reinforcement_learning/q_learning.py`)
  trained for up to 750k self-play games, but the state-table CSVs it
  produced (up to 4.5 million distinct rows from only 150k games) show the
  exact-position encoding essentially never recurs — a classic curse-of-
  dimensionality failure, not evidence against Q-learning as a paradigm.
- An **MCTS** bot (`IA/MCTS.py`) was abandoned before selection/expansion/
  backprop were ever implemented — it never actually ran.

This repo instead ports the method **summer_challenge_2026** used to copy
contest-winner "Indrill"'s approach for a different CodinGame game: a C++
rules engine for fast self-play (via pybind11), off-policy distributional
Q-learning (IQN + Munchausen + NoisyNets + prioritized n-step replay, the
"Beyond The Rainbow" family) with a CNN over the board, and a pure-numpy
single-file CG submission. A CNN function approximator directly fixes the
tabular approach's root-cause failure (generalizing across similar boards
instead of memorizing exact ones), and the generic BTR machinery transfers
directly — only the game-specific pieces (feature engineering, action
decode, reward shaping) needed redesigning for Tron's much simpler 4-way
action space and fixed board size. See
`.claude/plans/bright-baking-flask.md` (on the machine this was built on) or
`git log` for the full design reasoning.

A real training run hasn't been completed yet — see `training/tron_rl/run/`
(once one exists) for `train_log.txt`/`eval_metrics.csv` before assuming
anything below is accurate about model strength.

## Repository layout

```
legacy/            A copied reference (new_IA.py's flood-fill heuristic) from
                    the nested reinforcement_learning/ repo, kept only so the
                    C++ port in simulator/Heuristic.hpp has a committed diff target.
simulator/         C++ rules engine + a flood-fill heuristic + a Voronoi
                    territory map, exposed to Python via pybind11 as the
                    `tronsim` module. Used for fast self-play stepping,
                    as one of the self-play opponent policies, and (via
                    arena.py) for local evaluation.
training/tron_rl/  The training pipeline: features, network, reward, replay
                    buffer, learner, self-play collection, training loop,
                    and a local arena. Run train.py to train.
submission/        The actual CG submission: pure numpy (no torch), assembled
                    into one standalone file by build_submission.py.
```

### `simulator/`

| Path | Role |
|---|---|
| `Engine.hpp` | Core rules engine: `GameState`, `stepPlayer()` (one player's sub-turn — collision/elimination/round-robin advance). |
| `Heuristic.hpp` | `floodFillArea()` (multi-source BFS, ties favor the querying side) and `heuristicAction()` — a C++ port of `legacy/new_IA.py`, used as a scripted self-play opponent. |
| `Voronoi.hpp` | Multi-source BFS territory-ownership map across every alive player — backs both a feature channel and the reward's potential function. |
| `pybind_module.cpp` | Exposes the above to Python as the `tronsim` module. |
| `build_pybind.sh` | One-command build of `tronsim*.so` (run from WSL). |
| `test_*.cpp` | C++ unit tests for the engine/heuristic/Voronoi. |

### `training/tron_rl/` — the BTR pipeline

| Path | Role |
|---|---|
| `obs.py` (→ `submission/tron_infer/features_core.py`) | Feature engineering: per-player trail/head/alive planes, flood-fill distance, Voronoi territory, articulation points (cut vertices — the core space-splitting skill in Tron), static distance/coordinate planes, turn/alive-count scalars. The submission's copy is the single source of truth; this just re-exports it. |
| `net.py` | The network: spectral-norm convs + ReZero residual blocks + GlobalPoolingBias, feeding an IQN quantile embedding and a single dense dueling head over the plain 4-way move set (no combinatorial action space here, unlike the rail game). |
| `noisy.py` | NoisyNets linear layer (exploration; collapses to a plain linear at eval/export). |
| `reward.py` | Voronoi-territory-fraction potential + terminal ±1 — zero-sum by construction, so it generalizes to 2 or 3 players without an explicit "me minus foe" subtraction. |
| `munchausen.py` | The Munchausen RL log-policy term — a plain masked logsumexp over ≤4 actions, exact (no truncation needed, unlike the rail game's combinatorial version). |
| `replay.py` | Prioritized experience replay (sum-tree) with n-step returns over variable-length trajectories (flat circular transition arrays, not fixed-length trajectory slots — line-racing episodes don't share one fixed length). |
| `learner.py` | The IQN/Munchausen gradient step: online + target `DuelingIQNNet`, quantile-Huber loss, hard target sync. |
| `opponents.py` | Self-play opponent mixing: current network / frozen snapshot pool / the C++ flood-fill heuristic. |
| `collect.py` | Auto-reset vectorized self-play: `num_slots` in-flight games persist across calls, each tick batching every slot's current mover into one forward pass and immediately respawning any slot whose game just finished. |
| `train.py` | **Run this to train.** Writes `checkpoint.pt`/`train_log.txt`/`train_metrics.csv`/`eval_metrics.csv` under `--out-dir`. |
| `arena.py` | Local head-to-head evaluation between any mix of checkpoint/snapshot/heuristic/random policies. |
| `export.py` | Extracts a trained checkpoint's weights (folding spectral norm + ReZero, dropping NoisyNet sigma) into the plain-numpy format the submission uses. |
| `tests/` | The test suite — engine-level behavior via `tronsim`, feature/reward correctness, network parity, Munchausen exactness, replay buffer, learner, collector, numpy/torch forward-pass parity, weight codec roundtrip, CG protocol parsing (against `rules.md`'s own worked example), and a full training-loop smoke test. |

### `submission/` — the actual CG submission

| Path | Role |
|---|---|
| `tron_infer/features_core.py` | The feature engineering source of truth (see above). |
| `tron_infer/forward.py` | Pure-numpy mirror of `net.py`'s forward pass — verified to match it exactly (1e-3) across all-legal, some-illegal, and fully-boxed-in cases. |
| `tron_infer/weights_codec.py` | Quantize/pack/unpack a trained network's weights (lzma + stdlib base64 only — this network is small enough that the rail game's pluggable base91/bmp15 codecs would solve a problem that doesn't exist here). |
| `tron_infer/weights_blob.py` | **Generated** by `training/tron_rl/export.py` — not checked in until a checkpoint has been exported. |
| `cg_main.py` | The CG stdin/stdout protocol loop: reconstructs each player's full trail from the spawn+head-only protocol, decides an action, prints it. |
| `build_submission.py` | Concatenates everything above into one standalone `submission.py` (paste this into the CG IDE), safely stripping local imports/docstrings/comments. |

## One-time setup

Everything in `simulator/` and `training/` requires a C++ toolchain that
isn't available natively on this machine — **all of it runs through WSL**
(`wsl.exe -d Debian -- bash -lc "..."`).

```bash
wsl.exe -d Debian -- bash -lc "cd /mnt/c/Users/.../line-racing && \
  python3 -m venv training/.venv && \
  source training/.venv/bin/activate && \
  pip install pybind11 numpy torch --index-url https://download.pytorch.org/whl/cpu && \
  pip install pytest"
```

## Building the simulator

```bash
wsl.exe -d Debian -- bash -lc "bash /mnt/c/Users/.../line-racing/simulator/build_pybind.sh"
```

Re-run this after editing anything in `simulator/*.hpp` or `pybind_module.cpp`
— Python won't see C++ changes until the `.so` is rebuilt (a running training
process keeps its old in-memory mapping; only a fresh `import tronsim` in a
new process picks up the rebuilt one).

## Running the tests

```bash
# C++ (from simulator/)
g++ -std=c++17 -O2 -o test_engine test_engine.cpp && ./test_engine
g++ -std=c++17 -O2 -o test_heuristic test_heuristic.cpp && ./test_heuristic
g++ -std=c++17 -O2 -o test_voronoi test_voronoi.cpp && ./test_voronoi

# Python (from training/, after `source .venv/bin/activate`)
python3 -m pytest tron_rl/tests/ -q
```

## Running training

```bash
wsl.exe -d Debian -- bash -lc "cd .../training && source .venv/bin/activate && \
  python3 -m tron_rl.train --hours 12 --mix 0,0,1 --out-dir tron_rl/run"
```

`--mix current,snapshot,heuristic` controls self-play opponent mixing (falls
back toward `current` automatically until the snapshot pool has anything in
it) — start with `0,0,1` (pure heuristic bootstrap), then switch to
something like `0.5,0.3,0.2` once eval win-rate against the heuristic is
trending up, per summer_challenge_2026's own documented sequencing.
`--player-count-mix` controls the 2-vs-3-player sampling ratio. `--resume`
continues from `--out-dir`'s `checkpoint.pt`. See
`python3 -m tron_rl.train --help` for the rest.

## Running the arena

```bash
python3 -m tron_rl.arena --policies checkpoint:tron_rl/run/checkpoint.pt,heuristic --games 200
```

## Exporting a submission

```bash
python3 -c "from tron_rl.export import export_checkpoint; \
  export_checkpoint('tron_rl/run/checkpoint.pt', {'channels': 24, 'blocks': 2}, \
                     '../submission/tron_infer/weights_blob.py')"
cd ../submission && python3 build_submission.py
```

Paste the resulting `submission/submission.py` into the CodinGame IDE.

## Open questions / next steps

- No real training run has completed yet — the pipeline is built and unit
  tested end-to-end, but hasn't been judged for actual playing strength.
- Minor assumptions worth revisiting against real CG matches if play looks
  off: uniform-random spawn placement (rules.md doesn't specify a minimum
  spawn distance), and the turn-number feature's proxy value inside
  `cg_main.py` (it can't observe `simulator/Engine.hpp`'s exact `move_count`
  from the protocol alone, so it approximates).
- Ablations (reward weighting between the dense Voronoi term and the
  terminal bonus, n-step length, which feature channels actually matter)
  haven't been run — they need a meaningfully-trained checkpoint first.
