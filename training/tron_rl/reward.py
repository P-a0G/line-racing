"""Reward shaping for the Tron BTR pipeline: potential-based shaping using
Voronoi-territory fraction (the same multi-source-BFS primitive the
observation's voronoi_* channels use) plus a terminal +-1 win/loss bonus.

Mirrors summer_challenge_2026's "potential + delta + terminal" shape
(Indrill's "terminal + delta_delta_score"), but needs no score/economy model
and is zero-sum by construction -- the fractions sum to 1 across alive
players, so it generalizes to 2 or 3 players without the rail game's
explicit "me minus foe" subtraction (which assumed exactly 2 seats).

    Phi_p(s) = voronoi_area(p) / sum_over_alive_q(voronoi_area(q))

The dense per-turn reward is Phi's delta; the terminal win/-loss bonus is a
separate piece (see terminal_bonus()) because, under this engine's
round-robin turn order, a game can end on a sub-turn where the surviving
player didn't act at all (their opponent eliminated themselves) -- so
collect.py applies it to whichever action actually closes out each
trajectory, rather than this module trying to decide that on its own.
"""
import numpy as np

import tron_rl  # noqa: F401  (side effect: puts submission/tron_infer on sys.path)
from features_core import _voronoi_ownership

TERMINAL_WEIGHT = 1.0


def potential(grid_owner, heads, alive, player_id):
    """Phi_p(s): player_id's fraction of currently-contested free space, in
    [0, 1]. 0.0 if player_id is already eliminated, or if no free cell is
    reachable by anyone (a fully sealed-off endgame)."""
    if not alive[player_id]:
        return 0.0
    owner = _voronoi_ownership(grid_owner, heads, alive)
    assigned = owner[owner >= 0]
    if assigned.size == 0:
        return 0.0
    counts = np.bincount(assigned, minlength=len(heads))
    return float(counts[player_id]) / float(assigned.size)


def reward_from_potentials(prev_phi, next_phi):
    """The dense shaping term alone: Phi_p(s') - Phi_p(s)."""
    return next_phi - prev_phi


def terminal_bonus(won, lost):
    """+1 if this action's trajectory ends in a win, -1 if it ends in this
    player's own elimination, 0.0 otherwise (won and lost are never both
    true: see Engine.hpp's isTerminal() -- a game can only ever end at
    exactly one survivor)."""
    assert not (won and lost)
    if won:
        return TERMINAL_WEIGHT
    if lost:
        return -TERMINAL_WEIGHT
    return 0.0
