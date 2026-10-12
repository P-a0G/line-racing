"""Observation/feature engineering -- the single source of truth imported
unmodified by both training (via training/tron_rl/obs.py) and the CG
submission, so the network never sees features computed two different ways
(the same anti-drift reasoning summer_challenge_2026/submission/rlx_infer/
features_core.py documents for that project).

Pure numpy, no torch/scipy/pybind11 dependency -- this file must run inside
the CodinGame judge's plain Python sandbox.
"""
import numpy as np
from collections import deque

WIDTH = 30
HEIGHT = 20
MAX_OPPONENTS = 2  # line-racing supports 2 or 3 total players -> at most 2 opponents

CHANNEL_NAMES = [
    "my_trail", "my_head",
    "opp1_trail", "opp1_head", "opp1_alive",
    "opp2_trail", "opp2_head", "opp2_alive",
    "free_cell_mask", "dist_to_wall",
    "my_flood_dist", "opp1_flood_dist", "opp2_flood_dist",
    "voronoi_mine", "voronoi_opp1", "voronoi_opp2",
    "articulation_points",
    "x_coord", "y_coord", "turn_number", "num_alive",
]
NUM_CHANNELS = len(CHANNEL_NAMES)

_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _in_bounds(x, y):
    return 0 <= x < WIDTH and 0 <= y < HEIGHT


def _bfs_distance(grid_owner, start):
    """Unweighted BFS distance from `start` over free cells (grid_owner == -1),
    -1 for cells that are unreachable (including occupied ones)."""
    dist = np.full((HEIGHT, WIDTH), -1, dtype=np.int32)
    if start is None:
        return dist
    sx, sy = start
    dist[sy, sx] = 0
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        d = dist[y, x]
        for dx, dy in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            if _in_bounds(nx, ny) and dist[ny, nx] == -1 and grid_owner[ny, nx] == -1:
                dist[ny, nx] = d + 1
                q.append((nx, ny))
    return dist


def _voronoi_ownership(grid_owner, heads, alive):
    """Multi-source BFS: each free cell goes to whichever alive player's head
    reaches it first; ties go to the lower player id (heads are enqueued in
    id order, all at depth 0, so FIFO processing favors the lower id) --
    matches simulator/Voronoi.hpp's semantics exactly."""
    owner = np.full((HEIGHT, WIDTH), -1, dtype=np.int8)
    visited = np.zeros((HEIGHT, WIDTH), dtype=bool)
    q = deque()
    for pid, head in enumerate(heads):
        if head is None or not alive[pid]:
            continue
        x, y = head
        if visited[y, x]:
            continue
        visited[y, x] = True
        q.append((x, y, pid))
    while q:
        x, y, pid = q.popleft()
        for dx, dy in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            if _in_bounds(nx, ny) and not visited[ny, nx] and grid_owner[ny, nx] == -1:
                visited[ny, nx] = True
                owner[ny, nx] = pid
                q.append((nx, ny, pid))
    return owner


def _articulation_points(free_mask):
    """Cut vertices of the free-cell adjacency graph: free cells whose
    removal would split the remaining free space into more components --
    the core "space-splitting" skill in Tron-like games. Iterative Tarjan's
    algorithm (recursion would risk Python's call-stack limit on a long,
    corridor-shaped free region)."""
    def idx(x, y):
        return y * WIDTH + x

    n = WIDTH * HEIGHT
    free_flat = free_mask.reshape(-1)
    adj = [[] for _ in range(n)]
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if not free_mask[y, x]:
                continue
            i = idx(x, y)
            for dx, dy in _NEIGHBORS:
                nx, ny = x + dx, y + dy
                if _in_bounds(nx, ny) and free_mask[ny, nx]:
                    adj[i].append(idx(nx, ny))

    disc = [-1] * n
    low = [-1] * n
    is_art = [False] * n
    timer = 0
    sentinel = object()

    for s in range(n):
        if not free_flat[s] or disc[s] != -1:
            continue
        disc[s] = low[s] = timer
        timer += 1
        root_children = 0
        stack = [(s, -1, iter(adj[s]))]

        while stack:
            u, parent, it = stack[-1]
            v = next(it, sentinel)
            if v is sentinel:
                stack.pop()
                if parent != -1:
                    low[parent] = min(low[parent], low[u])
                    if parent != s and low[u] >= disc[parent]:
                        is_art[parent] = True
                continue
            if disc[v] == -1:
                disc[v] = low[v] = timer
                timer += 1
                if parent == -1:
                    root_children += 1
                stack.append((v, u, iter(adj[v])))
            elif v != parent:
                low[u] = min(low[u], disc[v])

        if root_children > 1:
            is_art[s] = True

    return np.array(is_art, dtype=bool).reshape(HEIGHT, WIDTH)


_DIST_TO_WALL = np.minimum.reduce([
    np.tile(np.arange(WIDTH), (HEIGHT, 1)),
    np.tile(WIDTH - 1 - np.arange(WIDTH), (HEIGHT, 1)),
    np.tile(np.arange(HEIGHT).reshape(-1, 1), (1, WIDTH)),
    np.tile(HEIGHT - 1 - np.arange(HEIGHT).reshape(-1, 1), (1, WIDTH)),
]).astype(np.float32) / 10.0

_X_COORD = np.tile(np.arange(WIDTH, dtype=np.float32) / (WIDTH - 1), (HEIGHT, 1))
_Y_COORD = np.tile((np.arange(HEIGHT, dtype=np.float32) / (HEIGHT - 1)).reshape(-1, 1), (1, WIDTH))

_FLOOD_DIST_NORM = float(WIDTH + HEIGHT)  # a BFS distance can never exceed this on a 30x20 board
_TURN_NORM = 300.0  # generous upper bound on a real game's length (max possible is WIDTH*HEIGHT)


def build_observation(grid_owner, heads, alive, my_id, turn):
    """Builds the (NUM_CHANNELS, HEIGHT, WIDTH) float32 observation for
    player `my_id`, always ordered [me, opp1, opp2] regardless of my_id's
    actual seat so the network never needs to know its own seat index.

    grid_owner: (HEIGHT, WIDTH) int array, -1 free else occupying player id.
    heads: list of (x, y) or None, indexed by player id (length 2 or 3).
    alive: list of bool, same indexing as heads.
    """
    num_players = len(heads)
    opp_ids = [p for p in range(num_players) if p != my_id]
    while len(opp_ids) < MAX_OPPONENTS:
        opp_ids.append(None)

    free_mask = (grid_owner == -1)
    voronoi = _voronoi_ownership(grid_owner, heads, alive)

    channels = {}
    channels["my_trail"] = (grid_owner == my_id).astype(np.float32)
    my_head_plane = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    if alive[my_id] and heads[my_id] is not None:
        mx, my = heads[my_id]
        my_head_plane[my, mx] = 1.0
    channels["my_head"] = my_head_plane
    channels["my_flood_dist"] = np.clip(_bfs_distance(grid_owner, heads[my_id]), -1, None).astype(np.float32)
    channels["my_flood_dist"][channels["my_flood_dist"] < 0] = _FLOOD_DIST_NORM
    channels["my_flood_dist"] /= _FLOOD_DIST_NORM
    channels["voronoi_mine"] = (voronoi == my_id).astype(np.float32)

    for slot, opp_id in enumerate(opp_ids, start=1):
        is_live_opp = opp_id is not None and alive[opp_id]
        trail = (grid_owner == opp_id).astype(np.float32) if is_live_opp else np.zeros((HEIGHT, WIDTH), np.float32)
        head_plane = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        flood = np.full((HEIGHT, WIDTH), _FLOOD_DIST_NORM, dtype=np.float32)
        voronoi_plane = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        if is_live_opp:
            ox, oy = heads[opp_id]
            head_plane[oy, ox] = 1.0
            flood = np.clip(_bfs_distance(grid_owner, heads[opp_id]), -1, None).astype(np.float32)
            flood[flood < 0] = _FLOOD_DIST_NORM
            voronoi_plane = (voronoi == opp_id).astype(np.float32)
        channels[f"opp{slot}_trail"] = trail
        channels[f"opp{slot}_head"] = head_plane
        channels[f"opp{slot}_alive"] = np.full((HEIGHT, WIDTH), 1.0 if is_live_opp else 0.0, dtype=np.float32)
        channels[f"opp{slot}_flood_dist"] = flood / _FLOOD_DIST_NORM
        channels[f"voronoi_opp{slot}"] = voronoi_plane

    channels["free_cell_mask"] = free_mask.astype(np.float32)
    channels["dist_to_wall"] = _DIST_TO_WALL
    channels["articulation_points"] = _articulation_points(free_mask).astype(np.float32)
    channels["x_coord"] = _X_COORD
    channels["y_coord"] = _Y_COORD
    channels["turn_number"] = np.full((HEIGHT, WIDTH), min(turn, _TURN_NORM) / _TURN_NORM, dtype=np.float32)
    num_alive = sum(1 for a in alive if a)
    channels["num_alive"] = np.full((HEIGHT, WIDTH), num_alive / 3.0, dtype=np.float32)

    return np.stack([channels[name] for name in CHANNEL_NAMES], axis=0)
