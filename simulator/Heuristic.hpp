#pragma once
#include "Engine.hpp"
#include <array>
#include <deque>
#include <utility>
#include <vector>

// C++ port of legacy/new_IA.py's n_accessible()/heuristicAction() flood-fill
// bot -- used both as a fast scripted self-play opponent (mirroring how
// summer_challenge_2026/simulator/Candidates.hpp's heuristicAction plays that
// role) and, via floodFillArea(), as a feature-engineering/reward primitive.
namespace sim {

// Multi-source BFS: `start` and every entry in `opponents` are enqueued at
// depth 0, `start` first. On an exact tie in BFS depth, `start` claims the
// cell because its expansion step runs before the opponents' same-depth
// expansion (FIFO order) -- the same tie-break new_IA.py's n_accessible()
// gets for free by appending opponents to the deque after the start cell.
// Returns the number of free cells `start` reaches strictly no later than
// any opponent.
inline int floodFillArea(const GameState& s, int start, const std::vector<int>& opponents)
{
    std::array<bool, WIDTH * HEIGHT> visited{};
    std::deque<std::pair<int, bool>> queue;

    queue.push_back({start, true});
    visited[start] = true;
    for (int opp : opponents) {
        // An opponent's head is always already part of their own trail in
        // s.cellOwner, so the inner-loop occupancy check would exclude it
        // from anyone's count regardless; this upfront mark is just dedup
        // against passing the same cell twice (or `opp == start`).
        if (opp == NONE || visited[opp]) continue;
        queue.push_back({opp, false});
        visited[opp] = true;
    }

    int count = 0;
    while (!queue.empty()) {
        auto [cell, mine] = queue.front();
        queue.pop_front();
        int x = cell % WIDTH, y = cell / WIDTH;
        for (int d = 0; d < 4; d++) {
            int nx = x + DX[d], ny = y + DY[d];
            if (!GameState::inBounds(nx, ny)) continue;
            int ncell = GameState::idx(nx, ny);
            if (visited[ncell] || s.cellOwner[ncell] != NONE) continue;
            visited[ncell] = true;
            if (mine) count++;
            queue.push_back({ncell, mine});
        }
    }
    return count;
}

// Picks the legal move that maximizes floodFillArea() from the resulting
// head position against the other alive players' current (unmoved) heads --
// a one-ply space-maximizing heuristic, not a search. Falls back to UP if no
// move is legal (matches new_IA.py defaulting to 'UP' when fully boxed in;
// the move is illegal either way, so the exact fallback choice doesn't
// matter for play, only for having a deterministic return value).
inline Direction heuristicAction(const GameState& s, int playerId)
{
    std::array<bool, 4> legal = legalMoves(s, playerId);

    std::vector<int> opponents;
    for (int p = 0; p < s.numPlayers; p++) {
        if (p != playerId && s.alive[p]) opponents.push_back(s.headCell[p]);
    }

    int head = s.headCell[playerId];
    int hx = head % WIDTH, hy = head / WIDTH;

    Direction best = UP;
    int bestArea = -1;
    for (int d = 0; d < 4; d++) {
        if (!legal[d]) continue;
        int candidate = GameState::idx(hx + DX[d], hy + DY[d]);
        int area = floodFillArea(s, candidate, opponents);
        if (area > bestArea) {
            bestArea = area;
            best = static_cast<Direction>(d);
        }
    }
    return best;
}

} // namespace sim
