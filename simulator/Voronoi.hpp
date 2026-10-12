#pragma once
#include "Engine.hpp"
#include <array>
#include <deque>
#include <utility>

// Territory-ownership map used both as a CNN feature channel and as the
// reward's potential function (fraction of contested space each player
// currently "owns"). Shares the same multi-source-BFS idea as
// Heuristic.hpp's floodFillArea(), generalized from "me vs. everyone else"
// to a full N-way partition.
namespace sim {

// Every free cell is assigned to whichever alive player's head reaches it
// first via BFS; an exact-depth tie goes to the lower player index, since
// all heads are enqueued at depth 0 in player-index order (FIFO). Cells that
// are already occupied (trail) or unreachable from any alive player stay
// NONE.
inline std::array<int8_t, WIDTH * HEIGHT> voronoiOwnership(const GameState& s)
{
    std::array<int8_t, WIDTH * HEIGHT> owner;
    owner.fill(NONE);
    std::array<bool, WIDTH * HEIGHT> visited{};
    std::deque<std::pair<int, int8_t>> queue;

    for (int p = 0; p < s.numPlayers; p++) {
        if (!s.alive[p]) continue;
        int cell = s.headCell[p];
        visited[cell] = true;
        queue.push_back({cell, static_cast<int8_t>(p)});
    }

    while (!queue.empty()) {
        auto [cell, player] = queue.front();
        queue.pop_front();
        int x = cell % WIDTH, y = cell / WIDTH;
        for (int d = 0; d < 4; d++) {
            int nx = x + DX[d], ny = y + DY[d];
            if (!GameState::inBounds(nx, ny)) continue;
            int ncell = GameState::idx(nx, ny);
            if (visited[ncell] || s.cellOwner[ncell] != NONE) continue;
            visited[ncell] = true;
            owner[ncell] = player;
            queue.push_back({ncell, player});
        }
    }
    return owner;
}

inline std::array<int, MAX_PLAYERS> voronoiCounts(const GameState& s)
{
    std::array<int8_t, WIDTH * HEIGHT> owner = voronoiOwnership(s);
    std::array<int, MAX_PLAYERS> counts{};
    for (int8_t o : owner) if (o != NONE) counts[o]++;
    return counts;
}

} // namespace sim
