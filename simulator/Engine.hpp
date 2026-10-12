#pragma once
#include <array>
#include <cassert>
#include <cstdint>
#include <random>

// Tron/"lumicycle" engine per line-racing/rules.md: each round, players act
// one at a time in index order (confirmed by rules.md's own worked example --
// the turn-2 input already shows player 0 at its post-move position). So the
// atomic step primitive here is one player's sub-turn, not "apply N moves
// then resolve collisions" -- there is no simultaneous head-on case to
// arbitrate at all.
namespace sim {

constexpr int WIDTH = 30, HEIGHT = 20, MAX_PLAYERS = 3;
constexpr int8_t NONE = -1;
// Each successful move claims exactly one previously-free cell, so a real
// game can never exceed WIDTH*HEIGHT move sub-turns plus a handful of death
// sub-turns. This is a defensive circuit-breaker against an engine bug, not
// a rule of the real game -- see isTerminal()'s comment.
constexpr int MAX_MOVES = 1000;

enum Direction { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };
constexpr int DX[4] = {0, 0, -1, 1};
constexpr int DY[4] = {-1, 1, 0, 0};

struct GameState {
    int numPlayers = 0;
    std::array<int8_t, WIDTH * HEIGHT> cellOwner{}; // NONE or playerId (trail incl. head)
    std::array<int, MAX_PLAYERS> headCell{};
    std::array<int, MAX_PLAYERS> spawnCell{};
    std::array<bool, MAX_PLAYERS> alive{};
    int nextToMove = 0;
    int moveCount = 0; // total stepPlayer() calls so far; circuit-breaker use only

    static int idx(int x, int y) { return y * WIDTH + x; }
    static bool inBounds(int x, int y) { return x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT; }

    // Aggregate-initializing a GameState does NOT give cellOwner the NONE
    // sentinel (int8_t{} is 0, a valid playerId) -- always go through reset()
    // rather than relying on `GameState s;` alone, including in tests.
    void reset(int players)
    {
        numPlayers = players;
        cellOwner.fill(NONE);
        headCell.fill(NONE);
        spawnCell.fill(NONE);
        alive.fill(false);
        nextToMove = 0;
        moveCount = 0;
    }
};

inline int aliveCount(const GameState& s)
{
    int n = 0;
    for (int p = 0; p < s.numPlayers; p++) if (s.alive[p]) n++;
    return n;
}

// Eliminations happen one player at a time (see namespace comment), so the
// alive count can only ever step down by exactly one -- it always lands on
// exactly 1, never 0, in a real game. isTerminal() is therefore enough on its
// own to find the winner; the MAX_MOVES cutoff exists purely as a defensive
// fallback against an engine bug and should never fire in practice.
inline bool isTerminal(const GameState& s)
{
    return aliveCount(s) <= 1 || s.moveCount >= MAX_MOVES;
}

// NONE if the MAX_MOVES safety cutoff fired (should never happen -- see
// isTerminal()); otherwise the sole surviving player.
inline int winner(const GameState& s)
{
    if (aliveCount(s) != 1) return NONE;
    for (int p = 0; p < s.numPlayers; p++) if (s.alive[p]) return p;
    return NONE;
}

inline std::array<bool, 4> legalMoves(const GameState& s, int playerId)
{
    std::array<bool, 4> legal{};
    int cell = s.headCell[playerId];
    int x = cell % WIDTH, y = cell / WIDTH;
    for (int d = 0; d < 4; d++) {
        int nx = x + DX[d], ny = y + DY[d];
        legal[d] = GameState::inBounds(nx, ny) && s.cellOwner[GameState::idx(nx, ny)] == NONE;
    }
    return legal;
}

// rules.md doesn't specify a minimum spawn distance -- uniform-random
// distinct cells is a placeholder, cheap to revisit if real matches turn out
// to use a different distribution.
inline GameState randomSpawn(int numPlayers, std::mt19937& rng)
{
    GameState s;
    s.reset(numPlayers);
    std::uniform_int_distribution<int> pick(0, WIDTH * HEIGHT - 1);
    for (int p = 0; p < numPlayers; p++) {
        int cell;
        do { cell = pick(rng); } while (s.cellOwner[cell] != NONE);
        s.cellOwner[cell] = static_cast<int8_t>(p);
        s.headCell[p] = cell;
        s.spawnCell[p] = cell;
        s.alive[p] = true;
    }
    return s;
}

// The atomic engine step: player `playerId` (must be the current
// s.nextToMove and alive) moves one cell in direction `dir`. Returns true if
// this move eliminated the player. An elimination clears every cell that
// player owned back to NONE immediately -- including for whoever's turn
// comes next in the same round, matching rules.md ("son ruban de lumière
// disparaît et le jeu continue").
inline bool stepPlayer(GameState& s, int playerId, Direction dir)
{
    assert(playerId == s.nextToMove && s.alive[playerId]);

    int cell = s.headCell[playerId];
    int x = cell % WIDTH, y = cell / WIDTH;
    int nx = x + DX[dir], ny = y + DY[dir];

    bool died = !GameState::inBounds(nx, ny) || s.cellOwner[GameState::idx(nx, ny)] != NONE;
    if (died) {
        s.alive[playerId] = false;
        for (auto& owner : s.cellOwner) if (owner == static_cast<int8_t>(playerId)) owner = NONE;
    } else {
        int ncell = GameState::idx(nx, ny);
        s.cellOwner[ncell] = static_cast<int8_t>(playerId);
        s.headCell[playerId] = ncell;
    }

    s.moveCount++;
    do {
        s.nextToMove = (s.nextToMove + 1) % s.numPlayers;
    } while (!s.alive[s.nextToMove] && aliveCount(s) > 0);

    return died;
}

} // namespace sim
