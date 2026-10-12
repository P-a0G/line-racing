#include "Heuristic.hpp"
#include <iostream>
#include <cassert>

using namespace sim;
using namespace std;

int main()
{
    // Test 1: on a fully open board with no opponents, floodFillArea reaches
    // every other cell on the board.
    {
        GameState s;
        s.reset(1);
        int count = floodFillArea(s, GameState::idx(15, 10), {});
        assert(count == WIDTH * HEIGHT - 1); // every cell but the start itself
        cout << "Test 1 OK: open board, no opponents -> reaches all " << count << " other cells.\n";
    }

    // Test 2: symmetric tie-break. With start and the opponent on the same
    // row at opposite edges, every cell's BFS distance to each of them is
    // its Manhattan distance (no obstacles), so the board splits in half at
    // the midpoint column, ties going to `start` (it's enqueued first).
    // Columns x=0..14 (15 of 30) are mine; x=15..29 are the opponent's.
    {
        GameState s;
        s.reset(2);
        int start = GameState::idx(0, 10);
        int opp = GameState::idx(29, 10);
        s.cellOwner[start] = 0;
        s.cellOwner[opp] = 1;
        int count = floodFillArea(s, start, {opp});
        assert(count == 15 * HEIGHT - 1); // 15 mine-columns x 20 rows, minus the start cell itself
        cout << "Test 2 OK: symmetric split gives " << count << " cells, ties resolved to start.\n";
    }

    // Test 3: heuristicAction must steer away from a sealed dead-end corridor
    // even though it's the first cell reached, in favor of the open board.
    //
    //   Player head at (5,5). RIGHT leads into a 1-wide corridor
    //   (6,5)-(7,5)-(8,5) walled off above/below/at the far end, capped on
    //   its near side by the player's own (already-occupied) head -- so its
    //   floodFillArea is tiny (2 cells) versus hundreds of cells open in
    //   every other direction.
    {
        GameState s;
        s.reset(1); // no opponents -- isolates the wall-shape effect from Voronoi-competition effects (Test 2 covers those)
        s.headCell[0] = s.spawnCell[0] = GameState::idx(5, 5);
        s.cellOwner[s.headCell[0]] = 0;
        s.alive[0] = true;

        for (int x : {6, 7, 8}) {
            s.cellOwner[GameState::idx(x, 4)] = 2; // wall above the corridor
            s.cellOwner[GameState::idx(x, 6)] = 2; // wall below the corridor
        }
        s.cellOwner[GameState::idx(9, 5)] = 2; // wall capping the far end

        int rightArea = floodFillArea(s, GameState::idx(6, 5), {});
        int upArea = floodFillArea(s, GameState::idx(5, 4), {});
        assert(rightArea == 2); // just (7,5) and (8,5)
        assert(upArea == 588);  // the rest of the open board: 592 free - 3 corridor - 1 (itself)

        Direction chosen = heuristicAction(s, 0);
        assert(chosen == UP); // ties among UP/DOWN/LEFT on the open board, UP checked first
        cout << "Test 3 OK: heuristic avoids the " << rightArea
             << "-cell dead end for the " << upArea << "-cell open board (picked UP).\n";
    }

    cout << "\nAll heuristic tests passed.\n";
    return 0;
}
