#include "Voronoi.hpp"
#include <iostream>
#include <cassert>

using namespace sim;
using namespace std;

int main()
{
    // Test 1: symmetric split, same setup/expectation as Heuristic's tie-break
    // test (299/299) -- ties go to the lower player index here instead of to
    // an explicit "start", since both heads are seeded at depth 0 in index order.
    {
        GameState s;
        s.reset(2);
        s.headCell[0] = s.spawnCell[0] = GameState::idx(0, 10);
        s.headCell[1] = s.spawnCell[1] = GameState::idx(29, 10);
        s.cellOwner[s.headCell[0]] = 0;
        s.cellOwner[s.headCell[1]] = 1;
        s.alive[0] = s.alive[1] = true;

        auto owner = voronoiOwnership(s);
        auto counts = voronoiCounts(s);
        assert(counts[0] == 15 * HEIGHT - 1);
        assert(counts[1] == 15 * HEIGHT - 1);
        assert(owner[GameState::idx(14, 10)] == 0); // last column resolving to player 0
        assert(owner[GameState::idx(15, 10)] == 1); // first column resolving to player 1
        cout << "Test 1 OK: symmetric split " << counts[0] << "/" << counts[1] << ".\n";
    }

    // Test 2: a 3-player partition must cover every free cell exactly once.
    {
        GameState s;
        s.reset(3);
        s.headCell[0] = s.spawnCell[0] = GameState::idx(0, 0);
        s.headCell[1] = s.spawnCell[1] = GameState::idx(29, 0);
        s.headCell[2] = s.spawnCell[2] = GameState::idx(15, 19);
        for (int p = 0; p < 3; p++) {
            s.cellOwner[s.headCell[p]] = static_cast<int8_t>(p);
            s.alive[p] = true;
        }

        auto counts = voronoiCounts(s);
        int total = counts[0] + counts[1] + counts[2];
        assert(total == WIDTH * HEIGHT - 3); // every free cell assigned to exactly one player
        assert(counts[0] > 0 && counts[1] > 0 && counts[2] > 0);
        cout << "Test 2 OK: 3-way partition covers all " << total << " free cells ("
             << counts[0] << "/" << counts[1] << "/" << counts[2] << ").\n";
    }

    // Test 3: a sealed pocket with no player inside stays unassigned even
    // though its cells are nominally free.
    {
        GameState s;
        s.reset(2);
        s.headCell[0] = s.spawnCell[0] = GameState::idx(0, 0);
        s.headCell[1] = s.spawnCell[1] = GameState::idx(29, 19);
        s.cellOwner[s.headCell[0]] = 0;
        s.cellOwner[s.headCell[1]] = 1;
        s.alive[0] = s.alive[1] = true;

        int pocket[4] = {GameState::idx(20, 10), GameState::idx(21, 10),
                          GameState::idx(20, 11), GameState::idx(21, 11)};
        int wall[8] = {GameState::idx(19, 10), GameState::idx(19, 11),
                        GameState::idx(22, 10), GameState::idx(22, 11),
                        GameState::idx(20, 9), GameState::idx(21, 9),
                        GameState::idx(20, 12), GameState::idx(21, 12)};
        for (int w : wall) s.cellOwner[w] = 2;

        auto owner = voronoiOwnership(s);
        for (int cell : pocket) assert(owner[cell] == NONE);
        cout << "Test 3 OK: sealed pocket with no player inside stays unassigned.\n";
    }

    cout << "\nAll voronoi tests passed.\n";
    return 0;
}
