#include "Engine.hpp"
#include <iostream>
#include <cassert>

using namespace sim;
using namespace std;

int main()
{
    // Test 1: a legal move extends the trail and hands the turn to the next player.
    {
        GameState s;
        s.reset(2);
        s.headCell[0] = s.spawnCell[0] = GameState::idx(5, 5);
        s.cellOwner[s.headCell[0]] = 0;
        s.alive[0] = true;
        s.headCell[1] = s.spawnCell[1] = GameState::idx(10, 10);
        s.cellOwner[s.headCell[1]] = 1;
        s.alive[1] = true;

        bool died = stepPlayer(s, 0, RIGHT);
        assert(!died);
        assert(s.headCell[0] == GameState::idx(6, 5));
        assert(s.cellOwner[GameState::idx(6, 5)] == 0);
        assert(s.cellOwner[GameState::idx(5, 5)] == 0); // old head stays part of the trail
        assert(s.nextToMove == 1);
        assert(!isTerminal(s));
        cout << "Test 1 OK: legal move extends trail, turn passes to next player.\n";
    }

    // Test 2: moving off the grid eliminates the player and clears their trail.
    {
        GameState s;
        s.reset(2);
        s.headCell[0] = s.spawnCell[0] = GameState::idx(0, 0);
        s.cellOwner[s.headCell[0]] = 0;
        s.alive[0] = true;
        s.headCell[1] = s.spawnCell[1] = GameState::idx(10, 10);
        s.cellOwner[s.headCell[1]] = 1;
        s.alive[1] = true;

        bool died = stepPlayer(s, 0, UP); // y=-1, off grid
        assert(died);
        assert(!s.alive[0]);
        assert(s.cellOwner[GameState::idx(0, 0)] == NONE); // trail vanished
        assert(s.nextToMove == 1); // skipped straight to the only alive player
        assert(isTerminal(s));
        assert(winner(s) == 1);
        cout << "Test 2 OK: off-grid move eliminates the player, trail clears, game ends.\n";
    }

    // Test 3: reversing into your own trail is just a generic collision, no special-casing needed.
    {
        GameState s;
        s.reset(2);
        s.headCell[0] = GameState::idx(5, 5);
        s.spawnCell[0] = GameState::idx(4, 5);
        s.cellOwner[GameState::idx(4, 5)] = 0;
        s.cellOwner[GameState::idx(5, 5)] = 0;
        s.alive[0] = true;
        s.headCell[1] = s.spawnCell[1] = GameState::idx(20, 15);
        s.cellOwner[s.headCell[1]] = 1;
        s.alive[1] = true;

        assert(legalMoves(s, 0)[LEFT] == false); // own trail blocks it before we even try
        bool died = stepPlayer(s, 0, LEFT);
        assert(died);
        cout << "Test 3 OK: moving back into your own trail eliminates you like any other collision.\n";
    }

    // Test 4: a mid-round elimination's trail disappears immediately, visible to the very
    // next sub-turn in the same round (rules.md: "son ruban de lumiere disparait et le jeu continue").
    {
        GameState s;
        s.reset(3);
        s.spawnCell[0] = GameState::idx(28, 3);
        s.headCell[0] = GameState::idx(29, 3);
        s.cellOwner[GameState::idx(28, 3)] = 0;
        s.cellOwner[GameState::idx(29, 3)] = 0;
        s.alive[0] = true;
        s.headCell[1] = s.spawnCell[1] = GameState::idx(28, 4);
        s.cellOwner[s.headCell[1]] = 1;
        s.alive[1] = true;
        s.headCell[2] = s.spawnCell[2] = GameState::idx(0, 0);
        s.cellOwner[s.headCell[2]] = 2;
        s.alive[2] = true;

        assert(legalMoves(s, 1)[UP] == false); // (28,3) still belongs to player 0 before their move

        bool died0 = stepPlayer(s, 0, RIGHT); // x=30, off grid -> player 0 eliminated
        assert(died0);
        assert(s.cellOwner[GameState::idx(28, 3)] == NONE);
        assert(s.nextToMove == 1);

        assert(legalMoves(s, 1)[UP] == true); // now free, visible within the same round
        bool died1 = stepPlayer(s, 1, UP);
        assert(!died1);
        assert(s.headCell[1] == GameState::idx(28, 3));
        cout << "Test 4 OK: an eliminated player's trail is immediately visible to later turns.\n";
    }

    // Test 5: round-robin skips dead players, and a 3-player game can only ever end at
    // exactly 1 survivor (never 0), since eliminations happen one sub-turn at a time.
    {
        GameState s;
        s.reset(3);
        s.headCell[0] = s.spawnCell[0] = GameState::idx(0, 0);
        s.cellOwner[s.headCell[0]] = 0;
        s.alive[0] = true;
        s.headCell[1] = s.spawnCell[1] = GameState::idx(29, 0); // top-right corner, dies moving UP
        s.cellOwner[s.headCell[1]] = 1;
        s.alive[1] = true;
        s.headCell[2] = s.spawnCell[2] = GameState::idx(29, 19); // bottom-right corner, dies moving RIGHT
        s.cellOwner[s.headCell[2]] = 2;
        s.alive[2] = true;

        bool died0 = stepPlayer(s, 0, RIGHT); // player 0 survives, moves normally
        assert(!died0);
        assert(s.nextToMove == 1);

        bool died1 = stepPlayer(s, 1, UP); // off grid -> player 1 eliminated
        assert(died1);
        assert(aliveCount(s) == 2);
        assert(s.nextToMove == 2); // skipped the dead player 1, landed on 2
        assert(!isTerminal(s));

        bool died2 = stepPlayer(s, 2, RIGHT); // off grid -> player 2 eliminated
        assert(died2);
        assert(aliveCount(s) == 1); // lands exactly on 1, never 0
        assert(isTerminal(s));
        assert(winner(s) == 0);
        cout << "Test 5 OK: round-robin skips dead players; alive count lands on exactly 1.\n";
    }

    cout << "\nAll engine tests passed.\n";
    return 0;
}
