#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Engine.hpp"
#include "Heuristic.hpp"
#include "Voronoi.hpp"

namespace py = pybind11;
using namespace sim;

PYBIND11_MODULE(tronsim, m)
{
    m.doc() = "C++ simulator for the line-racing (Tron/lumicycle) game";

    m.attr("WIDTH") = WIDTH;
    m.attr("HEIGHT") = HEIGHT;
    m.attr("MAX_PLAYERS") = MAX_PLAYERS;
    m.attr("NONE") = NONE;

    py::enum_<Direction>(m, "Direction")
        .value("UP", UP)
        .value("DOWN", DOWN)
        .value("LEFT", LEFT)
        .value("RIGHT", RIGHT)
        .export_values();

    py::class_<std::mt19937>(m, "Rng")
        .def(py::init<unsigned int>(), py::arg("seed"));

    py::class_<GameState>(m, "GameState")
        .def(py::init<>())
        .def("reset", &GameState::reset, py::arg("num_players"),
             "Zero out the board and mark all cells NONE; always call this before "
             "hand-building a state (the default constructor leaves cell_owner as 0, "
             "not NONE -- see Engine.hpp).")
        .def_static("idx", &GameState::idx, py::arg("x"), py::arg("y"))
        .def_static("in_bounds", &GameState::inBounds, py::arg("x"), py::arg("y"))
        .def_readonly("num_players", &GameState::numPlayers)
        .def_readonly("cell_owner", &GameState::cellOwner)
        .def_readonly("head_cell", &GameState::headCell)
        .def_readonly("spawn_cell", &GameState::spawnCell)
        .def_readonly("alive", &GameState::alive)
        .def_readonly("next_to_move", &GameState::nextToMove)
        .def_readonly("move_count", &GameState::moveCount);

    m.def("random_spawn", &randomSpawn, py::arg("num_players"), py::arg("rng"),
          "Sample a fresh GameState with num_players distinct uniform-random spawn cells.");
    m.def("step_player", &stepPlayer, py::arg("state"), py::arg("player_id"), py::arg("direction"),
          "Advance state.next_to_move's sub-turn in place; returns True if that move eliminated the player.");
    m.def("legal_moves", &legalMoves, py::arg("state"), py::arg("player_id"),
          "Per-direction legality mask [UP, DOWN, LEFT, RIGHT] for player_id's next move.");
    m.def("is_terminal", &isTerminal, py::arg("state"));
    m.def("winner", &winner, py::arg("state"),
          "The sole surviving player id, or NONE if the (should-never-fire) MAX_MOVES cutoff hit.");
    m.def("alive_count", &aliveCount, py::arg("state"));

    m.def("flood_fill_area", &floodFillArea, py::arg("state"), py::arg("start"), py::arg("opponents"),
          "Free cells `start` reaches no later than any of `opponents` (BFS ties favor `start`).");
    m.def("heuristic_action", &heuristicAction, py::arg("state"), py::arg("player_id"),
          "The flood-fill self-play opponent: the legal move maximizing flood_fill_area.");

    m.def("voronoi_ownership", &voronoiOwnership, py::arg("state"),
          "Per-cell owner (player id or NONE) from a multi-source BFS race between all alive heads.");
    m.def("voronoi_counts", &voronoiCounts, py::arg("state"),
          "Per-player cell counts from voronoi_ownership, for reward/feature use.");
}
