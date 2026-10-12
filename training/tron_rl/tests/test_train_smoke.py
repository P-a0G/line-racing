import csv

from tron_rl.train import main


def test_train_runs_end_to_end_and_produces_expected_files(tmp_path):
    main(
        max_iterations=3,
        num_slots=4,
        decisions_per_collect=60,
        min_sampling_size=10,
        grad_steps_per_collect=2,
        batch_size=8,
        mix=(0.0, 0.0, 1.0),  # pure heuristic bootstrap -- no snapshot pool needed yet
        player_count_mix=(0.5, 0.5),
        buffer_capacity_transitions=2000,
        eval_every=2,
        checkpoint_every=2,
        snapshot_every=2,
        eval_games=2,
        net_kwargs=dict(channels=8, blocks=2, gpb_blocks=(0,)),
        out_dir=str(tmp_path),
        seed=0,
    )

    assert (tmp_path / "checkpoint.pt").exists()
    assert (tmp_path / "train_log.txt").exists()

    with open(tmp_path / "train_metrics.csv", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["iter", "buffer_size", "loss", "mean_munchausen_bonus",
                        "games_finished", "mean_turns", "sec"]
    assert len(rows) == 4  # header + 3 iterations

    with open(tmp_path / "eval_metrics.csv", newline="") as f:
        eval_rows = list(csv.reader(f))
    assert eval_rows[0] == ["iter", "win_rate_2p", "win_rate_3p", "mean_turns_2p", "mean_turns_3p"]
    assert len(eval_rows) >= 2  # header + at least one eval (eval_every=2, 3 iterations)


def test_train_resume_continues_from_checkpoint(tmp_path):
    main(max_iterations=2, num_slots=2, decisions_per_collect=40, min_sampling_size=10_000,
          grad_steps_per_collect=1, batch_size=8, mix=(0.0, 0.0, 1.0), buffer_capacity_transitions=2000,
          eval_every=100, checkpoint_every=1, snapshot_every=100, eval_games=2,
          net_kwargs=dict(channels=8, blocks=2, gpb_blocks=(0,)), out_dir=str(tmp_path), seed=1)

    main(max_iterations=2, resume=True, num_slots=2, decisions_per_collect=40, min_sampling_size=10_000,
          grad_steps_per_collect=1, batch_size=8, mix=(0.0, 0.0, 1.0), buffer_capacity_transitions=2000,
          eval_every=100, checkpoint_every=1, snapshot_every=100, eval_games=2,
          net_kwargs=dict(channels=8, blocks=2, gpb_blocks=(0,)), out_dir=str(tmp_path), seed=1)

    with open(tmp_path / "train_metrics.csv", newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 5  # header + 2 (first run) + 2 (resumed run), no duplicate header
    assert rows[1][0] == "0" and rows[2][0] == "1" and rows[3][0] == "2" and rows[4][0] == "3"
