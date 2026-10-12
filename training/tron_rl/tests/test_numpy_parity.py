import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "tron_infer"))
from forward import forward as numpy_forward  # noqa: E402

from tron_rl.net import DuelingIQNNet
from tron_rl.obs import NUM_CHANNELS, HEIGHT, WIDTH
from tron_rl.export import extract_weights


def make_eval_net(seed=0, gpb_blocks=(0,)):
    torch.manual_seed(seed)
    net = DuelingIQNNet(channels=8, blocks=2, gpb_blocks=gpb_blocks)
    net.eval()  # NoisyNet collapses to weight_mu/bias_mu, matching extract_weights()
    return net


def test_forward_matches_torch_with_all_legal():
    net = make_eval_net()
    weights = extract_weights(net)
    grid = np.random.default_rng(0).standard_normal((NUM_CHANNELS, HEIGHT, WIDTH)).astype(np.float32)
    legal = np.array([True, True, True, True])
    taus = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    out_np = numpy_forward(grid, legal, taus, weights)

    with torch.no_grad():
        out_t = net(torch.from_numpy(grid).unsqueeze(0).float(),
                     torch.from_numpy(legal).unsqueeze(0),
                     torch.from_numpy(taus).unsqueeze(0).float())

    np.testing.assert_allclose(out_np["v"], out_t["v"][0].numpy(), atol=1e-3)
    np.testing.assert_allclose(out_np["q"], out_t["q"][0].numpy(), atol=1e-3)


def test_forward_matches_torch_with_some_illegal():
    net = make_eval_net(seed=1)
    weights = extract_weights(net)
    grid = np.random.default_rng(1).standard_normal((NUM_CHANNELS, HEIGHT, WIDTH)).astype(np.float32)
    legal = np.array([True, False, True, False])
    taus = np.array([0.5])

    out_np = numpy_forward(grid, legal, taus, weights)
    with torch.no_grad():
        out_t = net(torch.from_numpy(grid).unsqueeze(0).float(),
                     torch.from_numpy(legal).unsqueeze(0),
                     torch.from_numpy(taus).unsqueeze(0).float())

    np.testing.assert_allclose(out_np["q"][0, 1], -1e9)
    np.testing.assert_allclose(out_np["q"][0, 3], -1e9)
    np.testing.assert_allclose(out_np["q"], out_t["q"][0].numpy(), atol=1e-3)


def test_forward_matches_torch_fully_boxed_in():
    net = make_eval_net(seed=2)
    weights = extract_weights(net)
    grid = np.random.default_rng(2).standard_normal((NUM_CHANNELS, HEIGHT, WIDTH)).astype(np.float32)
    legal = np.array([False, False, False, False])
    taus = np.array([0.25, 0.75])

    out_np = numpy_forward(grid, legal, taus, weights)
    with torch.no_grad():
        out_t = net(torch.from_numpy(grid).unsqueeze(0).float(),
                     torch.from_numpy(legal).unsqueeze(0),
                     torch.from_numpy(taus).unsqueeze(0).float())

    np.testing.assert_allclose(out_np["q"], out_t["q"][0].numpy(), atol=1e-3)
    assert out_np["q"][0].argmax() == 0  # ties resolve to UP, both sides


def test_forward_without_gpb_blocks():
    net = make_eval_net(seed=3, gpb_blocks=())
    weights = extract_weights(net)
    grid = np.random.default_rng(3).standard_normal((NUM_CHANNELS, HEIGHT, WIDTH)).astype(np.float32)
    legal = np.array([True, True, True, True])
    taus = np.array([0.4])

    out_np = numpy_forward(grid, legal, taus, weights)
    with torch.no_grad():
        out_t = net(torch.from_numpy(grid).unsqueeze(0).float(),
                     torch.from_numpy(legal).unsqueeze(0),
                     torch.from_numpy(taus).unsqueeze(0).float())

    assert all("gpb" not in b for b in weights["trunk"]["blocks"])
    np.testing.assert_allclose(out_np["q"], out_t["q"][0].numpy(), atol=1e-3)
