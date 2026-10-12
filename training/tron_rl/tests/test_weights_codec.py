import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "tron_infer"))
from weights_codec import _flatten, _unflatten, quantize_weights, dequantize_weights, pack, unpack  # noqa: E402
from forward import forward  # noqa: E402

from tron_rl.net import DuelingIQNNet
from tron_rl.obs import NUM_CHANNELS, HEIGHT, WIDTH
from tron_rl.export import extract_weights


def test_flatten_unflatten_roundtrip_preserves_structure():
    nested = {
        "trunk": {"stem": {"weight": np.zeros((3, 2, 1, 1))},
                  "blocks": [{"conv1": np.ones((2, 2, 3, 3)), "gpb": {"weight": np.ones((2, 4)), "bias": np.zeros(2)}},
                             {"conv1": np.full((2, 2, 3, 3), 2.0)}]},
        "value_fc1": {"weight": np.ones((5, 4)), "bias": np.zeros(5)},
    }
    flat = dict(_flatten(nested))
    assert "trunk.blocks0.conv1" in flat
    assert "trunk.blocks1.conv1" in flat
    assert "trunk.blocks0.gpb.weight" in flat
    assert "value_fc1.weight" in flat

    rebuilt = _unflatten(flat)
    assert rebuilt["trunk"]["blocks"][0]["conv1"].shape == (2, 2, 3, 3)
    assert rebuilt["trunk"]["blocks"][1]["conv1"].shape == (2, 2, 3, 3)
    np.testing.assert_array_equal(rebuilt["trunk"]["blocks"][1]["conv1"], nested["trunk"]["blocks"][1]["conv1"])
    assert "gpb" not in rebuilt["trunk"]["blocks"][1]  # block 1 never had one -- must not be invented


def test_quantize_dequantize_is_close_within_int8_precision():
    weights = {"a": {"weight": (np.random.default_rng(0).standard_normal((8, 4)) * 3).astype(np.float32)}}
    quantized, scales, shapes = quantize_weights(weights)
    restored = dequantize_weights(quantized, scales, shapes)
    original = weights["a"]["weight"]
    scale = scales["a.weight"]
    np.testing.assert_allclose(restored["a"]["weight"], original, atol=scale)  # within one quantization step


def test_pack_unpack_roundtrip_on_a_real_net():
    torch.manual_seed(0)
    net = DuelingIQNNet(channels=8, blocks=2, gpb_blocks=(0,))
    net.eval()
    weights = extract_weights(net)

    literal, raw_bytes = pack(weights)
    assert isinstance(literal, str)
    assert raw_bytes > 0

    restored = unpack(literal)
    assert set(restored.keys()) == set(weights.keys())
    assert len(restored["trunk"]["blocks"]) == len(weights["trunk"]["blocks"])


def test_roundtripped_weights_still_produce_similar_q_values():
    torch.manual_seed(1)
    net = DuelingIQNNet(channels=8, blocks=2, gpb_blocks=(0,))
    net.eval()
    weights = extract_weights(net)
    literal, _ = pack(weights)
    restored = unpack(literal)

    grid = np.random.default_rng(1).standard_normal((NUM_CHANNELS, HEIGHT, WIDTH)).astype(np.float32)
    legal = np.array([True, True, True, True])
    taus = np.array([0.5])

    out_original = forward(grid, legal, taus, weights)
    out_restored = forward(grid, legal, taus, restored)
    # int8 quantization introduces real error, but the mean-Q ranking (what
    # decode() actually uses) should survive a round trip comfortably at
    # this tolerance for a freshly-initialized (i.e. not adversarially tuned) net.
    np.testing.assert_allclose(out_original["q"], out_restored["q"], atol=0.5)
