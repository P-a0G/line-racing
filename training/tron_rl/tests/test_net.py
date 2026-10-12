import torch

from tron_rl.net import DuelingIQNNet, NUM_ACTIONS
from tron_rl.obs import NUM_CHANNELS, HEIGHT, WIDTH


def make_net():
    torch.manual_seed(0)
    return DuelingIQNNet(channels=8, blocks=2, gpb_blocks=(0,))


def test_forward_shapes():
    net = make_net()
    B, K = 4, 8
    grid = torch.randn(B, NUM_CHANNELS, HEIGHT, WIDTH)
    legal = torch.ones(B, NUM_ACTIONS, dtype=torch.bool)
    taus = torch.rand(B, K)
    out = net(grid, legal, taus)
    assert out["v"].shape == (B, K)
    assert out["q"].shape == (B, K, NUM_ACTIONS)
    assert torch.isfinite(out["q"]).all()


def test_illegal_actions_are_driven_far_below_legal_ones():
    net = make_net()
    B, K = 2, 4
    grid = torch.randn(B, NUM_CHANNELS, HEIGHT, WIDTH)
    legal = torch.zeros(B, NUM_ACTIONS, dtype=torch.bool)
    legal[:, 0] = True  # only UP is legal
    taus = torch.rand(B, K)
    out = net(grid, legal, taus)
    assert (out["q"][:, :, 0] > out["q"][:, :, 1:].amax(dim=-1) + 100).all()


def test_advantage_is_mean_zero_over_the_legal_set():
    net = make_net()
    B, K = 3, 4
    grid = torch.randn(B, NUM_CHANNELS, HEIGHT, WIDTH)
    legal = torch.tensor([[True, True, True, True]] * B)
    taus = torch.rand(B, K)
    out = net(grid, legal, taus)
    a = out["q"] - out["v"][:, :, None]
    assert torch.allclose(a.mean(dim=-1), torch.zeros(B, K), atol=1e-4)


def test_fully_boxed_in_does_not_crash_and_argmax_picks_up():
    net = make_net()
    grid = torch.randn(1, NUM_CHANNELS, HEIGHT, WIDTH)
    legal = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool)  # no legal move at all
    taus = torch.rand(1, 4)
    out = net(grid, legal, taus)
    assert torch.isfinite(out["q"]).all()
    assert out["q"][0, 0].argmax().item() == 0  # ties broken to index 0 == UP


def test_eval_mode_is_deterministic_noisenet_collapses_to_mu():
    net = make_net()
    net.eval()
    grid = torch.randn(1, NUM_CHANNELS, HEIGHT, WIDTH)
    legal = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
    taus = torch.full((1, 4), 0.5)
    out1 = net(grid, legal, taus)
    net.reset_noise()  # must have no effect at eval()
    out2 = net(grid, legal, taus)
    assert torch.equal(out1["q"], out2["q"])


def test_train_mode_noise_changes_output():
    net = make_net()
    net.train()
    grid = torch.randn(1, NUM_CHANNELS, HEIGHT, WIDTH)
    legal = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
    taus = torch.full((1, 4), 0.5)
    out1 = net(grid, legal, taus)
    net.reset_noise()
    out2 = net(grid, legal, taus)
    assert not torch.equal(out1["q"], out2["q"])
