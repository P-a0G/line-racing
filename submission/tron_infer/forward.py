"""Pure numpy mirror of training/tron_rl/net.py's DuelingIQNNet forward
pass, for the actual CG submission (no torch in the judge's Python).
Consumes the weights export.py produces: every spectral-norm conv's
EFFECTIVE weight (`.weight` on a `torch.nn.utils.parametrizations.
spectral_norm`-wrapped module already returns the normalized weight --
verified empirically, same as summer_challenge_2026's forward.py), each
ReZero block's `alpha` folded directly into that block's conv2 weight (a
scalar multiply, since alpha only scales conv2's output before the residual
add and conv2 has no bias), and every NoisyLinear's `weight_mu`/`bias_mu`
only -- sigma/noise exist for training exploration alone and never ship.

Runs at the board's one true size (30x20, always -- no padding/masking
needed at all, unlike the rail game's size-agnostic trunk), so every array
here is shaped (C, 20, 30).
"""
import numpy as np


def relu(x):
    return np.maximum(x, 0.0)


def conv3x3(x, weight):
    """x: (Cin, H, W); weight: (Cout, Cin, 3, 3), same-padding. Implemented
    as 9 shifted (Cout,Cin)@(Cin,HW) matmuls rather than an im2col
    materialization -- copied from summer_challenge_2026's forward.py,
    fully generic."""
    Cin, H, W = x.shape
    Cout = weight.shape[0]
    out = np.zeros((Cout, H, W), dtype=x.dtype)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            wslice = weight[:, :, dy + 1, dx + 1]  # (Cout, Cin)
            if wslice.size == 0 or not np.any(wslice):
                continue
            ys, ye = max(0, -dy), H - max(0, dy)
            xs, xe = max(0, -dx), W - max(0, dx)
            if ys >= ye or xs >= xe:
                continue
            src = x[:, ys + dy:ye + dy, xs + dx:xe + dx]  # (Cin, h, w)
            h, w = src.shape[1], src.shape[2]
            contrib = wslice @ src.reshape(Cin, h * w)
            out[:, ys:ye, xs:xe] += contrib.reshape(Cout, h, w)
    return out


def conv1x1(x, weight, bias=None):
    """x: (Cin,H,W); weight: (Cout,Cin,1,1) or (Cout,Cin); bias: (Cout,) or None."""
    Cin, H, W = x.shape
    w2d = weight.reshape(weight.shape[0], Cin)
    out = (w2d @ x.reshape(Cin, H * W)).reshape(-1, H, W)
    if bias is not None:
        out = out + bias[:, None, None]
    return out


def linear(x, weight, bias):
    """x: (..., in); weight: (out,in); bias: (out,)."""
    return x @ weight.T + bias


def global_pool2(x):
    """x: (C,H,W). Returns (2C,): [mean, max] -- every cell is always real
    on this fixed 30x20 board, so unlike the rail game's global_pool3 there
    is no size-scaled-mean term or masking to reproduce."""
    C, H, W = x.shape
    flat = x.reshape(C, H * W)
    return np.concatenate([flat.mean(axis=1), flat.max(axis=1)])


def rezero_block(x, block):
    y = relu(conv3x3(x, block["conv1"]))
    if "gpb" in block:
        pooled = global_pool2(y)
        bias = linear(pooled, block["gpb"]["weight"], block["gpb"]["bias"])
        y = y + bias[:, None, None]
    y = conv3x3(y, block["conv2"])
    return x + y


def trunk_forward(grid, weights):
    x = conv1x1(grid, weights["stem"]["weight"])
    for block in weights["blocks"]:
        x = rezero_block(x, block)
    return x


def cosine_embedding(taus, num_cosines):
    """taus: (K,). Returns (K, num_cosines)."""
    i = np.arange(1, num_cosines + 1, dtype=np.float64)
    angles = np.pi * i[None, :] * taus[:, None]
    return np.cos(angles)


def iqn_embedding(taus, weights):
    cos = cosine_embedding(taus, weights["weight"].shape[1])
    return relu(linear(cos, weights["weight"], weights["bias"]))  # (K, C)


def forward(grid, legal_mask, taus, weights):
    """grid: (NUM_CHANNELS,20,30); legal_mask: (4,) bool; taus: (K,) float
    in [0,1]. Returns dict with 'v' (K,) and 'q' (K,4), dueling-combined and
    mean-normalized over the legal actions exactly like net.py's torch
    version (illegal actions driven to -1e9; if none are legal, all four tie
    and a downstream argmax picks index 0 == UP, matching
    legacy/new_IA.py's own boxed-in fallback)."""
    K = len(taus)
    trunk_out = trunk_forward(grid, weights["trunk"])  # (C,H,W)
    phi = iqn_embedding(taus, weights["iqn"])            # (K,C)
    psi = trunk_out[None] * phi[:, :, None, None]         # (K,C,H,W)

    v = np.empty(K, dtype=np.float64)
    a_raw = np.empty((K, 4), dtype=np.float64)
    for k in range(K):
        pool = global_pool2(psi[k])
        v[k] = linear(relu(linear(pool, weights["value_fc1"]["weight"], weights["value_fc1"]["bias"])),
                        weights["value_fc2"]["weight"], weights["value_fc2"]["bias"])[0]
        a_raw[k] = linear(relu(linear(pool, weights["adv_fc1"]["weight"], weights["adv_fc1"]["bias"])),
                            weights["adv_fc2"]["weight"], weights["adv_fc2"]["bias"])

    legal_count = max(int(legal_mask.sum()), 1)
    sum_a = np.where(legal_mask[None], a_raw, 0.0).sum(axis=1)
    mean_a = sum_a / legal_count
    a = np.where(legal_mask[None], a_raw - mean_a[:, None], -1e9)

    return {"v": v, "q": v[:, None] + a}
