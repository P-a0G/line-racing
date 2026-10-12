"""Weight (de)serialization shared between training/tron_rl/export.py and
the actual CG submission's generated weights_blob.py.

Unlike summer_challenge_2026's codec.py/weights_codec.py, there is no
pluggable base91/bmp15 codec here: this network is far smaller (a handful of
24-48-channel convs and small head linears over a fixed 30x20 board, versus
the rail game's 48-channel/60-input-channel/factored-dueling network), so
even a plain lzma+base64 blob lands at a few KB -- nowhere near CodinGame's
~100k-character cap. Porting the extra bytes/char a custom codec buys (and
the unresolved UTF-16-vs-bytes cap question that would come with it -- see
summer_challenge_2026's probe_size.py) would be solving a problem this
network doesn't have.

Nested weight dicts (as forward.py's `weights` argument expects, and
export.py's extract_weights() produces) are flattened to {dotted.name:
array} for quantization/packing, then rebuilt on load -- blocks (a list) are
addressed by a positional index segment (e.g. "trunk.blocks0.conv1").
"""
import base64
import lzma
import pickle

import numpy as np


def _flatten(weights, prefix=""):
    for key, value in weights.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _flatten(value, name)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                yield from _flatten(item, f"{name}{i}")
        else:
            yield name, value


def _unflatten(flat):
    root = {}
    for name, arr in flat.items():
        parts = name.split(".")
        if parts[0] == "trunk" and parts[1].startswith("blocks"):
            idx = int(parts[1][len("blocks"):])
            root.setdefault("trunk", {}).setdefault("blocks", {})
            root["trunk"]["blocks"].setdefault(idx, {})
            cur = root["trunk"]["blocks"][idx]
            for p in parts[2:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = arr
        else:
            cur = root
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = arr

    if "trunk" in root and "blocks" in root["trunk"]:
        blocks_dict = root["trunk"]["blocks"]
        root["trunk"]["blocks"] = [blocks_dict[i] for i in sorted(blocks_dict)]
    return root


def quantize_weights(weights):
    """Per-tensor int8 quantization (scale = max(abs(w))/127). Returns
    (quantized_flat: {name: int8 array}, scales: {name: float}, shapes:
    {name: tuple})."""
    quantized, scales, shapes = {}, {}, {}
    for name, arr in _flatten(weights):
        scale = (float(np.max(np.abs(arr))) / 127.0) or 1.0  # an all-zero tensor would else divide by zero
        quantized[name] = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
        scales[name] = scale
        shapes[name] = arr.shape
    return quantized, scales, shapes


def dequantize_weights(quantized, scales, shapes):
    flat = {name: quantized[name].astype(np.float32).reshape(shapes[name]) * scales[name] for name in quantized}
    return _unflatten(flat)


def pack(weights, codec_name="base64"):
    assert codec_name == "base64", "only base64 is supported here -- see this module's docstring"
    quantized, scales, shapes = quantize_weights(weights)
    blob = lzma.compress(pickle.dumps({"quantized": quantized, "scales": scales, "shapes": shapes}),
                           preset=9 | lzma.PRESET_EXTREME)
    return base64.b64encode(blob).decode("ascii"), len(blob)


def unpack(literal, codec_name="base64"):
    assert codec_name == "base64", "only base64 is supported here -- see this module's docstring"
    payload = pickle.loads(lzma.decompress(base64.b64decode(literal.encode("ascii"))))
    return dequantize_weights(payload["quantized"], payload["scales"], payload["shapes"])
