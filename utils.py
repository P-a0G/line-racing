import json

import numpy as np


def load_bar(i, n):
    end = f"{i + 1}/{n}"
    i = (i + 1) * 100 // n
    bar = f"[{'■' * i}{' ' * (100 - i)}]\t {end}"
    return bar


def read_json(path):
    with open(path, 'r', encoding='utf-8-sig') as fi:
        data = json.load(fi)

    return data


def write_json(path, data):
    with open(path, 'w', encoding='utf-8-sig') as fo:
        json.dump(data, fo, ensure_ascii=False, indent=4, default=convert)


def convert(o):
    if isinstance(o, str):
        return o
    elif isinstance(o, int):
        return o
    elif isinstance(o, np.int32) or isinstance(o, np.int64):
        return int(o)
    print(type(o))
    raise TypeError
