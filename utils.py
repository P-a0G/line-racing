import json
import time
from datetime import timedelta
import numpy as np


def load_bar(i, n, start_time=None):
    end = f"{i + 1}/{n}"
    j = (i + 1) * 100 // n
    bar = f"[{'■' * j}{' ' * (100 - j)}]\t {end}"
    if start_time is not None and i > 0:
        now = time.time()
        tot_time = (n / i) * (now - start_time) + start_time
        remaining_time = int(tot_time - now)
        bar += f" [{timedelta(seconds=int(now - start_time))} < {timedelta(seconds=remaining_time)}]"
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
