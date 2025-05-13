import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import warnings

GRID_SIZE = config.GRID_SIZE


def my_round(x: float) -> int:
    y = round(x / GRID_SIZE)
    if abs(x - y * GRID_SIZE) > GRID_SIZE / 10:
        warnings.warn(f"Possible error in rounding: {x} -> {y}")
    return y
