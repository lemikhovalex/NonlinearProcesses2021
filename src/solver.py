from enum import Enum
from typing import List, Tuple

from ._solver_cases import one_step_godunov, one_step_mc_cormac, one_step_holodov, one_step_forth, one_step_fifth_hybrid
import numpy as np
from functools import partial


class NumericMethods(Enum):
    Godunov = partial(one_step_godunov)
    McCormac = partial(one_step_mc_cormac)
    Holodov = partial(one_step_holodov)
    Forth = partial(one_step_forth)
    Fifth = partial(one_step_fifth_hybrid)


def process_method(method, f: np.ndarray, c: float, n_ts: int, sh: float = 0.) -> Tuple[List[np.ndarray], float]:
    out = []
    for _ in range(n_ts):
        out.append(f)
        f = method(f=f, c=c)
        sh += c
        if sh >= 1:
            f = np.concatenate([f[int(sh):], np.zeros(int(sh))])
            sh -= int(sh)
    out.append(f)
    return out, sh

