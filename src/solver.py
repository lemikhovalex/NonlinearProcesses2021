from enum import Enum
from ._solver_cases import one_step_godunov
import numpy as np
from functools import partial


class NumericMethods(Enum):
    Godunov = partial(one_step_godunov)


def one_time_step(f: np.ndarray, c: float, method: NumericMethods) -> np.ndarray:
    return method.value(f, c)

