import numpy as np


def f_plus_one(f: np.ndarray) -> np.ndarray:
    return np.concatenate([f[1:], np.array([0])])


def f_minus_one(f: np.ndarray) -> np.ndarray:
    return np.concatenate([np.array([1]), f[:-1]])


def f_delta(f: np.ndarray) -> np.ndarray:
    out = f - f_minus_one(f)
    return out


def f_delta_square(f: np.ndarray) -> np.ndarray:
    out = f_delta(f) - f_delta(f_minus_one(f))
    return out


def one_step_godunov(f: np.ndarray, c: float) -> np.ndarray:
    _f_minus_one = f_minus_one(f)
    out = f - c * (f - _f_minus_one)
    return out


def one_step_mc_cormac(f: np.ndarray, c: float) -> np.ndarray:
    _f_minus_one = f_minus_one(f)
    _f_plus_one = f_plus_one(f)
    out = f - 0.5 * c * (_f_plus_one - _f_minus_one) + 0.5 * (c*c) * (_f_plus_one - 2 * f + _f_minus_one)
    return out


def one_step_holodov(f: np.ndarray, c: float) -> np.ndarray:
    _f_minus_one = f_minus_one(f)
    _f_minus_two = f_minus_one(_f_minus_one)
    _f_plus_one = f_plus_one(f)
    out = f - c * (f - _f_minus_one) - 0.25 * c * (1 - c) * (_f_plus_one - f - _f_minus_one + _f_minus_two)
    return out


def one_step_forth(f: np.ndarray, c: float) -> np.ndarray:
    _f_minus_one = f_minus_one(f)
    _f_minus_two = f_minus_one(_f_minus_one)
    _f_plus_one = f_plus_one(f)
    out = f - c * (f - _f_minus_one)
    out -= c / 6 * (2 - c) * (1 - c) * _f_plus_one
    out += c / 2 * (1 - c ) ** 2 * f
    out += (c ** 2) / 2 * (1 - c) * _f_minus_one
    out += c / 6 * (c ** 2 - 1) * _f_minus_two
    return out


def one_step_fifth_hybrid(f: np.ndarray, c: float) -> np.ndarray:
    switcher = np.array(f_delta(f) * f_delta_square(f) <= 0)
    _f_minus_one = f_minus_one(f)
    _f_minus_two = f_minus_one(_f_minus_one)
    _f_plus_one = f_plus_one(f)

    out_first_case = -0.5 * c * (1 - c) * _f_plus_one
    out_first_case += (1 - c ** 2) * f
    out_first_case += 0.5 * c * (1 + c) * _f_minus_one

    out_second_case = (1 - 0.5 * c * (3 - c)) * f
    out_second_case += c * (2 - c) * _f_minus_one
    out_second_case -= 0.5 * c * (1 - c) * _f_minus_two

    out = np.where(switcher, out_first_case, out_second_case)
    return out

