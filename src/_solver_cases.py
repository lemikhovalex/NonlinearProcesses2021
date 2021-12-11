import numpy as np


def get_f_minus_one(f: np.ndarray) -> np.ndarray:
    f_no_board = f[1: -1]
    f_minus_one = f_no_board[1:]
    f_minus_one = np.concatenate([np.array([1]),
                                  f_minus_one]
                                 )
    return f_minus_one


def one_step_godunov(f: np.ndarray, c: float) -> np.ndarray:
    f_no_board = f[1:]
    out_no_board = f_no_board - c * (f_no_board - f[:-1])
    return np.concatenate([np.array([1]),
                           out_no_board,
                           np.array([0])
                           ]
                          )
