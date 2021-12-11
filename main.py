import numpy as np
from src._solver_cases import one_step_godunov
from src.solver import one_time_step
from src.solver import NumericMethods


def main():
    # godunov
    f = np.concatenate([
        np.ones(10),
        np.zeros(9)
    ])
    for _ in range(10):
        f = one_time_step(f, 0.1, method=NumericMethods.Godunov)
        print(f)


if __name__ == '__main__':
    main()
