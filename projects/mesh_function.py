import numpy as np
from collections.abc import Callable
import math

def mesh_function(f: Callable[[float], float], t: np.ndarray) -> np.ndarray:
    y=np.array([f(time) for time in t])
    return y



def func(t: float) -> float:
    return np.exp(-t)


def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-4)])
    # print f but with 4 decimal places
    print(np.round(f, 4))
    fun = mesh_function(func, t)
    print(fun)
    assert np.allclose(fun, f)

if __name__ == "__main__":
    test_mesh_function()
