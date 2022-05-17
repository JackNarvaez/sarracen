import numpy as np
from numba import jit

from sarracen.kernels import BaseKernel


class QuarticSplineKernel(BaseKernel):
    """
    An implementation of the Quartic Spline kernel, in 1, 2, and 3 dimensions.
    """

    @staticmethod
    def get_radius() -> float:
        return 2.5

    @staticmethod
    @jit(fastmath=True)
    def w(q: float, ndim: int):
        norm = 1 / 24 if (ndim == 1) else \
            96 / (1199 * np.pi) if (ndim == 2) else \
            1 / (20 * np.pi)

        return norm * (((5 / 2) - q) ** 4 * (q < 2.5) - 5 * ((3 / 2) - q) ** 4 * (q < 1.5) + 10 * ((1 / 2) - q) ** 4 * (
                    q < 0.5))
