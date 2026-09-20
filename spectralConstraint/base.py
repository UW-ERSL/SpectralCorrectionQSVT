"""ChebIterPolynomial, copied verbatim from PolynomialApproximators.py."""
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev


class ChebIterPolynomial:
    @staticmethod
    def _z0(a: float) -> float:
        return (1.0 + a * a) / (1.0 - a * a)

    @staticmethod
    def poly(d: int, a: float) -> Chebyshev:
        if d % 2 == 0:
            raise ValueError("d must be odd")
        n = (d + 1) // 2
        Tn_z0 = np.cosh(n * np.arccosh(ChebIterPolynomial._z0(a)))

        def f(x):
            u = (2.0 * x * x - (1.0 + a * a)) / (1.0 - a * a)
            inside = np.abs(u) <= 1.0
            Tn = np.where(inside,
                          np.cos(n * np.arccos(np.clip(u, -1.0, 1.0))),
                          np.cosh(n * np.arccosh(np.maximum(np.abs(u), 1.0)))
                          * np.sign(u) ** n)
            return (1.0 - ((-1) ** n) * Tn / Tn_z0) / np.where(x == 0, 1e-300, x)

        coef = np.polynomial.chebyshev.chebinterpolate(f, d)
        coef[0::2] = 0.0
        return Chebyshev(coef)

    @staticmethod
    def error_for_degree(d: int, a: float) -> float:
        n = (d + 1) // 2
        return 1.0 / np.cosh(n * np.arccosh(ChebIterPolynomial._z0(a)))

    @staticmethod
    def mindegree(epsilon: float, a: float) -> int:
        n = int(np.ceil(np.arccosh(1.0 / epsilon)
                        / np.arccosh(ChebIterPolynomial._z0(a))))
        return 2 * n - 1
