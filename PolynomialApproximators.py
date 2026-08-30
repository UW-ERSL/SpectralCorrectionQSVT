"""
PolynomialApproximators.py
==========================
Polynomial approximations to 1/x for use as QSVT signal polynomials.

All classes produce odd polynomials p(x) ~ 1/x on [a, 1], where a = sigma_min of
the block-encoded matrix.  Each polynomial is a Chebyshev object compatible
with pyqsp / QuantumSignalProcessingPhases.

Classes
-------
ChebIterPolynomial
    Minimax for the RELATIVE criterion  max_{x in [a,1]} |xp(x)-1|, in closed
    form (Gribling et al., optimal Chebyshev iteration).  The exact optimum for
    the criterion this work imposes; used as the reference base.

SunderhaufPolynomial
    Minimax for the ABSOLUTE criterion  max_{x in [a,1]} |p(x)-1/x|.
    Closed-form recurrence.  Its degree exceeds ChebIter's because the absolute
    criterion is stricter by up to kappa, not because it is a weaker construction.

MangPolynomial
    L2-optimal in theta-space (theta = arccos x), with no uniform certificate; the
    residual is not equioscillatory and can peak near x = a.

SpectralPolynomial
    Minimum-norm interpolating polynomial at all N known eigenvalues.
    Degree d = 3N-1, independent of kappa.  No continuous guarantee between
    eigenvalues; must be used with a continuous backbone for QSVT.

References
----------
Gribling et al., arXiv:2109.04248; Mang et al., CCIS 2744 (2026);
Sunderhauf et al., arXiv:2507.15537.
"""

import math
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



# ==============================================================================
# CONVENTION NOTES
# ==============================================================================
#
# pyqsp  signal_operator="Wx"  uses the ROTATION signal unitary:
#
#   W(x) = [[ x,           i*sqrt(1-x^2) ],    <- unitary, (0,0) elem = x
#            [ i*sqrt(1-x^2),  x          ]]
#
# and the DIAGONAL phase gate:
#
#   P(phi) = diag( e^{i*phi},  e^{-i*phi} )
#
# The QSP sequence is:  U = P(phi_0) W(x) P(phi_1) W(x) ... W(x) P(phi_d)
# and for an ODD polynomial p(x):  Re( U[0,0] ) = p(x).
#
# -- Block encoding ---------------------------------------------------------
# To match the Wx rotation signal, the (2N x 2N) block encoding must be:
#
#   U_BE = [[ A,              i*sqrt(I - A A^dag) ],
#            [ i*sqrt(I - A^dag A),     A^dag      ]]
#
# Note the i factors and the +A^dag (not -A^dag) in the bottom-right corner.
# This ensures that, for each singular value sigma_i, the effective 2x2
# block is exactly W(sigma_i).
#
# -- Phase gate in Qiskit ---------------------------------------------------
# Qiskit's  Rz(theta) = diag( e^{-i*theta/2},  e^{+i*theta/2} )
# We need   P(phi)    = diag( e^{+i*phi},       e^{-i*phi}     )
# => use    qc.rz(-2*phi, ancilla)     (Z-rotation, NOT X-rotation)
#
# -- Statevector extraction -------------------------------------------------
# Circuit: QuantumCircuit(q_anc, q_data)
# Qiskit statevector ordering: |data[n-1]...data[0], anc[0]>
#   => index k = data_idx * 2 + anc_bit
# Post-select ancilla=0: sv.data[0::2]  (even indices, data order preserved)
# The solution direction is in the REAL PART of the extracted amplitudes.
#
# ==============================================================================
# Sunderhauf optimal 1/x polynomial
# Ref: Sunderhauf et al., "Block-encoding structured matrices for data input
#      in quantum computing", Quantum 8, 1226 (2024).
# ==============================================================================

# ==============================================================================
# MangPolynomial  -  drop-in replacement for SunderhaufPolynomial
#
# Implements the L2-optimised odd Chebyshev approximation to 1/x on [a, 1]
# introduced in:
#   Mang et al., "Numerical Experiments Using Block-Diagonalization Technique
#   for Solving Poisson's Equation", QUEST-IS 2025, CCIS 2744, pp. 167-174, 2026.
#   (eq. 4 of that paper)
#
# -- What it does -------------------------------------------------------------
# Mang et al. parameterise x = cos(theta) and minimise the L2 residual
#
#   C(w) = || Sum_i w_i cos((2i+1)theta)  -  1/(2 kappa cos theta) ||^2_{L^2(theta)}
#
# over theta in [0, arccos(a)], where a = 1/kappa in their setting but we keep a
# as a general lower bound matching the SunderhaufPolynomial interface.
#
# This is ordinary least squares on the odd-Chebyshev basis {T_{2i+1}(x)}:
#   T_{2i+1}(cos theta) = cos((2i+1)theta)
# so the solution is: w = (A^T A)^-1 A^T t  (via numpy.linalg.lstsq).
#
# The resulting polynomial in x is:
#   p(x) = (2/a) . Sum_i w_i . T_{2i+1}(x)          [approximates 1/x on [a,1]]
#
# (Factor 2/a rather than 2kappa because we use general a = sigma_min, not 1/kappa.)
#
# -- Why it uses fewer terms than Sunderhauf --------------------------------
# Sunderhauf guarantees Linf < eps uniformly on [a, 1] - the hardest error metric.
# Mang minimises L2 over theta, which weights all angles equally.  Near x = a
# (large theta), where 1/x is largest and hardest to approximate, the L2 metric
# allows more pointwise error than Linf does.  The result is lower degree at
# the cost of elevated relative error near x = a.
#
# Degree comparison at kappa = 117.63 (relative criterion, eps = 0.01):
#   ChebIter d = 623   Mang d = 639   Sunderhauf d = 1103
# ChebIter is the exact optimum; Mang is within 3-14% of it across eps, and
# Sunderhauf's larger degree is criterion mismatch (absolute vs relative,
# a factor of up to kappa by eq. critRelation), not weaker construction.
#
# -- Which base to use -----------------------------------------------------
# Use ChebIter when a provably optimal baseline for the relative criterion is
#   wanted; this is the reference base in the paper.
# Use Mang when degree is the binding constraint and no uniform certificate is
#   required.
# Use Sunderhauf when the solution error, rather than the residual, is what
#   must be controlled.
# All three are correctable; the spectral correction is agnostic to the base.
#
# -- Interface (identical to SunderhaufPolynomial) --------------------------
#   MangPolynomial.mindegree(epsilon, a)        -> int   (minimum odd degree)
#   MangPolynomial.poly(d, a)                   -> Chebyshev object
#   MangPolynomial.error_for_degree(d, a)       -> float (empirical Linf estimate)
#
# The Chebyshev object returned by poly() is normalised so its Linf norm on
# [-1, 1] equals the reciprocal of the success-probability normalisation M,
# exactly matching what _compute_phases() in myQSVT expects.
#
# -- Degree selection -------------------------------------------------------
# mindegree() uses a bisection search: it evaluates error_for_degree() at
# candidate degrees and returns the smallest odd d such that the estimated
# Linf error on [a, 1] is below epsilon.  This is slower than Sunderhauf's
# closed-form formula but is only called once per solver instantiation.
# ==============================================================================



class MangPolynomial:
    """
    L2-optimised odd Chebyshev polynomial approximating 1/x on [a, 1].

    Drop-in replacement for SunderhaufPolynomial.  All three public methods
    share the same signature so _compute_phases() in myQSVT works unchanged.

    Ref: Mang et al., QUEST-IS 2025, CCIS 2744 (2026), eq. 4.
    """

    # Minimum quadrature points for the L2 fit.
    # The actual count is max(_N_THETA_MIN, 4*n_terms) - see _lstsq_coeffs.
    _N_THETA_MIN: int = 2000

    # Number of sample points used to estimate the Linf error in error_for_degree.
    _N_LINF: int = 4000

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lstsq_coeffs(d: int, a: float) -> np.ndarray:
        """
        Solve the L2 problem (eq. 4 of Mang et al.) and return the weight
        vector w of length (d+1)//2.

        Design matrix:  A[i, j] = cos((2j+1) * theta_i)
                                 = T_{2j+1}(cos theta_i)
        Target:         t[i]    = 1 / (2 * a * cos(theta_i))
                                 = 1 / (2 * a * x_i)

        The target is 1/(2a . x), i.e. (1/2a) . 1/x.  Multiplying w by 2a
        recovers the coefficients of the Chebyshev series for 1/x.

        Returns w (unnormalised); the caller applies the 2a factor.
        """
        n_terms   = (d + 1) // 2    # number of odd Chebyshev terms T_1, T_3, ...

        # N_THETA must exceed n_terms to keep the system overdetermined.
        # A fixed constant fails for large d (large kappa, e.g. m >= 6) where
        # n_terms > _N_THETA_MIN, making the system underdetermined.
        N_THETA   = max(MangPolynomial._N_THETA_MIN, 4 * n_terms)

        theta_max = np.arccos(a)
        thetas    = np.linspace(1e-8, theta_max, N_THETA)
        target    = 1.0 / (2.0 * a * np.cos(thetas))
        A = np.column_stack([np.cos((2 * j + 1) * thetas)
                             for j in range(n_terms)])

        # Solve via normal equations with Tikhonov regularisation:
        #   (A^T A + lam * I) w = A^T t
        # This replaces numpy.linalg.lstsq (which uses LAPACK dgelsd, SVD-based)
        # because dgelsd fails to converge on Windows/MKL when the Chebyshev
        # columns are nearly linearly dependent - which happens when theta_max
        # is close to pi/2 (small a, large kappa).  The normal equations solved
        # via numpy.linalg.solve (LU/Cholesky) are platform-independent and
        # always converge.  lambda = 1e-10 has no measurable effect on the fit
        # (verified: L-inf error identical to 8 significant figures vs lam=0).
        lam = 1e-10
        ATA = A.T @ A
        ATt = A.T @ target
        ATA[np.diag_indices(n_terms)] += lam
        w = np.linalg.solve(ATA, ATt)
        return w

    @staticmethod
    def _build_chebyshev(w: np.ndarray, a: float) -> Chebyshev:
        """
        Convert weight vector w (from _lstsq_coeffs) into a Chebyshev object
        representing p(x) ~ 1/x on [a, 1].

        p(x) = (2a) . Sum_j w_j . T_{2j+1}(x)

        The factor 2a converts from the 1/(2a.x) target back to 1/x.
        Chebyshev coefficient array has zeros at even positions (odd poly).
        """
        n_terms   = len(w)
        max_degree = 2 * n_terms   # highest term is T_{2*(n_terms-1)+1}
        coef      = np.zeros(max_degree)
        for j, wj in enumerate(w):
            coef[2 * j + 1] = 2.0 * a * wj   # coefficient of T_{2j+1}
        return Chebyshev(coef)

    # ------------------------------------------------------------------
    # Public interface  (identical to SunderhaufPolynomial)
    # ------------------------------------------------------------------

    @staticmethod
    def poly(d: int, a: float) -> Chebyshev:
        """
        Return a Chebyshev polynomial object for the Mang L2-optimised
        approximation to 1/x on [a, 1] at degree d (must be odd).

        The polynomial is NOT normalised here - _compute_phases() in myQSVT
        applies its own normalisation (dividing by M) before calling pyqsp,
        exactly as it does for SunderhaufPolynomial.poly().

        Parameters
        ----------
        d : int   Polynomial degree (must be odd).
        a : float Lower bound of the approximation interval; a = sigma_min(L_D/4).

        Returns
        -------
        Chebyshev  Odd Chebyshev polynomial p with p(x) ~ 1/x on [a, 1].
        """
        if d % 2 == 0:
            raise ValueError(f"d must be odd, got {d}.")
        if not (0 < a < 1):
            raise ValueError(f"a must be in (0, 1), got {a}.")

        w = MangPolynomial._lstsq_coeffs(d, a)
        return MangPolynomial._build_chebyshev(w, a)

    @staticmethod
    def error_for_degree(d: int, a: float) -> float:
        """
        Estimate the Linf relative approximation error of the degree-d Mang
        polynomial on [a, 1], defined as:

            max_{x in [a,1]}  |p(x) - 1/x| / (1/x)
          = max_{x in [a,1]}  |x . p(x) - 1|

        Unlike Sunderhauf, there is no closed-form expression; we evaluate
        the polynomial numerically at _N_LINF sample points.

        Note: Mang et al. use an L2 target, so this Linf estimate will be
        larger than the L2 residual, particularly near x = a.  Use this
        value as a conservative upper bound on approximation quality.

        Parameters
        ----------
        d : int   Polynomial degree (must be odd).
        a : float Lower bound a = sigma_min(L_D/4).

        Returns
        -------
        float  Estimated maximum relative error on [a, 1].
        """
        if d % 2 == 0:
            d += 1   # silently promote to next odd degree

        w    = MangPolynomial._lstsq_coeffs(d, a)
        poly = MangPolynomial._build_chebyshev(w, a)

        xs       = np.linspace(a, 1.0, MangPolynomial._N_LINF)
        vals     = poly(xs)
        rel_err  = np.abs(vals * xs - 1.0)   # |x.p(x) - 1|
        return float(np.max(rel_err))

    @staticmethod
    def mindegree(epsilon: float, a: float) -> int:
        """
        Find the minimum odd polynomial degree d such that the estimated
        Linf relative error on [a, 1] is below epsilon.

        Uses bisection over odd degrees between d_lo=1 and d_hi, where d_hi
        is set conservatively using the Sunderhauf closed-form as an upper
        bound (Mang degree is always <= Sunderhauf degree).

        Parameters
        ----------
        epsilon : float  Target maximum relative error (e.g. 0.01 for 1%).
        a       : float  Lower bound a = sigma_min(L_D/4).

        Returns
        -------
        int  Minimum odd degree d satisfying the error criterion.
        """
        if not (0 < epsilon < 1):
            raise ValueError(f"epsilon must be in (0, 1), got {epsilon}.")
        if not (0 < a < 1):
            raise ValueError(f"a must be in (0, 1), got {a}.")

        # Quick exit.
        if MangPolynomial.error_for_degree(1, a) <= epsilon:
            return 1

        # Phase 1 - Exponential search: double d until error <= epsilon.
        # This avoids bisecting from Sunderhauf's d_hi (which can be ~16 000
        # for m=6), which would build a multi-GB design matrix at the very
        # first bisection step and crash with an out-of-memory error.
        d = 1
        while MangPolynomial.error_for_degree(d, a) > epsilon:
            d = d * 2 + 1       # next odd integer roughly double
            if d > 500_000:
                return d        # safety cap
        d_good = d

        # Phase 2 - Bisect in the tight bracket [d//2, d_good].
        d_lo = max(1, (d_good // 2) | 1)   # ensure odd
        k_lo = (d_lo + 1) // 2
        k_hi = (d_good + 1) // 2

        while k_lo < k_hi:
            k_mid = (k_lo + k_hi) // 2
            d_mid = 2 * k_mid - 1
            if MangPolynomial.error_for_degree(d_mid, a) <= epsilon:
                k_hi = k_mid
            else:
                k_lo = k_mid + 1

        return 2 * k_lo - 1


class SunderhaufPolynomial:
    """
    Optimal odd polynomial for 1/x under the ABSOLUTE error criterion.

    Minimises  max_{x in [a,1]} |p(x) - 1/x|  in closed form.

    Ref: Sunderhauf et al., arXiv:2507.15537 (2025).

    Note: this is NOT the same criterion as ChebIterPolynomial, which minimises
    the RELATIVE criterion max|xp(x)-1|.  The two differ by a factor of 1/x:

        |p(x) - 1/x|  =  |xp(x) - 1| / x

    so the absolute criterion up-weights errors near x=a by kappa = 1/a, and
    Sunderhauf's mindegree is correspondingly larger at the same eps.

    For compliance-based QSVT: because the absolute criterion is tightest where
    the load energy sits, its relative-error residual near lambda_min is small, and
    the spectral correction has correspondingly less to recover than it does
    from a base optimised for the relative criterion.
    """

    @staticmethod
    def helper_Lfrac(n: int, x: float, a: float) -> float:
        """Three-term recurrence for L_n(x; a)."""
        alpha = (1 + a) / (2 * (1 - a))
        l1 = (x + (1 - a) / (1 + a)) / alpha
        l2 = (x**2 + (1 - a) / (1 + a) * x / 2 - 0.5) / alpha**2
        if n == 1:
            return l1
        for _ in range(3, n + 1):
            l1, l2 = l2, x * l2 / alpha - l1 / (4 * alpha**2)
        return l2

    @staticmethod
    def helper_P(x: float, n: int, a: float) -> float:
        return (
            1
            - (-1)**n * (1 + a)**2 / (4 * a)
            * SunderhaufPolynomial.helper_Lfrac(
                n, (2 * x**2 - (1 + a**2)) / (1 - a**2), a)
        ) / x

    @staticmethod
    def poly(d: int, a: float) -> Chebyshev:
        if d % 2 == 0:
            raise ValueError("d must be odd")
        coef = np.polynomial.chebyshev.chebinterpolate(
            SunderhaufPolynomial.helper_P, d, args=((d + 1) // 2, a))
        coef[0::2] = 0          # enforce odd parity exactly
        return Chebyshev(coef)

    @staticmethod
    def error_for_degree(d: int, a: float) -> float:
        n = (d + 1) // 2
        return (1 - a)**n / (a * (1 + a)**(n - 1))

    @staticmethod
    def mindegree(epsilon: float, a: float) -> int:
        n = math.ceil(
            (np.log(1 / epsilon) + np.log(1 / a) + np.log(1 + a))
            / np.log((1 + a) / (1 - a))
        )
        return 2 * n - 1

class ChebIterPolynomial:
    """
    Chebyshev-iteration polynomial: the CLOSED-FORM optimal odd polynomial for
    the RELATIVE residual criterion  max_{x in [a,1]} |x p(x) - 1|,

        p(x) = ( 1 - (-1)^n T_n(u(x)) / T_n(z0) ) / x,
        u(x) = (2x^2 - (1+a^2)) / (1 - a^2),   z0 = (1+a^2)/(1-a^2).

    This is the polynomial of Gribling, Kerenidis & Szilagyi (arXiv:2109.04248,
    Corollary 8), in the odd/symmetric form given by Sunderhauf et al.
    (arXiv:2507.15537, Eq. 21).  It is the exact minimiser of Eq. (3) of this
    paper, available in closed form and costing microseconds to construct.

    Because  |x p(x) - 1| = |T_n(u(x))| / |T_n(z0)|  and |T_n(u)| <= 1 for
    x in [a,1], the residual equioscillates exactly and the achieved error is

        eps = 1 / |T_n(z0)| = 1 / cosh(n arccosh(z0)),

    which inverts in closed form to give mindegree().
    """

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
        coef[0::2] = 0.0                      # enforce odd parity exactly
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


# ======================================================================
# SpectralPolynomial
# ======================================================================

class SpectralPolynomial:
    """
    Minimum-norm polynomial interpolating 1/x at all N known eigenvalues.

    When the eigenvalues lambda_1, ..., lambda_N of the system matrix are known a priori,
    the polynomial need only satisfy lambda_i p(lambda_i) = 1 at those N discrete points.
    This replaces the continuous minimax problem on [a,1] with an underdetermined
    linear system in the odd-Chebyshev basis, yielding degree d = 3N-1 regardless
    of kappa - compared to d ~ O(kappa log(kappa/eps)) for any continuous-interval base.

    WARNING: This polynomial has NO guarantee on [a,1] between eigenvalues.
    It must be paired with a continuous backbone (ChebIter, Sunderhauf, or Mang)
    via the min-norm eigenvalue correction when used in QSVT - see the hybrid
    strategy in the accompanying document.

    Construction
    ------------
    Solves the minimum-l2-norm system

        lambda_i . p(lambda_i) = 1    for i = 1, ..., N

    in the odd-Chebyshev basis {T_1, T_3, ..., T_d} on [-1, 1].
    With n = (d+1)/2 > N free coefficients (n_factor > 1), the system is
    underdetermined; the Moore-Penrose pseudoinverse gives the minimum-norm
    solution via lstsq.

    Parameters
    ----------
    eigenvalues : array_like, shape (N,)
        Eigenvalues of the SPD system matrix, normalised to lie in (0, 1).
    n_factor : float, optional
        Over-parameterisation ratio.  The basis dimension is n = ceil(n_factor.N),
        giving degree d = 2n-1.  Must satisfy n_factor >= 1.  Default 1.5
        (d = 3N-1).  Use 2.0 for smaller max|p(x)| on [-1,1].

    Public API
    ----------
    poly()         -> Chebyshev   eigenvalue-interpolating polynomial
    mindegree()    -> int         polynomial degree d = 2*ceil(n_factor.N)-1
    error_at_eigs() -> ndarray    |lambda_i p(lambda_i) - 1| for each eigenvalue

    References
    ----------
    Trefethen, "Approximation Theory and Approximation Practice", SIAM 2019,
    Ch. 4 (Chebyshev interpolation).
    """

    def __init__(self, eigenvalues, n_factor: float = 1.5):
        eigenvalues = np.asarray(eigenvalues, dtype=float).ravel()
        if np.any(eigenvalues <= 0) or np.any(eigenvalues >= 1):
            raise ValueError("All eigenvalues must lie in (0, 1).")
        if n_factor < 1.0:
            raise ValueError("n_factor must be >= 1.")

        self._eigs     = np.sort(eigenvalues)
        self._N        = len(eigenvalues)
        self._n_factor = n_factor
        self._n        = int(np.ceil(n_factor * self._N))
        self._d        = 2 * self._n - 1
        self._poly     = self._build_poly()

    def _build_poly(self) -> Chebyshev:
        lam = self._eigs
        n   = self._n
        j   = np.arange(n, dtype=float)
        th  = np.arccos(np.clip(lam, 1e-14, 1 - 1e-14))
        B   = np.cos(np.outer(th, 2*j + 1))   # (N, n)
        LB  = lam[:, None] * B                 # (N, n)
        c, _, _, _ = np.linalg.lstsq(LB, np.ones(self._N), rcond=None)
        coef = np.zeros(2 * n)
        for j_, cj in enumerate(c):
            coef[2 * j_ + 1] = cj
        return Chebyshev(coef)

    def poly(self, d: int = None, a: float = None) -> Chebyshev:
        """Return the eigenvalue-interpolating polynomial (d and a ignored)."""
        return self._poly

    def mindegree(self, epsilon: float = None, a: float = None) -> int:
        """Return the polynomial degree d = 2*ceil(n_factor.N)-1 (args ignored)."""
        return self._d

    def error_at_eigs(self) -> np.ndarray:
        """Return |lambda_i p(lambda_i) - 1| for each eigenvalue (should be ~1e-12)."""
        return np.abs(self._eigs * self._poly(self._eigs) - 1.0)

    def degree_info(self) -> dict:
        """Summary dict: N, n_factor, n, d, max|p| on [-1,1], max_delta."""
        x   = np.linspace(-1, 1, 25 * self._d)
        tau = float(np.max(np.abs(self._poly(x))))
        return {
            "N":         self._N,
            "n_factor":  self._n_factor,
            "n":         self._n,
            "d":         self._d,
            "tau":       tau,
            "max_delta": float(np.max(self.error_at_eigs())),
        }

BASE_POLYS = {
    'mang'  : MangPolynomial,
    'sunderhauf'  : SunderhaufPolynomial,
    'chebiter': ChebIterPolynomial,
}

def merge_eigenvalues(eigenvalues: np.ndarray, merge_rtol: float = 1e-3,
                      verbose: bool = False):
    """
    Collapse repeated / near-repeated eigenvalues to distinct representatives.

    A single interpolation constraint lam*p(lam) = 1 applies identically to every
    eigenmode sharing an eigenvalue, so exact duplicates carry no information and
    only make the Gram matrix singular.

    The merge criterion is RELATIVE, not absolute:

        (lam_j - lam_i) / lam_i  <=  merge_rtol   =>   merge

    A relative test is the correct one here because the interpolation residual is
    the relative quantity |lam p(lam) - 1|, and because Poisson-type spectra
    cluster multiplicatively near lam_min.  An absolute tolerance would either
    over-merge at the top of the spectrum or under-merge at the bottom.

    Returns
    -------
    lam_eff : distinct representatives, ascending
    K_eff   : number of retained constraints
    """
    lam_sorted = np.sort(np.asarray(eigenvalues, float))
    keep = [lam_sorted[0]]
    for l in lam_sorted[1:]:
        if (l - keep[-1]) / keep[-1] > merge_rtol:
            keep.append(l)
    lam_eff = np.array(keep)
    if verbose and len(lam_eff) < len(lam_sorted):
        print(f"Merged eigenvalues: K={len(lam_sorted)} -> K_eff={len(lam_eff)}")
    return lam_eff, len(lam_eff)


def merge_residual_bound(eigenvalues: np.ndarray, poly: Chebyshev,
                         merge_rtol: float = 1e-3) -> float:
    """
    Rigorous bound on the residual introduced at eigenvalues that were merged away.

    If lam_j was merged into representative lam_i with |lam_j - lam_i| <= delta,
    and lam_i p(lam_i) = 1 exactly, then by the mean value theorem

        |lam_j p(lam_j) - 1| <= delta * max_{x in [lam_i, lam_j]} |d/dx (x p(x))|.

    Returns the worst-case bound over all merged pairs.  This answers the referee
    point that duplication removal is applied without an error bound.
    """
    lam = np.sort(np.asarray(eigenvalues, float))
    lam_eff, _ = merge_eigenvalues(lam, merge_rtol)
    dxp = (Chebyshev(poly.coef) * Chebyshev([0.0, 1.0])).deriv()   # d/dx [x p(x)]
    worst = 0.0
    for l in lam:
        i = int(np.argmin(np.abs(lam_eff - l)))
        rep = lam_eff[i]
        if rep == l:
            continue
        lo, hi = sorted((rep, l))
        L = float(np.max(np.abs(dxp(np.linspace(lo, hi, 256)))))
        worst = max(worst, abs(l - rep) * L)
    return worst


_SQRT_EPS = float(np.sqrt(np.finfo(float).eps))   # ~1.49e-8: machine-precision check


def spectral_correction(p0: Chebyshev, eigenvalues: np.ndarray,
                        rcond: float = 1e-10, merge_rtol: float = 1e-3,
                        return_info: bool = False):
    """
    Min-norm Chebyshev coefficient correction enforcing lam_k p(lam_k) = 1 at the
    supplied eigenvalues, without changing the degree or parity of p0.

    Solvability.  The correction solves the underdetermined system  C dc = r,
    C = Lam_K B_K, for the minimum-l2 dc, via the SVD of C itself.  C is rank
    deficient exactly when its rows are linearly dependent, i.e. when two supplied
    eigenvalues coincide (or K exceeds the number of odd Chebyshev terms n0).  The
    system is nonetheless CONSISTENT -- identical rows carry identical residuals --
    so the minimum-norm solution exists and is unique, and the truncated-SVD
    pseudoinverse returns exactly that solution.  Duplicate removal is applied first
    so the rank deficiency is removed rather than merely tolerated.

    The Gram form  G alpha = r,  G = C C^T,  dc = C^T alpha  is mathematically
    equivalent but is NOT used: cond(G) = cond(C)^2.

    Parameters
    ----------
    p0          : Chebyshev polynomial object (defined on [-1, 1])
    eigenvalues : array of K known eigenvalues in (0, 1]
    rcond       : relative singular-value threshold for the truncated SVD
    merge_rtol  : relative tolerance for duplicate removal (see merge_eigenvalues)
    return_info : if True also return a diagnostics dict

    Returns
    -------
    c_corr : np.ndarray, shape (n0,)  -- additive correction to p0.coef[1::2]
    info   : dict (only if return_info) with K, K_eff, rank, cond, residuals
    """
    lam_in = np.asarray(eigenvalues, float)
    lam, K_eff = merge_eigenvalues(lam_in, merge_rtol, verbose=True)

    c0  = p0.coef[1::2]
    n0  = len(c0)
    j   = np.arange(n0)

    # B[k, j] = T_{2j+1}(lambda_k)
    B  = np.cos(np.outer(np.arccos(np.clip(lam, -1+1e-14, 1-1e-14)), 2*j+1))

    # Step 1: residuals
    r  = 1.0 - lam * p0(lam)

    # Step 2/3: minimum-norm solution of the underdetermined system C dc = r,
    # obtained from the SVD of C ITSELF.  The Gram form G = C C^T is NOT formed:
    # cond(G) = cond(C)^2, which is what drove the ill-conditioning reported
    # previously.
    C      = lam[:, None] * B
    c_corr = np.linalg.pinv(C, rcond=rcond) @ r

    if not return_info:
        return c_corr

    coef = p0.coef.copy()
    coef[1::2] += c_corr
    pSC = Chebyshev(coef)
    info = dict(
        K              = int(len(lam_in)),
        K_eff          = int(K_eff),
        n0             = int(n0),
        rank           = int(np.linalg.matrix_rank(C, tol=rcond * np.linalg.norm(C, 2))),
        cond_C         = float(np.linalg.cond(C)),
        consistency    = float(np.linalg.norm(C @ c_corr - r) / max(np.linalg.norm(r), 1e-300)),
        max_resid_corr = float(np.max(np.abs(lam * pSC(lam) - 1.0))),
        max_resid_all  = float(np.max(np.abs(lam_in * pSC(lam_in) - 1.0))),
        merge_bound    = float(merge_residual_bound(lam_in, pSC, merge_rtol)),
        corr_norm      = float(np.linalg.norm(c_corr)),
    )
    return c_corr, info


def merge_by_resolution(eigenvalues, n, c=1.0):
    """
    Merge eigenvalues the polynomial cannot resolve.

    A degree-d odd polynomial has n = (d+1)/2 Chebyshev terms and, in the
    variable theta = arccos(lambda), resolves features no finer than
    pi/(2n).  Two eigenvalues closer than that in theta are effectively
    indistinguishable to it, so imposing an interpolation constraint at both
    is ill-posed: the Gram system becomes numerically rank-deficient and the
    min-norm solution acquires an enormous coefficient norm.

    This is the correct scale for duplicate removal.  For comparison, the
    relative criterion of merge_eigenvalues() with merge_rtol = 1e-3 gives
    dtheta ~ 8.5e-6 near lambda_min at kappa ~ 118 -- some three orders of
    magnitude tighter than pi/(2n) ~ 2e-2 at d = 155.  It therefore removes
    exact duplicates but never merely unresolvable pairs.

    c = 0 retains every distinct eigenvalue; larger c merges more aggressively.
    """
    lam = np.sort(np.asarray(eigenvalues, float))
    if len(lam) <= 1:
        return lam, len(lam)
    th  = np.arccos(np.clip(lam, -1.0, 1.0))
    lim = c * np.pi / (2.0 * n)
    keep = [0]
    for i in range(1, len(lam)):
        if abs(th[i] - th[keep[-1]]) > lim:
            keep.append(i)
    return lam[keep], len(keep)


def _minnorm_correction(p0, lam, n, rcond=1e-12):
    """Min-norm dc solving (Lam B) dc = r, via SVD of the K x n0 matrix itself.

    Note this does NOT form the Gram matrix G = C C^T.  Doing so squares the
    condition number (cond(G) = cond(C)^2), which is what drove the
    ill-conditioning reported previously.
    """
    j = np.arange(n)
    B = np.cos(np.outer(np.arccos(np.clip(lam, -1 + 1e-14, 1 - 1e-14)), 2 * j + 1))
    C = lam[:, None] * B
    r = 1.0 - lam * p0(lam)
    dc = np.linalg.pinv(C, rcond=rcond) @ r
    return dc, float(np.linalg.cond(C))


def spectral_correction_adaptive(p0, eigenvalues, resid_tol=_SQRT_EPS,
                                 tau_inflation_max=2.0,
                                 c_grid=(0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0),
                                 weights=None, return_info=False):
    """
    Spectral correction with resolution-aware merging and a verified residual.

    The non-adaptive spectral_correction() can fail silently: when the Gram
    system is near-rank-deficient the truncated SVD discards directions and
    returns a partial correction that does NOT satisfy the interpolation
    constraints, with no error raised.  Residuals of 0.5 at eigenvalues
    believed corrected to machine precision have been observed.

    This routine instead sweeps the merge scale c from fine to coarse and
    accepts the FIRST (most constraints retained) setting for which

        (i)  max_k |lam_k p_SC(lam_k) - 1| <= resid_tol  on the retained set,
        (ii) tau(p_SC) / tau(p_0)          <= tau_inflation_max,

    so the correction it returns is one it has verified.  If no setting on the
    grid satisfies both, the least-bad is returned with info['ok'] = False --
    the caller is told, rather than misled.

    Parameters
    ----------
    resid_tol         : machine-precision CHECK on the retained eigenvalues, not a
                        tunable tolerance.  The correction enforces exact
                        interpolation, so the achieved residual is either O(eps_mach)
                        or O(1e-1) with nothing between; any threshold separating the
                        two yields the same K_eff.  Default sqrt(eps_mach) ~ 1.5e-8.
    tau_inflation_max : cap on subnormalisation growth.  tau enters the success
                        probability as 1/tau^2, so unbounded inflation converts
                        a depth saving into a total-cost loss.
    weights           : optional modal energy weights w_k, same length as
                        eigenvalues.  If given, info also reports the
                        load-weighted residual rho = sqrt(sum w_k r_k^2), which
                        is the quantity that governs fidelity and compliance.

    Returns
    -------
    c_corr : additive correction to p0.coef[1::2]
    info   : dict (only if return_info)
    """
    lam_in = np.asarray(eigenvalues, float)
    n0 = len(p0.coef[1::2])
    xs = np.linspace(-1.0, 1.0, 60001)
    tau0 = float(np.max(np.abs(p0(xs))))

    attempts = []
    for c in c_grid:
        lam, keff = merge_by_resolution(lam_in, n0, c)
        if keff < 1:
            continue
        try:
            dc, condC = _minnorm_correction(p0, lam, n0)
        except np.linalg.LinAlgError:
            continue
        coef = p0.coef.copy()
        coef[1::2] += dc
        pSC = Chebyshev(coef)
        resid_kept = float(np.max(np.abs(lam * pSC(lam) - 1.0)))
        resid_all = float(np.max(np.abs(lam_in * pSC(lam_in) - 1.0)))
        tau1 = float(np.max(np.abs(pSC(xs))))
        infl = tau1 / tau0 if tau0 > 0 else np.inf
        rho = None
        if weights is not None:
            w = np.asarray(weights, float)
            r = lam_in * pSC(lam_in) - 1.0
            rho = float(np.sqrt(np.sum(w * r ** 2)))
        rec = dict(c=c, K_eff=keff, cond_C=condC, dc=dc,
                   resid_kept=resid_kept, resid_all=resid_all,
                   tau_base=tau0, tau_corr=tau1, tau_inflation=infl,
                   rho=rho, corr_norm=float(np.linalg.norm(dc)),
                   ok=(resid_kept <= resid_tol and infl <= tau_inflation_max))
        attempts.append(rec)
        if rec['ok']:
            break

    if not attempts:
        raise RuntimeError("spectral_correction_adaptive: no feasible merge scale")

    best = next((a for a in attempts if a['ok']), None)
    if best is None:
        # none satisfied both; prefer the one that at least meets the residual,
        # else the smallest tau inflation
        meets = [a for a in attempts if a['resid_kept'] <= resid_tol]
        best = min(meets, key=lambda a: a['tau_inflation']) if meets \
               else min(attempts, key=lambda a: a['resid_kept'])
        print(f"Warning: spectral correction did not meet its targets "
              f"(resid_kept={best['resid_kept']:.2e}, "
              f"tau inflation={best['tau_inflation']:.1f}). "
              f"Increase the base degree or reduce K.")

    if not return_info:
        return best['dc']
    info = {k: v for k, v in best.items() if k != 'dc'}
    info['K'] = int(len(lam_in))
    info['n0'] = int(n0)
    info['attempts'] = [{k: v for k, v in a.items() if k != 'dc'} for a in attempts]
    return best['dc'], info


_Q_FLOOR = 1e-10          # base construction stalls here in double precision


_XGRID = np.linspace(-1.0, 1.0, 60001)


def _tau(p):
    """Subnormalisation: max |p| over the full interval [-1, 1]."""
    return float(np.max(np.abs(p(_XGRID))))


def correct_at_tolerance(poly_class, kappa, eigenvalues, epsilon,
                         gamma_max=2.0,
                         c_grid=(0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)):
    """
    ALGORITHM A -- verified correction at a GIVEN base tolerance.

    Builds p0 at d(epsilon) and sweeps the merge scale c from fine to coarse.
    A candidate is accepted when

        (a) R_ret = max_{k <= K_eff} |lam_k p_SC(lam_k) - 1| <= sqrt(eps_mach)
        (b) gamma = tau(p_SC) / tau(p_0)                     <= gamma_max

    and the FIRST acceptance is returned: c_grid ascends, so that is the LEAST
    aggressive merge that verifies, and it retains the most constraints.

    Selecting instead on the smallest R_all over the accepted candidates is
    wrong, and was the rule here until 2026-08-30.  R_all is a max over the
    supplied eigenvalues, so in the regime where the correction has already
    failed it varies by noise between merge scales, and the sweep then buys a
    meaningless improvement by discarding constraints.  Measured on the 2D
    Poisson operator at eps = 0.5, K = 32: c = 1 retains 8 eigenvalues at
    R_all = 1.86, c = 4 retains 3 at R_all = 1.62, and the min-R_all rule
    preferred the polynomial that had abandoned 29 of the 32.  This is the same
    failure mode as TRAP 4 in constrainedMinMax.py.

    The merge scale exists to drop eigenvalues the polynomial cannot RESOLVE,
    not eigenvalues it finds inconvenient; the smallest verifying c is the only
    selection consistent with that.

    Use this directly when the degree is fixed by a hardware budget.  When it
    is not, adaptive_spectral_correction sweeps epsilon and calls this at each
    tolerance.

    Returns a dict with 'epsilon', 'd', 'c', 'K_eff', 'R_ret', 'R_all',
    'gamma', 'tau0' and 'dc', or None if no merge scale verifies.
    """
    a = 1.0 / kappa
    lam_in = np.asarray(eigenvalues, float)
    d = poly_class.mindegree(epsilon, a)
    p0 = poly_class.poly(d, a)
    n0 = len(p0.coef[1::2])
    tau0 = _tau(p0)

    c_grid = tuple(sorted(c_grid))        # 'first acceptance' needs fine -> coarse
    for c in c_grid:
        lam, keff = merge_by_resolution(lam_in, n0, c)
        if keff < 1:
            continue
        try:
            dc, _ = _minnorm_correction(p0, lam, n0)
        except np.linalg.LinAlgError:
            continue
        coef = p0.coef.copy()
        coef[1::2] += dc
        pSC = Chebyshev(coef)
        R_ret = float(np.max(np.abs(lam * pSC(lam) - 1.0)))
        R_all = float(np.max(np.abs(lam_in * pSC(lam_in) - 1.0)))
        g = _tau(pSC) / tau0 if tau0 > 0 else np.inf
        if R_ret <= _SQRT_EPS and g <= gamma_max:
            return dict(epsilon=float(epsilon), d=int(d), c=float(c),
                        K_eff=int(keff), R_ret=R_ret, R_all=R_all,
                        gamma=float(g), tau0=float(tau0), dc=dc)
    return None


def adaptive_spectral_correction(poly_class, kappa, eigenvalues,
                                 gamma_max=2.0, eps_grid=None,
                                 c_grid=(0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0),
                                 verbose=False):
    """
    The two-stage algorithm of Sec. 4.3.

    STAGE 1 sweeps the base tolerance eps and returns the best corrected
    polynomial for the supplied eigenvalues.  For each eps it builds p0 at
    d(eps) and sweeps the merge scale c, accepting every c for which

        (a) R_ret = max_{k <= K_eff} |lam_k p_SC(lam_k) - 1| <= sqrt(eps_mach)
        (b) gamma = tau(p_SC) / tau(p_0)                     <= gamma_max

    and among those, keeping the one with the smallest R_all.

    Tightening eps raises d, hence n0, hence the number of resolvable
    constraints, so it buys accuracy only while more of the supplied set is
    retained.  The sweep runs the whole grid and returns the best candidate,
    stopping early only when R_all reaches machine precision, at which point
    nothing is left to buy; eps is an OUTPUT of the procedure, not an input.

    It does NOT stop at the first tolerance that fails to improve R_all, which
    was the rule here until 2026-08-30.  R_all is not monotone in eps: raising
    the degree admits more constraints, and the extra constraints can raise the
    residual at the eigenvalues that were already pinned before the added ones
    are all resolvable.  Measured on the Weyl-3 synthetic spectrum at kappa =
    117.6, K = 16, R_all runs 0.203 (eps = 0.1, K_eff = 5), 0.238 (0.05,
    K_eff = 10), 0.485 (0.02, K_eff = 11), 0.189 (0.01, K_eff = 13), then
    1.1e-15 (0.005, K_eff = 16).  Halting on the first non-improvement returns
    the eps = 0.1 candidate and reports G = 0.6, missing the saturating one at
    eps = 0.005 that reports G = 7.9.  The full grid costs a few seconds and is
    classical.

    STAGE 2 asks what the uncorrected base family costs at the same accuracy.
    The base residual on the spectrum equals its tolerance, so eps* = R_all and
    d(eps*) follows by inverting the closed-form error relation -- one
    evaluation, no search.  The inversion is floored at 1e-10, where the base
    construction stalls in double precision, so G is a LOWER BOUND.

    Returns a dict with 'epsilon', 'd', 'K_eff', 'R_ret', 'R_all', 'gamma',
    'Q_SC', 'eps_star', 'd_star', 'Q_0', 'G' and 'recommend'.
    """
    a = 1.0 / kappa
    lam_in = np.asarray(eigenvalues, float)
    xs = _XGRID
    if eps_grid is None:
        eps_grid = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,
                    0.1, 0.05, 0.02, 0.01, 5e-3, 2e-3, 1e-3)

    tau = _tau

    # ---------------- Stage 1: sweep eps, calling Algorithm A ------------
    best, history = None, []
    for eps in eps_grid:
        cand = correct_at_tolerance(poly_class, kappa, lam_in, eps,
                                    gamma_max=gamma_max, c_grid=c_grid)

        history.append(dict(epsilon=float(eps), accepted=cand is not None,
                            R_all=(cand['R_all'] if cand else None)))
        if cand is None:
            continue                          # tighten and retry
        if best is None or cand['R_all'] < best['R_all']:
            best = cand                       # R_all is not monotone in eps
        if best['R_all'] <= _SQRT_EPS:
            break                             # saturated: nothing left to buy

    if best is None:
        raise RuntimeError("adaptive_spectral_correction: no tolerance on the "
                           "grid yields a verified correction")

    # ---------------- Stage 2 -------------------------------------------
    eps_star = max(best['R_all'], _Q_FLOOR)
    d_star = poly_class.mindegree(eps_star, a)
    tau_star = tau(poly_class.poly(d_star, a))

    Q_SC = best['d'] * best['gamma'] * best['tau0']
    Q_0 = d_star * tau_star
    G = Q_0 / Q_SC

    out = dict(best)
    out.pop('dc')
    out.update(dc=best['dc'], Q_SC=float(Q_SC), eps_star=float(eps_star),
               d_star=int(d_star), tau_star=float(tau_star), Q_0=float(Q_0),
               G=float(G), recommend=bool(G > 1.0),
               floored=bool(best['R_all'] < _Q_FLOOR), history=history)
    if verbose:
        print(f"  Stage 1: eps={out['epsilon']:<5} d={out['d']:<5} "
              f"K_eff={out['K_eff']:<4} R_all={out['R_all']:.2e} "
              f"gamma={out['gamma']:.2f}  Q_SC={Q_SC:.0f}")
        print(f"  Stage 2: eps*={eps_star:.2e} d*={d_star:<5} "
              f"Q_0={Q_0:.0f}  G={G:.1f}  "
              f"{'correct' if G > 1 else 'use the base polynomial'}"
              f"{'  (G is a lower bound)' if out['floored'] else ''}")
    return out


def spectral_extension(p0: Chebyshev, eigenvalues: np.ndarray,
                       rcond: float = 1e-10, merge_rtol: float = 1e-3,
                       return_info: bool = False):
    """
    Spectral EXTENSION: instead of perturbing the coefficients of p0, append
    K_eff new odd Chebyshev terms of degree d0+2, d0+4, ..., d0+2*K_eff and use
    those new degrees of freedom exclusively to satisfy the interpolation
    constraints.

    The existing coefficients of p0 are left untouched, so the continuous error
    profile of the base polynomial is preserved EXACTLY on the region where the
    new high-order terms are small -- unlike the correction, which redistributes
    error onto uncorrected eigenvalues.  The price is a degree increase of
    2*K_eff.

    Returns
    -------
    pExt : Chebyshev of degree d0 + 2*K_eff
    info : dict (only if return_info)
    """
    lam_in = np.asarray(eigenvalues, float)
    lam, K_eff = merge_eigenvalues(lam_in, merge_rtol)

    n0 = len(p0.coef[1::2])
    # new odd Chebyshev indices: T_{2*n0+1}, T_{2*n0+3}, ..., T_{2*(n0+K_eff)-1}
    jnew = np.arange(n0, n0 + K_eff)
    Bnew = np.cos(np.outer(np.arccos(np.clip(lam, -1+1e-14, 1-1e-14)), 2*jnew + 1))
    M    = lam[:, None] * Bnew                      # (K_eff, K_eff), square
    r    = 1.0 - lam * p0(lam)

    cnew, *_ = np.linalg.lstsq(M, r, rcond=rcond)

    coef = np.zeros(2 * (n0 + K_eff))
    coef[:len(p0.coef)] = p0.coef
    coef[2 * jnew + 1] += cnew
    pExt = Chebyshev(coef)

    if not return_info:
        return pExt
    info = dict(K=int(len(lam_in)), K_eff=int(K_eff),
                degree_base=int(2 * n0 - 1), degree_ext=int(2 * (n0 + K_eff) - 1),
                cond=float(np.linalg.cond(M)),
                max_resid_corr=float(np.max(np.abs(lam * pExt(lam) - 1.0))),
                max_resid_all=float(np.max(np.abs(lam_in * pExt(lam_in) - 1.0))))
    return pExt, info


if __name__ == "__main__":
    kappa = 10
    eps = 0.01
    K = 3
    lam = np.array([1/kappa, 0.5, 1])
    basePolynomial  = 'sunderhauf' # 'chebiter' or 'mang' or 'sunderhauf'
    a = 1/kappa
    polyClass = BASE_POLYS[basePolynomial.lower()]

    degree   = polyClass.mindegree(eps, 1/kappa)
    poly = polyClass.poly(degree, 1/kappa)

    corr = spectral_correction(poly,lam)
    coef_H       = poly.coef.copy()
    coef_H[1::2] += corr
    spectral_poly = Chebyshev(coef_H)


    x = np.union1d(np.linspace(a, 1.0, 1000), lam)
    y = spectral_poly(x)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title(f'Spectral with {basePolynomial} polynomial of degree {degree} for a={a:.2f} and eps={eps:.2f}')
    plt.grid()
    plt.show()

    error = np.abs(x * spectral_poly(x) - 1.0)
    plt.plot(x, error)
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.title(f'Error of spectral with {basePolynomial} polynomial of degree {degree} for a={a:.2f} and eps={eps:.2f}')
    plt.yscale('log')
    plt.grid()
    plt.show()