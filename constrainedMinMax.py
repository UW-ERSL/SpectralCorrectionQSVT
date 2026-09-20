"""
constrainedMinimax.py -- constrained min-max construction of QSVT polynomials.

STATUS 2026-09-17: this is the construction of the paper.  `spectral_constraint()` is
the procedure the paper states: merge at c = 1/2, solve the LP at d = 1.5 d(eps),
certify.  `constrained_minimax()` is the earlier research driver with a merge sweep
and the tail variant, kept for the rebuttal diagnostics.  See `caveats()`.

THE PROBLEM
-----------
QSVT applies an odd polynomial p of degree d to A (spectrum in [a,1], a = 1/kappa) and
we want p(x) ~ 1/x.  Write the residual  r(x) = x p(x) - 1.  It is EVEN, of degree
d+1, and satisfies r(0) = -1 identically.

Gribling solves      min_p  max_{x in [a,1]} |r(x)|                 -> eps(d) = 1/T_n(z0)

in closed form.  Given K known eigenvalues we solve the SAME extremal problem with
interpolation constraints, in one of two variants:

  FULL     min_p  max_{x in [a,1]}   |r(x)|   s.t. r(lam_k) = 0        <- variant "full"
  TAIL     min_p  max_{x in [mu,1]}  |r(x)|   s.t. r(lam_k) = 0        <- variant "tail"

where mu is a certified lower bound on every eigenvalue NOT supplied.

WHICH VARIANT
-------------
"full" is the safe one and the recommended default.  Its certificate covers the whole
continuum [a,1], so it bounds the residual at every eigenvalue of any matrix with
spectrum in that interval: no mu, no assumption that the supplied set is the K smallest,
and WRONG EIGENVALUES COST BENEFIT, NEVER SAFETY.  Because p0 is already minimax-optimal
on [a,1], the optimum always satisfies t >= eps(d): the price is paid in degree
(~1.5x to match the base's guarantee while also interpolating).

"tail" gives much better constants (~1.12x degree) but assumes the supplied set is the K
SMALLEST eigenvalues.  That hypothesis is unverifiable from the supplied data and fails
SILENTLY: with 15%-wrong eigenvalues it certified eta = 5.0e-5 while the true residual
over the spectrum was 0.66, four times worse than the base.  Use it only when the
ordering is genuinely known, and consider deriv_windows= below.

WHY NOT CONSTRAIN NOTHING / EVERYTHING
--------------------------------------
Minimising ||dc||_2 subject to interpolation only (the min-norm correction) leaves the
rest of the spectrum unbounded: R over the full spectrum reached 3.84 against a base
tolerance of 0.2, and whether it helps then depends entirely on the right-hand side.
Conversely, no degree-d polynomial beats eps(d) on [a,1], so "full" can never improve
the worst case at fixed degree -- it buys exactness at the K points instead.

TRAPS (each of these produced a wrong result during development)
---------------------------------------------------------------
1. ngrid MUST exceed the residual degree d+1.  Otherwise the polynomial oscillates
   freely between grid points, the LP reports success, and the certificate correctly
   catches a huge sup.  This module enforces ngrid >= 3*(d+1).
2. The LP's own optimum t* is a LOWER bound on the true sup (finitely many constraints
   is a relaxation).  NEVER quote it as the guarantee.  Quote certify().
3. The merge is not optional: perturbed eigenvalues split exact degeneracies into
   pairs closer than the degree resolves.  With no merge (c = 0) and 1% eigenvalue
   error, 2D Poisson at K = 32 keeps all 32 split values and needs 1.81 d(eps) to
   certify, against 1.36 d(eps) at c = 1/2.
4. Do not merge too coarsely either.  c = 1 (one full resolution cell) discards
   distinct eigenvalues the LP resolves without difficulty -- pairs at 0.1 of a cell
   are fine -- and leaves them unpinned: on 2D Poisson at K = 32 the accuracy gain at
   iso-guarantee falls from 248x (c = 0) and 236x (c = 1/2) to 16x (c = 1).
   c = 1/2 was within a few percent of the best c on every case tested (1D, 2D with
   0-10% eigenvalue error, 3D), so it is FIXED in spectral_constraint() and no sweep
   is run.  A sweep that selects on t cheats (c = 4 kept 5 of 18 eigenvalues and
   certified 6.6e-9 while the true residual over the spectrum was 1.29).
"""
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from scipy.optimize import linprog

from PolynomialApproximators import certify, _tau

__all__ = ["spectral_constraint", "constrained_minimax", "certify",
           "apply_correction", "merge_by_resolution", "caveats"]

C_MERGE = 0.5        # merge scale, in units of the resolution pi/(2 n0)
MU_DEGREE = 1.5      # degree multiplier over the base degree d(eps)


# ---------------------------------------------------------------- basis helpers
def _resid_basis(x, n0):
    """M[i,j] = x_i T_{2j+1}(x_i);  r(x) = r0(x) + M @ dc."""
    th = np.arccos(np.clip(x, -1.0 + 1e-14, 1.0 - 1e-14))
    return x[:, None] * np.cos(np.outer(th, 2 * np.arange(n0) + 1))


def _dresid_basis(x, n0):
    """d/dx [x T_m(x)] = T_m(x) + m x U_{m-1}(x),  m = 2j+1.
    With x = cos(th): = cos(m th) + m cos(th) sin(m th)/sin(th)."""
    th = np.arccos(np.clip(x, -1.0 + 1e-14, 1.0 - 1e-14))
    s = np.sin(th); s = np.where(np.abs(s) < 1e-12, 1e-12, s)
    m = 2 * np.arange(n0) + 1
    return np.cos(np.outer(th, m)) + (np.cos(th) / s)[:, None] * m * np.sin(np.outer(th, m))


def _cheb_grid(lo, hi, npts):
    """Cosine-spaced in theta: clusters where the residual oscillates fastest."""
    return np.cos(np.linspace(np.arccos(min(hi, 1.0)), np.arccos(max(lo, -1.0)), npts))


def merge_by_resolution(eigenvalues, n0, c=1.0):
    """Drop eigenvalues the degree-d polynomial cannot resolve.  A degree-d odd
    polynomial has n0 = (d+1)/2 terms and resolves no finer than pi/(2 n0) in
    theta = arccos(lambda).  c = 0 keeps every distinct value."""
    lam = np.sort(np.asarray(eigenvalues, float))
    if len(lam) <= 1:
        return lam, len(lam)
    th = np.arccos(np.clip(lam, -1.0, 1.0))
    lim = c * np.pi / (2.0 * n0)
    keep = [0]
    for i in range(1, len(lam)):
        if abs(th[i] - th[keep[-1]]) > lim:
            keep.append(i)
    return lam[keep], len(keep)


# certify() is imported from PolynomialApproximators so that one definition, with
# the corrected Ehlich-Zeller constant, serves both modules.


def apply_correction(p0, dc):
    coef = p0.coef.copy()
    coef[1::2] += dc
    return Chebyshev(coef)


# ---------------------------------------------------------------- the LP
def _solve_lp(p0, lam, n0, region, ngrid, gamma_max=None, tau0=None,
              deriv_windows=None, deriv_L=None):
    lo, hi = region
    x = _cheb_grid(lo, hi, ngrid)
    M = _resid_basis(x, n0)
    r0 = x * p0(x) - 1.0
    m = len(x)
    A_ub = [np.hstack([M, -np.ones((m, 1))]), np.hstack([-M, -np.ones((m, 1))])]
    b_ub = [-r0, r0]

    if gamma_max is not None:                      # |p_SC| <= gamma_max * tau0 on [0,1]
        xg = np.linspace(0.0, 1.0, max(2000, 2 * n0))
        th = np.arccos(np.clip(xg, -1 + 1e-14, 1 - 1e-14))
        P = np.cos(np.outer(th, 2 * np.arange(n0) + 1))
        pv = p0(xg); g = len(xg)
        A_ub += [np.hstack([P, np.zeros((g, 1))]), np.hstack([-P, np.zeros((g, 1))])]
        b_ub += [gamma_max * tau0 - pv, gamma_max * tau0 + pv]

    if deriv_windows is not None and deriv_L is not None:
        xs = np.concatenate([np.linspace(l, h, 201) for l, h in deriv_windows])
        D = _dresid_basis(xs, n0)
        d0 = (Chebyshev(p0.coef) * Chebyshev([0.0, 1.0])).deriv()(xs)
        q = len(xs)
        A_ub += [np.hstack([D, np.zeros((q, 1))]), np.hstack([-D, np.zeros((q, 1))])]
        b_ub += [deriv_L - d0, deriv_L + d0]

    A_eq = np.hstack([_resid_basis(lam, n0), np.zeros((len(lam), 1))])
    b_eq = -(lam * p0(lam) - 1.0)
    c = np.zeros(n0 + 1); c[-1] = 1.0
    res = linprog(c, A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub),
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=[(None, None)] * n0 + [(0.0, None)], method="highs")
    return (res.x[:n0], float(res.x[-1])) if res.success else (None, None)


def spectral_constraint(poly_class, kappa, eigenvalues, epsilon=0.2,
                        mu=MU_DEGREE, c=C_MERGE, degree=None):
    """THE PROCEDURE OF THE PAPER.  Spectrally constrained polynomial.

    1. Base degree and guarantee: d0 = d(epsilon); t0 = certificate of p0 at d0.
    2. Degree: d = mu * d0 (odd), unless `degree` is given.
    3. Merge the supplied eigenvalues at scale c * pi/(2 n0), n0 = (d+1)/2.
    4. Solve   min t  s.t.  |r(x)| <= t on a 3(d+1)-point grid of [a,1],
                            r(lam_k) = 0 at the retained eigenvalues,
       with r(x) = x p(x) - 1 and p = p_base(d) + sum_j dc_j T_{2j+1}.
    5. Certify p on [a,1].  certified = (t <= t0).  If not, the caller raises d.

    kappa is that of the NORMALISED operator (a = 1/kappa) and must not be smaller
    than the true one, or [a,1] does not contain the spectrum.  The supplied
    eigenvalues may be approximate: errors cost accuracy, never the certificate.

    Returns a dict: p, p0 (base at d), d, d0, mu, a, K, K_eff, t, t0, certified,
    t_lp (LP optimum, a lower bound, never quote it), gamma = tau(p)/tau(p0),
    R_ret (residual at retained eigenvalues), dc.
    """
    a = 1.0 / kappa
    d0 = int(poly_class.mindegree(epsilon, a))
    t0 = certify(poly_class.poly(d0, a), [(a, 1.0)], d0 + 1)
    d = (int(round(mu * d0)) | 1) if degree is None else (int(degree) | 1)

    p0 = poly_class.poly(d, a)
    n0 = len(p0.coef[1::2])
    lam_in = np.clip(np.asarray(eigenvalues, float), a * (1 + 1e-9), 1 - 1e-12)
    lam, keff = merge_by_resolution(lam_in, n0, c)

    dc, t_lp = _solve_lp(p0, lam, n0, (a, 1.0), 3 * (d + 1))
    if dc is None:
        raise RuntimeError(f"spectral_constraint: LP infeasible at d = {d}")
    p = apply_correction(p0, dc)
    t = certify(p, [(a, 1.0)], d + 1)
    return dict(p=p, p0=p0, d=d, d0=d0, mu=d / d0, a=a,
                K=int(len(lam_in)), K_eff=int(keff),
                t=float(t), t0=float(t0), certified=bool(t <= t0),
                t_lp=float(t_lp), gamma=_tau(p) / _tau(p0),
                R_ret=float(np.max(np.abs(lam * p(lam) - 1.0))), dc=dc)


def constrained_minimax(poly_class, kappa, lam_hat, degree, variant="full",
                        mu=None, ngrid=None, c_grid=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
                        gamma_max=None, deriv_delta=None, verbose=False):
    """Build the constrained-min-max corrected polynomial.

    poly_class  base family with .poly(d,a) and .mindegree(eps,a)  (e.g. ChebIterPolynomial)
    kappa       condition number of the NORMALISED operator; a = 1/kappa.  Must come
                from the real operator: a homogeneous surrogate got kappa wrong by the
                material contrast (116 -> 412 at contrast 4) and would size the degree
                by that factor.
    lam_hat     the K supplied eigenvalues, normalised into (a, 1).  Approximate is fine
                for variant="full" (costs benefit, not safety).
    degree      d.  Odd.  "full" typically needs ~1.5x, "tail" ~1.12x, the base d(eps).
    variant     "full" (recommended) or "tail" (needs mu and the K-smallest assumption).
    deriv_delta if set, also bound |(x p)'| on +/- deriv_delta windows around each
                supplied point, so the MVT gives |r| <= L*delta at a true eigenvalue
                within delta.  UNTESTED.  Windows only -- bounding the derivative on the
                whole free region triggers the minimax obstruction and kills the gain.

    Returns a dict with 'p' (Chebyshev), 't' (certified bound on the region),
    'R_all' (residual over ALL supplied eigenvalues), 'guarantee' = max(t, R_all)
    -- QUOTE THIS ONE -- 't_lp' (the LP optimum, a LOWER bound: never quote it),
    'c', 'K_eff', 'gamma', 'region', 'dc'.  None if no merge scale is feasible.
    """
    a = 1.0 / kappa
    if degree % 2 == 0:
        degree += 1
    p0 = poly_class.poly(degree, a)
    n0 = len(p0.coef[1::2])
    xs = np.linspace(-1.0, 1.0, 60001)
    tau0 = float(np.max(np.abs(p0(xs))))

    if variant == "full":
        region = (a, 1.0)
    elif variant == "tail":
        if mu is None:
            mu = float(np.max(lam_hat))
        region = (float(mu), 1.0)
    else:
        raise ValueError("variant must be 'full' or 'tail'")

    if ngrid is None:                    # TRAP 1: the grid must beat the degree
        ngrid = 3 * (degree + 1)
    ngrid = max(ngrid, 3 * (degree + 1))

    segs = [region]
    lam_in = np.clip(np.asarray(lam_hat, float), a * (1 + 1e-9), 1 - 1e-12)
    best = None
    for c in c_grid:                     # TRAP 3: sweep c, keep the BEST, not the first
        lam, keff = merge_by_resolution(lam_in, n0, c)
        if keff < 1:
            continue
        win = None
        if deriv_delta is not None:
            win = [(max(l - deriv_delta, a), min(l + deriv_delta, 1.0)) for l in lam]
        dc, t_lp = _solve_lp(p0, lam, n0, region, ngrid, gamma_max, tau0,
                             win, None if deriv_delta is None else np.inf)
        if dc is None:
            continue
        p = apply_correction(p0, dc)
        t = certify(p, segs, degree + 1)          # TRAP 2: certify, do not trust t_lp
        # TRAP 4: residual over ALL supplied eigenvalues, including any the merge
        # discarded.  Selecting on t alone lets the sweep cheat by merging constraints
        # away: a coarse c gives a smaller t on a polynomial that has abandoned most of
        # the spectrum.  Observed: c=4 kept 5 of 18 and certified 6.6e-9 while the true
        # residual over the spectrum was 1.29.  Select on R_all first.
        R_all = float(np.max(np.abs(lam_in * p(lam_in) - 1.0)))
        guarantee = max(t, R_all)
        if verbose:
            print(f"    c={c:<5} K_eff={keff:<4} t_lp={t_lp:.3e} cert={t:.3e} "
                  f"R_all={R_all:.3e} -> {guarantee:.3e}")
        key = (R_all, t)
        if best is None or key < best["_key"]:
            best = dict(p=p, t=t, t_lp=t_lp, R_all=R_all, guarantee=guarantee,
                        c=float(c), K_eff=int(keff),
                        gamma=float(np.max(np.abs(p(xs)))) / tau0,
                        region=region, dc=dc, degree=degree, tau0=tau0, _key=key)
    if best is not None:
        best.pop("_key")
    return best


def caveats():
    return [
        "One base family (ChebIter/Gribling) tested; Mang and Sunderhauf untested.",
        "No predictive formula for t: the Gribling reading gives a CEILING kappa*lam_K "
        "of which only 10-14% is realised, because the weight W dominates. Every "
        "operator is an experiment.",
        "mu = 1.5 is measured, not derived: the smallest certifying multiplier was "
        "1.23-1.48 on every case tested, and 1.48 at 10% eigenvalue error leaves a "
        "thin margin. There is no analytic upper bound on eps_K(d).",
        "The LP grows with d: about 1 s at d = 400 and 70-100 s at d = 1660.",
        "deriv_delta is implemented but UNTESTED.",
        "variant='tail' fails silently if the supplied set is not the K smallest: it "
        "certified 5.0e-5 with a true residual of 0.66 under 15% eigenvalue error.",
        "Benefit needs load energy in the supplied modes: on 3D Poisson (K = 16 of "
        "4096) a random load is less accurate than the base at the same degree.",
        "Where a good classical preconditioner exists, solve the system classically: "
        "PCG with a homogeneous preconditioner solved the test problems in 5-26 "
        "iterations, and computing the K eigenvalues cost 25-213x that.",
    ]