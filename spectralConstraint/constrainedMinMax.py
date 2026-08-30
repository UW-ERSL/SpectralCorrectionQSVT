"""
constrainedMinimax.py -- constrained min-max construction of QSVT polynomials.

STATUS 2026-08-28: research prototype. Verified on 2D Poisson (N=256..1024) and on a
stiff-inclusion operator, at the polynomial level and on a statevector circuit. NOT
publication-ready; see `caveats()` and `claude/tail-constrained-correction.md`.

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
3. The merge scale c is not optional: perturbed eigenvalues split exact degeneracies
   into unresolvable pairs.  Sweep it and take the BEST certified value, not the first
   that passes (this module does; an earlier version took the first and produced a
   non-monotone eta).
4. Select the merge scale on the residual over ALL supplied eigenvalues, not on t.
   Selecting on t lets the sweep merge constraints away to make t small: c=4 kept 5 of
   18 eigenvalues and certified 6.6e-9 while the true residual over the spectrum was
   1.29.  Report guarantee = max(t, R_all).
"""
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from scipy.optimize import linprog

__all__ = ["constrained_minimax", "certify", "apply_correction",
           "merge_by_resolution", "caveats"]


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


# ---------------------------------------------------------------- certificate
def certify(p, segments, deg, M=9000):
    """RIGOROUS upper bound on max |x p(x) - 1| over a union of closed intervals.

    Ehlich-Zeller: for a polynomial of degree D sampled at M > D cosine-spaced points
    of an interval,  sup <= max_grid / cos(pi D / 2M).  Applied per segment; the union
    bound is the max.  This is a theorem, not a fine grid, and it is the same
    inequality QSVTSolvers uses to bound tau.  Cost ~0.04%; tight to 4 figures.
    """
    if M <= deg:
        M = 4 * deg + 1
    best = 0.0
    y = np.cos(np.arange(M) * np.pi / (M - 1))
    for lo, hi in segments:
        if hi - lo < 1e-14:
            continue
        x = 0.5 * (lo + hi) + 0.5 * (hi - lo) * y
        g = float(np.max(np.abs(x * p(x) - 1.0)))
        best = max(best, g / np.cos(np.pi * deg / (2 * M)))
    return best


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
        "deriv_delta is implemented but UNTESTED.",
        "variant='tail' fails silently if the supplied set is not the K smallest: it "
        "certified 5.0e-5 with a true residual of 0.66 under 15% eigenvalue error.",
        "The open practical question is how much gain survives at a few percent "
        "eigenvalue error. Brackets: 0.1% keeps ~200x, 24% keeps only 10-21x and stops "
        "improving with degree. 4% is untested and decides the case.",
        "Where a good classical preconditioner exists, solve the system classically: "
        "PCG with a homogeneous preconditioner solved the test problems in 5-26 "
        "iterations, and computing the K eigenvalues cost 25-213x that.",
    ]