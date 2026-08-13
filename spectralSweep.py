"""
spectralSweep.py -- how far can the base tolerance be loosened, as a function of
spectral difficulty?

MOTIVATION
----------
The depth ratio d(eps_tight)/d(eps_loose) is independent of kappa and of the
operator: for the optimal polynomial, n = arccosh(1/eps)/arccosh(z0) and the
kappa dependence cancels in the ratio.  So a larger Poisson problem cannot tell
us anything new about the size of the advantage.

What IS operator-dependent is how far eps_loose can be pushed before the
correction inflates the subnormalisation factor tau so badly that the total
query cost goes backwards.  That is the quantity this sweep maps.

DIFFICULTY PARAMETER
--------------------
We parametrise spectra by the spatial dimension of the operator they come from.
By Weyl's law, the eigenvalue counting function of a d-dimensional Laplacian
satisfies  N(lambda) ~ lambda^{d/2},  hence

    lambda_k  ~  k^{2/d},      k = 1, ..., N,      kappa = N^{2/d}.

d = 1 gives a sparse low spectrum, d = 3 a dense one; d is continuous here so
the sweep interpolates.  This is the physically meaningful axis, because the low
end of the spectrum is both where the correction must work and where higher
spatial dimension crowds the eigenvalues together.

COST MODEL
----------
Total block-encoding queries under amplitude amplification are
Q = d / sqrt(P_succ) with P_succ = ||p(A)b||^2 / tau^2.  Once the relevant
eigenvalues are corrected, ||p(A)b|| ~ ||A^{-1}b|| is fixed, so

    Q  proportional to  d * tau,

which is computable from the polynomial alone -- no circuit simulation needed.
The gain over a tight uncorrected reference is therefore

    G(eps) = [ d(eps_tight) * tau(eps_tight) ] / [ d(eps) * tau_SC(eps) ].

We maximise G over eps.  G > 1 means the correction pays; the maximiser is the
principled choice of eps_loose, and max G is the achievable advantage.

Usage:  python spectralSweep.py [quick]
"""
import sys
import numpy as np
from numpy.polynomial import Chebyshev

from PolynomialApproximators import ChebIterPolynomial as CI, spectral_correction

EPS_TIGHT = 1e-3
EPS_GRID = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]


def weyl_spectrum(kappa, dim, n_max=4096):
    """lambda_k ~ k^{2/dim} on [1/kappa, 1], with N = kappa^{dim/2} eigenvalues."""
    a = 1.0 / kappa
    N = int(min(round(kappa ** (dim / 2.0)), n_max))
    N = max(N, 4)
    k = np.arange(1, N + 1, dtype=float)
    lam = a * k ** (2.0 / dim)
    return np.clip(lam, a, 1.0), N


def tau_of(p, ngrid=100001):
    x = np.linspace(-1.0, 1.0, ngrid)
    return float(np.max(np.abs(p(x))))


def evaluate(lam, kappa, eps, K):
    """Return (degree, tau_base, tau_corrected, gram_cond, K_eff, |dc|)."""
    a = 1.0 / kappa
    d = CI.mindegree(eps, a)
    p0 = CI.poly(d, a)
    cc, info = spectral_correction(p0, lam[:K], return_info=True)
    coef = p0.coef.copy()
    coef[1::2] += cc
    pSC = Chebyshev(coef)
    return d, tau_of(p0), tau_of(pSC), info['gram_cond'], info['K_eff'], info['corr_norm']


def best_epsilon(lam, kappa, K, eps_grid=EPS_GRID):
    """Maximise the query-cost gain G(eps) over the grid."""
    a = 1.0 / kappa
    d_t = CI.mindegree(EPS_TIGHT, a)
    tau_t = tau_of(CI.poly(d_t, a))
    Q_tight = d_t * tau_t

    best = None
    curve = []
    for eps in eps_grid:
        try:
            d, t0, t1, gc, keff, dc = evaluate(lam, kappa, eps, K)
        except Exception:
            continue
        G = Q_tight / (d * t1)
        curve.append((eps, d, t1 / t0, G, gc))
        if best is None or G > best[3]:
            best = (eps, d, t1 / t0, G, gc, keff, dc)
    return best, curve, d_t


def main():
    quick = 'quick' in sys.argv
    dims = [1.0, 1.5, 2.0, 2.5, 3.0] if not quick else [1.0, 2.0, 3.0]
    kappas = [117.6, 441.0] if not quick else [117.6]
    Ks = [16, 32] if not quick else [16]

    print("=" * 108)
    print("SPECTRAL DIFFICULTY SWEEP")
    print(f"Weyl spectra lambda_k ~ k^(2/dim);  eps_tight = {EPS_TIGHT};  "
          "gain G = (d_t tau_t)/(d tau_SC)")
    print("Higher dim = denser low spectrum = harder for the correction.")
    print("=" * 108)
    print(f"{'dim':>5} {'kappa':>8} {'N':>6} {'K':>4} {'Keff':>5} | {'best eps':>9} "
          f"{'d':>6} {'d_tight':>8} | {'tau infl':>9} {'gram cond':>11} "
          f"{'|dc|':>9} | {'GAIN':>7}")
    rows = []
    for kappa in kappas:
        for dim in dims:
            lam, N = weyl_spectrum(kappa, dim)
            for K in Ks:
                if K > N:
                    continue
                best, curve, d_t = best_epsilon(lam, kappa, K)
                if best is None:
                    print(f"{dim:>5.1f} {kappa:>8.1f} {N:>6} {K:>4}   ---   FAILED")
                    continue
                eps, d, infl, G, gc, keff, dc = best
                rows.append((dim, kappa, N, K, keff, eps, d, d_t, infl, gc, dc, G))
                print(f"{dim:>5.1f} {kappa:>8.1f} {N:>6} {K:>4} {keff:>5} | {eps:>9} "
                      f"{d:>6} {d_t:>8} | {infl:>9.2f} {gc:>11.2e} "
                      f"{dc:>9.1f} | {G:>7.2f}x")

    print("\n" + "=" * 108)
    print("IS cond(G) A USABLE A PRIORI PREDICTOR OF THE ACHIEVABLE GAIN?")
    print("=" * 108)
    if len(rows) >= 3:
        gc = np.array([r[9] for r in rows])
        gn = np.array([r[11] for r in rows])
        ok = np.isfinite(gc) & np.isfinite(gn) & (gc > 0) & (gn > 0)
        if ok.sum() >= 3:
            c = np.corrcoef(np.log10(gc[ok]), np.log10(gn[ok]))[0, 1]
            sl, ic = np.polyfit(np.log10(gc[ok]), np.log10(gn[ok]), 1)
            print(f"  log10(gain) vs log10(cond G):  r = {c:+.3f}, "
                  f"slope = {sl:+.3f}")
            print(f"  fitted rule:  gain ~ {10**ic:.2f} * cond(G)^({sl:+.3f})")
            print("\n  Predicted vs achieved:")
            print(f"  {'dim':>5} {'K':>4} {'cond G':>11} {'gain':>8} {'predicted':>10}")
            for r in rows:
                pred = 10 ** (ic + sl * np.log10(r[9])) if r[9] > 0 else np.nan
                print(f"  {r[0]:>5.1f} {r[3]:>4} {r[9]:>11.2e} {r[11]:>8.2f} {pred:>10.2f}")

    print("\n" + "=" * 108)
    print("GAIN CURVE vs eps, hardest and easiest case")
    print("=" * 108)
    for dim in (dims[0], dims[-1]):
        lam, N = weyl_spectrum(kappas[0], dim)
        _, curve, _ = best_epsilon(lam, kappas[0], Ks[0])
        print(f"\n  dim = {dim}, kappa = {kappas[0]}, N = {N}, K = {Ks[0]}")
        print(f"  {'eps':>6} {'d':>6} {'tau infl':>10} {'gain':>8}")
        for eps, d, infl, G, _ in curve:
            mark = "  <-- best" if abs(G - max(c[3] for c in curve)) < 1e-12 else ""
            print(f"  {eps:>6} {d:>6} {infl:>10.2f} {G:>8.2f}{mark}")


if __name__ == "__main__":
    main()
