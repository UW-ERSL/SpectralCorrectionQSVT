"""
paperTables.py -- single reproducible driver for every table in

    "Spectrally Corrected Polynomial Approximation for Quantum Singular
     Value Transformation"

Every table in the manuscript is produced here, from logged parameters, so that
captions cannot drift from the code that generated them.

Usage
-----
    python paperTables.py            # all tables
    python paperTables.py 1 3        # only tables 1 and 3
    python paperTables.py scaling    # degree-scaling verification
    python paperTables.py ext        # spectral extension vs correction

Conventions (stated once, used everywhere)
------------------------------------------
* A and its spectrum are normalised by 1.01 * lambda_max, so lambda_max = 0.9901
  and all singular values lie strictly inside (0, 1) as the block encoding requires.
* kappa is therefore 1 / lambda_min of the NORMALISED matrix.
* tau is the maximum of |p_hat| over the FULL interval [-1, 1], including the
  central gap (-a, a).
* P_succ is  ||p(A)b||^2 / tau^2  -- the probability of the joint post-selection
  (LCU ancilla = 0, QSP ancilla = 0) in the real-part-extracting circuit.  This
  is the quantity in Eq. (2) and it obeys  P_succ <= ||A^{-1}b||^2 / tau^2 <= kappa^2 / tau^2.
* Total query cost is reported as d / sqrt(P_succ), the number of block-encoding
  queries needed under amplitude amplification.
"""
import sys, io, time, contextlib
import numpy as np
from numpy.polynomial import Chebyshev

from PolynomialApproximators import (RemezPolynomial, MangPolynomial,
                                     SunderhaufPolynomial, SpectralPolynomial,
                                     spectral_correction, spectral_extension,
                                     merge_eigenvalues)
from PoissonFunctions import (build_1d_poisson, eigs_1d_poisson,
                              build_2d_poisson, eigs_2d_poisson)
from QSVTSolvers import (StandardQSVT, PureSpectralQSVT,
                         SpectrallyBootstrappedQSVT)

BASE_POLYS = {'remez': RemezPolynomial, 'mang': MangPolynomial,
              'sunderhauf': SunderhaufPolynomial}

OUT = []


def emit(s=""):
    print(s)
    OUT.append(s)


def quiet(fn, *a, **kw):
    """Run fn silently (the solvers are chatty)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def setup_1d(m, load="uniform"):
    A, b = build_1d_poisson(m, function_type=load)
    eigs = eigs_1d_poisson(m)
    scale = 1.01 * eigs.max()
    eigs = eigs / scale
    A = A / scale
    kappa = 1.0 / float(eigs[0])
    un = np.linalg.solve(A, b)
    return dict(A=A, b=b, eigs=eigs, kappa=kappa,
                C_true=float(b @ un), x_cl=un / np.linalg.norm(un),
                norm_Ainvb=float(np.linalg.norm(un)))


def setup_2d(m, load="uniform"):
    A, b = build_2d_poisson(m, function_type=load)
    eigs = np.sort(eigs_2d_poisson(m))
    scale = 1.01 * eigs.max()
    eigs = eigs / scale
    A = A / scale
    kappa = 1.0 / float(eigs[0])
    un = np.linalg.solve(A, b)
    return dict(A=A, b=b, eigs=eigs, kappa=kappa,
                C_true=float(b @ un), x_cl=un / np.linalg.norm(un),
                norm_Ainvb=float(np.linalg.norm(un)))


def metrics(solver, x, P, nr, S):
    C = float(S['b'] @ x) * solver.tau * nr
    return dict(d=solver.degree,
                F=float(np.abs(np.vdot(x, S['x_cl'])) ** 2),
                cerr=abs(C - S['C_true']) / abs(S['C_true']),
                P=P, tau=solver.tau,
                bound_k=(S['kappa'] / solver.tau) ** 2,
                bound_b=(S['norm_Ainvb'] / solver.tau) ** 2,
                cost=solver.degree / np.sqrt(P) if P > 0 else np.inf)


# ==========================================================================
# Table 1 -- pure spectral polynomial
# ==========================================================================
def table1():
    emit("\n" + "=" * 104)
    emit("TABLE 1  Pure spectral polynomial, 1D Poisson (uniform load)")
    emit("P_succ is now ||p(A)b||^2/tau^2; the bound column is kappa^2/tau^2.")
    emit("=" * 104)
    for m in (3, 4):
        S = setup_1d(m)
        emit(f"\n  m = {m}  (N = {2**m}, kappa = {S['kappa']:.4f}, "
             f"||A^-1 b|| = {S['norm_Ainvb']:.3f})")
        emit(f"  {'n_f':>4} {'d':>5} {'tau/kappa':>10} {'P_succ':>9} "
             f"{'kappa^2/tau^2':>14} {'||Ainv b||^2/tau^2':>19} {'F':>10} {'OK':>4}")
        for nf in (2, 3, 4, 5, 8):
            s = quiet(PureSpectralQSVT, S['A'], S['b'], eigenvalues=S['eigs'],
                      kappa=S['kappa'], n_factor=nf)
            x, P, nr = quiet(s.solve)
            M = metrics(s, x, P, nr, S)
            ok = "yes" if P <= M['bound_k'] + 1e-9 else "NO"
            emit(f"  {nf:>4} {M['d']:>5} {s.tau/S['kappa']:>10.3f} {P:>9.4f} "
                 f"{M['bound_k']:>14.4f} {M['bound_b']:>19.4f} {M['F']:>10.6f} {ok:>4}")


# ==========================================================================
# Table 2 -- degree/accuracy trade-off (polynomial level, no QSVT)
# ==========================================================================
def table2():
    emit("\n" + "=" * 104)
    emit("TABLE 2  Degree vs eigenvalue residual, Mang and spectral-Mang")
    emit("=" * 104)
    N = 4
    eigs = eigs_1d_poisson(2)
    eigs = eigs / (1.01 * eigs.max())
    kappa = 1.0 / eigs[0]
    K = 2
    emit(f"  N = {N}, kappa = {kappa:.4f}, K = {K}, "
         f"lambda = {np.array2string(eigs, precision=4)}")
    emit(f"  {'Method':<26} {'eps':>7} {'d':>5} {'Eeig (K corr.)':>16} {'Eeig (all N)':>15}")
    for eps in (0.2, 0.1, 0.01):
        d = MangPolynomial.mindegree(eps, 1.0 / kappa)
        p0 = MangPolynomial.poly(d, 1.0 / kappa)
        e0 = np.max(np.abs(eigs * p0(eigs) - 1))
        emit(f"  {'Mang':<26} {eps:>7} {d:>5} {e0:>16.2e} {e0:>15.2e}")
        cc = spectral_correction(p0, eigs[:K])
        coef = p0.coef.copy(); coef[1::2] += cc
        pSC = Chebyshev(coef)
        eK = np.max(np.abs(eigs[:K] * pSC(eigs[:K]) - 1))
        eA = np.max(np.abs(eigs * pSC(eigs) - 1))
        emit(f"  {'Spectral-Mang (K=2)':<26} {eps:>7} {d:>5} {eK:>16.2e} {eA:>15.2e}")


# ==========================================================================
# Table 3 -- QSVT validation, 1D Poisson
# ==========================================================================
def table3(m=4, base='Mang', eps_loose=0.5, eps_tight=1e-3):
    emit("\n" + "=" * 116)
    emit(f"TABLE 3  QSVT, 1D Poisson, m={m}, base={base}, K=N={2**m}, "
         f"eps_loose={eps_loose}, eps_tight={eps_tight}")
    emit("=" * 116)
    for load in ("uniform", "delta"):
        S = setup_1d(m, load)
        K = 2 ** m
        emit(f"\n  {load} load   (kappa = {S['kappa']:.2f})")
        emit(f"  {'Method':<26} {'d':>5} {'Fidelity':>10} {'RelComplErr':>12} "
             f"{'P_succ':>8} {'bound':>8} {'tau':>8} {'d/sqrt(P)':>10}")
        rows = {}
        specs = [(f"{base} (eps={eps_loose})", 'std', eps_loose),
                 (f"{base} (eps={eps_tight})", 'std', eps_tight),
                 (f"Spectral-{base} (eps={eps_loose})", 'sc', eps_loose)]
        for label, kind, eps in specs:
            if kind == 'std':
                s = quiet(StandardQSVT, S['A'], S['b'], kappa=S['kappa'],
                          target_error=eps, polyMethod=base)
            else:
                s = quiet(SpectrallyBootstrappedQSVT, S['A'], S['b'],
                          lam_K=S['eigs'][:K], kappa=S['kappa'],
                          target_error=eps, polyMethod=base)
            x, P, nr = quiet(s.solve)
            M = metrics(s, x, P, nr, S)
            rows[label] = M
            emit(f"  {label:<26} {M['d']:>5} {M['F']:>10.6f} {M['cerr']:>12.2e} "
                 f"{P:>8.4f} {M['bound_k']:>8.4f} {M['tau']:>8.1f} {M['cost']:>10.1f}")
        t, sc = rows[specs[1][0]], rows[specs[2][0]]
        emit(f"    depth ratio        tight/spectral = {t['d']/sc['d']:.2f}x")
        emit(f"    query-cost ratio   tight/spectral = {t['cost']/sc['cost']:.2f}x")


# ==========================================================================
# Table 4 -- robustness to eigenvalue perturbation
# ==========================================================================
def table4(m=4, base='Mang', eps=0.5, n_trials=10, seed=42):
    emit("\n" + "=" * 104)
    emit(f"TABLE 4  Robustness to eigenvalue perturbation, 1D Poisson, m={m}, "
         f"base={base}, K=N={2**m}, eps={eps}, {n_trials} trials, seed={seed}")
    emit("(same problem, base and eps as Table 3 -- eta=0 row MUST match Table 3)")
    emit("=" * 104)
    S = setup_1d(m, "uniform")
    K = 2 ** m
    rng = np.random.default_rng(seed)
    a = 1.0 / S['kappa']
    emit(f"  {'eta':>8} {'Fidelity':>24} {'RelComplErr':>26} {'P_succ':>9}")
    for eta in (0.0, 1e-2, 1e-1):
        F, C, P = [], [], []
        for _ in range(1 if eta == 0.0 else n_trials):
            lam = S['eigs'][:K] * (1.0 + rng.uniform(-eta, eta, size=K))
            lam = np.clip(lam, a + 1e-10, 1.0 - 1e-10)
            s = quiet(SpectrallyBootstrappedQSVT, S['A'], S['b'], lam_K=lam,
                      kappa=S['kappa'], target_error=eps, polyMethod=base)
            x, p, nr = quiet(s.solve)
            M = metrics(s, x, p, nr, S)
            F.append(M['F']); C.append(M['cerr']); P.append(p)
        fs = (f"{np.mean(F):.6f}" if np.std(F) < 1e-10
              else f"{np.mean(F):.6f} +/- {np.std(F):.1e}")
        cs = (f"{np.mean(C):.2e}" if np.std(C) < 1e-14
              else f"{np.mean(C):.2e} +/- {np.std(C):.1e}")
        emit(f"  {eta:>8.0e} {fs:>24} {cs:>26} {np.mean(P):>9.4f}")


# ==========================================================================
# Table 5 -- 2D Poisson
# ==========================================================================
def table5(m=4, base='Mang', eps=0.2, KRange=(0, 1, 4, 8, 16, 32)):
    emit("\n" + "=" * 104)
    emit(f"TABLE 5  QSVT, 2D Poisson, N1={2**m}, N={4**m}, base={base}, eps={eps}")
    emit("=" * 104)
    S = setup_2d(m)
    peak_cl = float(np.max(np.abs(S['x_cl'])))
    emit(f"  kappa = {S['kappa']:.2f}, classical peak = {peak_cl:.4f}")
    emit(f"  {'K':>4} {'K_eff':>6} {'d':>5} {'Fidelity':>11} {'RelComplErr':>12} "
         f"{'P_succ':>8} {'bound':>8} {'tau':>8} {'peak':>8} {'peak err':>9}")
    for K in KRange:
        t0 = time.time()
        if K == 0:
            s = quiet(StandardQSVT, S['A'], S['b'], kappa=S['kappa'],
                      target_error=eps, polyMethod=base)
            Keff = 0
        else:
            s = quiet(SpectrallyBootstrappedQSVT, S['A'], S['b'],
                      lam_K=S['eigs'][:K], kappa=S['kappa'],
                      target_error=eps, polyMethod=base)
            Keff = s.correction_info['K_eff']
        x, P, nr = quiet(s.solve)
        M = metrics(s, x, P, nr, S)
        pk = float(np.max(np.abs(x)))
        emit(f"  {K:>4} {Keff:>6} {M['d']:>5} {M['F']:>11.7f} {M['cerr']:>12.2e} "
             f"{P:>8.4f} {M['bound_k']:>8.4f} {M['tau']:>8.1f} {pk:>8.4f} "
             f"{100*(pk-peak_cl)/peak_cl:>8.2f}%   [{time.time()-t0:.0f}s]")


# ==========================================================================
# Degree scaling verification -- is it sqrt(kappa) or kappa?
# ==========================================================================
def scaling():
    emit("\n" + "=" * 104)
    emit("DEGREE SCALING  minimum degree d(kappa, eps) for the three base polynomials")
    emit("Fitting log d = c + alpha log kappa at fixed eps.  alpha ~ 1 => Theta(kappa),")
    emit("alpha ~ 0.5 => Theta(sqrt(kappa)).")
    emit("=" * 104)
    kappas = np.array([10, 20, 40, 80, 160, 320], float)
    for eps in (0.1, 0.01):
        emit(f"\n  eps = {eps}")
        emit(f"  {'kappa':>8} " + "".join(f"{n:>14}" for n in
                                          ('Remez d', 'Mang d', 'Sunderhauf d')))
        table = {n: [] for n in ('remez', 'mang', 'sunderhauf')}
        for kp in kappas:
            row = []
            for n in ('remez', 'mang', 'sunderhauf'):
                d = BASE_POLYS[n].mindegree(eps, 1.0 / kp)
                table[n].append(d); row.append(d)
            emit(f"  {kp:>8.0f} " + "".join(f"{v:>14d}" for v in row))
        for n in ('remez', 'mang', 'sunderhauf'):
            al = np.polyfit(np.log(kappas), np.log(table[n]), 1)[0]
            emit(f"    {n:>12}: fitted exponent alpha = {al:.3f}")


# ==========================================================================
# Spectral extension vs spectral correction (small-K regime)
# ==========================================================================
def extension():
    emit("\n" + "=" * 104)
    emit("SPECTRAL EXTENSION vs SPECTRAL CORRECTION -- polynomial level, 2D Poisson spectrum")
    emit("Addresses the non-monotonic small-K behaviour: correction perturbs shared")
    emit("coefficients and can degrade UNcorrected eigenvalues; extension cannot.")
    emit("=" * 104)
    m, eps, base = 4, 0.2, 'mang'
    eigs = np.sort(eigs_2d_poisson(m)); eigs = eigs / (1.01 * eigs.max())
    kappa = 1.0 / eigs[0]
    d0 = BASE_POLYS[base].mindegree(eps, 1.0 / kappa)
    p0 = BASE_POLYS[base].poly(d0, 1.0 / kappa)
    e_base = np.max(np.abs(eigs * p0(eigs) - 1))
    emit(f"  kappa = {kappa:.2f}, base {base} d0 = {d0}, "
         f"base max residual over all {len(eigs)} eigenvalues = {e_base:.3e}")
    def resid(p, lam):
        return float(np.max(np.abs(lam * p(lam) - 1.0)))

    emit(f"  {'K':>4} {'Keff':>5} | {'CORRECTION (d fixed)':>38} | {'EXTENSION (d grows)':>40}")
    emit(f"  {'':>4} {'':>5} | {'d':>5} {'E(targeted)':>12} {'E(full spec)':>13} {'|dc|':>6} | "
         f"{'d':>5} {'E(targeted)':>12} {'E(full spec)':>13} {'cond':>8}")
    for K in (1, 4, 8, 16, 32):
        cc, ic = spectral_correction(p0, eigs[:K], return_info=True)
        coef = p0.coef.copy(); coef[1::2] += cc
        pSC = Chebyshev(coef)
        pEX, ie = spectral_extension(p0, eigs[:K], return_info=True)
        lamT, _ = merge_eigenvalues(eigs[:K])
        emit(f"  {K:>4} {ic['K_eff']:>5} | {d0:>5} {resid(pSC, lamT):>12.2e} "
             f"{resid(pSC, eigs):>13.2e} {ic['corr_norm']:>6.2f} | "
             f"{ie['degree_ext']:>5} {resid(pEX, lamT):>12.2e} "
             f"{resid(pEX, eigs):>13.2e} {ie['cond']:>8.1e}")
    emit("\n  E(targeted)  = max residual at the K_eff targeted eigenvalues")
    emit("  E(full spec) = max residual over ALL 256 eigenvalues (this drives fidelity)")
    emit("  NOTE: the extension system is a square K_eff x K_eff solve in the")
    emit("  HIGHEST-order Chebyshev terms evaluated near x = 1/kappa, where those")
    emit("  terms are nearly linearly dependent -- hence the conditioning blow-up.")


# ==========================================================================
# Why is fidelity non-monotonic in K?  (mode-resolved analysis)
# ==========================================================================
def nonmonotonic():
    emit("\n" + "=" * 110)
    emit("NON-MONOTONIC FIDELITY IN K -- mode-resolved explanation, 2D Poisson")
    emit("=" * 110)
    m, eps, base = 4, 0.2, 'mang'
    S = setup_2d(m)
    A, b = S['A'], S['b']
    w, V = np.linalg.eigh(A)
    beta = V.T @ b                       # modal weights of the load
    kappa = S['kappa']
    d0 = BASE_POLYS[base].mindegree(eps, 1.0 / kappa)
    p0 = BASE_POLYS[base].poly(d0, 1.0 / kappa)
    eigs = S['eigs']

    def predicted_F(p):
        ue = beta / w;      ue = ue / np.linalg.norm(ue)
        uq = p(w) * beta;   uq = uq / np.linalg.norm(uq)
        return float(abs(np.dot(ue, uq)) ** 2)

    # modal energy: fraction of ||A^-1 b||^2 carried by each mode
    energy = (beta / w) ** 2
    energy = energy / energy.sum()
    order = np.argsort(w)
    cum = np.cumsum(energy[order])
    emit(f"  Load energy concentration (uniform load): mode 1 carries "
         f"{100*energy[order][0]:.2f}% of ||A^-1 b||^2;")
    emit(f"  the 10 lowest modes carry {100*cum[9]:.2f}%; the 32 lowest carry "
         f"{100*cum[31]:.2f}%.")
    emit("")
    emit("  Writing r(lam) = lam p(lam) - 1, the QSVT state is the exact solution")
    emit("  modulated mode-by-mode by (1 + r).  A CONSTANT r is a pure rescaling and")
    emit("  costs no fidelity at all; only the SPREAD of r across energy-carrying modes")
    emit("  does.  To second order  1 - F  ~=  energy-weighted variance of r.")
    emit("")
    emit(f"  {'K':>4} {'Keff':>5} {'1-F (pred)':>12} {'1-F (2nd order)':>16} "
         f"{'mean r':>10} {'std r':>10} {'|r|_rms':>10} {'|r|_max':>10}")
    for K in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48):
        if K == 0:
            p = p0; Keff = 0
        else:
            cc, ic = spectral_correction(p0, eigs[:K], return_info=True)
            coef = p0.coef.copy(); coef[1::2] += cc
            p = Chebyshev(coef); Keff = ic['K_eff']
        r = w * p(w) - 1.0
        mu = float(np.sum(energy * r))
        var = float(np.sum(energy * (r - mu) ** 2))
        rms = float(np.sqrt(np.sum(energy * r ** 2)))
        emit(f"  {K:>4} {Keff:>5} {1-predicted_F(p):>12.3e} {var:>16.3e} "
             f"{mu:>10.3e} {np.sqrt(var):>10.3e} {rms:>10.3e} "
             f"{np.abs(r).max():>10.3e}")
    emit("")
    emit("  READING.  The base Mang polynomial has a LARGE but nearly UNIFORM residual")
    emit("  across the energy-carrying modes (mean r ~ -0.19, std r tiny), which is")
    emit("  almost a pure rescaling -- hence its deceptively high fidelity 0.99987")
    emit("  despite a 19% error.  Correcting a single eigenvalue sets r = 0 on the mode")
    emit("  carrying 99% of the energy while leaving r ~ -0.19 on the rest: the MEAN")
    emit("  improves by an order of magnitude but the SPREAD grows, so fidelity DROPS.")
    emit("  Fidelity only recovers once K covers essentially all energy-carrying modes.")
    emit("")
    emit("  CONSEQUENCE FOR THE PAPER: fidelity is a poor headline metric here because")
    emit("  it is blind to the uniform part of the error.  The relative compliance error")
    emit("  (Table 5) is sensitive to it and falls monotonically, 1.9e-1 -> 7.2e-5, a")
    emit("  2700x improvement.  That is the honest statement of what the correction buys.")


# ==========================================================================
def main():
    args = [a.lower() for a in sys.argv[1:]] or ['1', '2', '3', '4', '5',
                                                 'scaling', 'ext', 'nm']
    t0 = time.time()
    for a in args:
        {'1': table1, '2': table2, '3': table3, '4': table4, '5': table5,
         'scaling': scaling, 'ext': extension, 'nm': nonmonotonic}[a]()
    emit(f"\nTotal wall time: {time.time()-t0:.0f}s")
    with open("paperTables_output.txt", "w") as f:
        f.write("\n".join(OUT))
    print("\n[written to paperTables_output.txt]")


if __name__ == "__main__":
    main()
