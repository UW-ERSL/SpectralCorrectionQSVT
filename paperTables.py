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
import sys, io, time, gc, contextlib
import numpy as np
from numpy.polynomial import Chebyshev

from PolynomialApproximators import (ChebIterPolynomial, MangPolynomial,
                                     SunderhaufPolynomial, SpectralPolynomial,
                                     spectral_correction, spectral_extension,
                                     merge_eigenvalues, merge_by_resolution,
                                     adaptive_spectral_correction)
from PoissonFunctions import (build_1d_poisson, eigs_1d_poisson,
                              build_2d_poisson, eigs_2d_poisson)
from QSVTSolvers import (StandardQSVT, PureSpectralQSVT,
                         SpectrallyBootstrappedQSVT)

BASE_POLYS = {'chebiter': ChebIterPolynomial, 'mang': MangPolynomial,
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
    xex = S['x_cl'] * S['norm_Ainvb']           # the classical solution, unnormalised
    xq = solver.tau * nr * x                    # the recovered QSVT solution
    return dict(d=solver.degree,
                F=float(np.abs(np.vdot(x, S['x_cl'])) ** 2),
                cerr=abs(C - S['C_true']) / abs(S['C_true']),
                ex=float(np.linalg.norm(xq - xex) / np.linalg.norm(xex)),
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
    emit("REBUTTAL ONLY -- the paper no longer carries this table; the rebuttal")
    emit("uses it to show P_succ = ||A^-1 b||^2/tau^2 for the exact interpolant.")
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
    emit("TABLE 2  Degree vs eigenvalue residual, base p0 (ChebIter) and p_SC")
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
        d = ChebIterPolynomial.mindegree(eps, 1.0 / kappa)
        p0 = ChebIterPolynomial.poly(d, 1.0 / kappa)
        e0 = np.max(np.abs(eigs * p0(eigs) - 1))
        emit(f"  {'p0  (ChebIter)':<26} {eps:>7} {d:>5} {e0:>16.2e} {e0:>15.2e}")
        cc = spectral_correction(p0, eigs[:K])
        coef = p0.coef.copy(); coef[1::2] += cc
        pSC = Chebyshev(coef)
        eK = np.max(np.abs(eigs[:K] * pSC(eigs[:K]) - 1))
        eA = np.max(np.abs(eigs * pSC(eigs) - 1))
        emit(f"  {'p_SC (K=2)':<26} {eps:>7} {d:>5} {eK:>16.2e} {eA:>15.2e}")


# ==========================================================================
# Table 3 -- QSVT validation, 1D Poisson
# ==========================================================================
def table3(m=4, base='ChebIter', eps_loose=0.5, eps_tight=1e-3):
    emit("\n" + "=" * 116)
    emit(f"TABLE 3  QSVT, 1D Poisson, m={m}, base={base}, K=N={2**m}, "
         f"eps_loose={eps_loose}, eps_tight={eps_tight}")
    emit("=" * 116)
    for load in ("uniform", "delta"):
        S = setup_1d(m, load)
        K = 2 ** m
        emit(f"\n  {load} load   (kappa = {S['kappa']:.2f})")
        emit(f"  {'Method':<26} {'d':>5} {'Fidelity':>10} {'e_C':>12} {'e_x':>12} "
             f"{'P_succ':>8} {'bound':>8} {'tau':>8} {'d/sqrt(P)':>10}")
        rows = {}
        specs = [(f"p0   (eps={eps_loose})", 'std', eps_loose),
                 (f"p0   (eps={eps_tight})", 'std', eps_tight),
                 (f"p_SC (eps={eps_loose})", 'sc', eps_loose)]
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
                 f"{M['ex']:>12.2e} {P:>8.4f} {M['bound_k']:>8.4f} {M['tau']:>8.1f} "
                 f"{M['cost']:>10.1f}")
        t, sc = rows[specs[1][0]], rows[specs[2][0]]
        emit(f"    depth ratio        tight/spectral = {t['d']/sc['d']:.2f}x")
        emit(f"    query-cost ratio   tight/spectral = {t['cost']/sc['cost']:.2f}x")


# ==========================================================================
# Table 4 -- robustness to eigenvalue perturbation
# ==========================================================================
def table4(m=4, base='ChebIter', eps=0.5, n_trials=60, seed=42):
    emit("\n" + "=" * 104)
    emit(f"TABLE 4  Robustness to eigenvalue perturbation, 1D Poisson, m={m}, "
         f"base={base}, K=N={2**m}, eps={eps}, {n_trials} trials, seed={seed}")
    emit("(same problem, base and eps as Table 3 -- eta=0 row MUST match Table 3)")
    emit("=" * 104)
    S = setup_1d(m, "uniform")
    K = 2 ** m
    rng = np.random.default_rng(seed)
    a = 1.0 / S['kappa']
    emit(f"  {'eta':>8} {'Fidelity':>24} {'e_C':>26} {'e_x':>26} {'P_succ':>9}"
         f" {'K_eff':>9}")
    for eta in (0.0, 0.01, 0.1, 0.2):
        F, C, P, X, KE = [], [], [], [], []
        for _ in range(1 if eta == 0.0 else n_trials):
            lam = S['eigs'][:K] * (1.0 + rng.uniform(-eta, eta, size=K))
            if eta > 0.0:
                # the clip is a perturbation in its own right; at eta = 0 the
                # row must be the same computation as Table 3.
                lam = np.clip(lam, a + 1e-10, 1.0 - 1e-10)
            s = quiet(SpectrallyBootstrappedQSVT, S['A'], S['b'], lam_K=lam,
                      kappa=S['kappa'], target_error=eps, polyMethod=base)
            x, p, nr = quiet(s.solve)
            M = metrics(s, x, p, nr, S)
            F.append(M['F']); C.append(M['cerr']); P.append(p); X.append(M['ex'])
            KE.append(s.correction_info['K_eff'])
            del s, x
            gc.collect()
        fs = (f"{np.mean(F):.6f}" if np.std(F) < 1e-10
              else f"{np.mean(F):.6f} +/- {np.std(F):.1e}")
        cs = (f"{np.mean(C):.2e}" if np.std(C) < 1e-14
              else f"{np.mean(C):.2e} +/- {np.std(C):.1e}")
        xs = (f"{np.mean(X):.2e}" if np.std(X) < 1e-14
              else f"{np.mean(X):.2e} +/- {np.std(X):.1e}")
        ks = (f"{min(KE)}" if min(KE) == max(KE) else f"{min(KE)}-{max(KE)}")
        emit(f"  {eta:>8} {fs:>24} {cs:>26} {xs:>26} {np.mean(P):>9.4f} {ks:>9}")


# ==========================================================================
# Table 5 -- 2D Poisson
# ==========================================================================
def table5(m=4, base='ChebIter', eps=0.2, KRange=(0, 1, 4, 8, 16, 32)):
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
        gc.collect()          # aer holds the previous circuit otherwise
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
        del s, x



# ==========================================================================
# Degree scaling verification -- is it sqrt(kappa) or kappa?
# ==========================================================================
def scaling():
    emit("\n" + "=" * 104)
    emit("DEGREE SCALING  minimum degree d(kappa, eps) -- REBUTTAL ONLY (R2 Technical 4).")
    emit("The paper uses the single base p0 = ChebIter; the other two families appear")
    emit("here only to show the linear kappa-dependence is family-independent.")
    emit("Fitting log d = c + alpha log kappa at fixed eps.  alpha ~ 1 => Theta(kappa),")
    emit("alpha ~ 0.5 => Theta(sqrt(kappa)).")
    emit("=" * 104)
    kappas = np.array([10, 20, 40, 80, 160, 320], float)
    for eps in (0.1, 0.01):
        emit(f"\n  eps = {eps}")
        emit(f"  {'kappa':>8} " + "".join(f"{n:>14}" for n in
                                          ('ChebIter d', 'Mang d', 'Sunderhauf d')))
        table = {n: [] for n in ('chebiter', 'mang', 'sunderhauf')}
        for kp in kappas:
            row = []
            for n in ('chebiter', 'mang', 'sunderhauf'):
                d = BASE_POLYS[n].mindegree(eps, 1.0 / kp)
                table[n].append(d); row.append(d)
            emit(f"  {kp:>8.0f} " + "".join(f"{v:>14d}" for v in row))
        for n in ('chebiter', 'mang', 'sunderhauf'):
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
    m, eps, base = 4, 0.2, 'chebiter'
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
    emit("REBUTTAL ONLY -- the paper states the mechanism qualitatively in Sec. 5.3.8;")
    emit("the Var_w(r) vs 1-F match below is the evidence, carried in the rebuttal.")
    emit("=" * 110)
    m, eps, base = 4, 0.2, 'chebiter'
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
    r0 = w * p0(w) - 1.0
    mu0 = float(np.sum(energy * r0))
    emit(f"  READING.  The base polynomial p0 has a LARGE but nearly UNIFORM residual")
    emit(f"  across the energy-carrying modes (mean r = {mu0:.3f}, std r "
         f"{np.sqrt(np.sum(energy*(r0-mu0)**2)):.1e}), which is almost a pure")
    emit(f"  rescaling -- hence its deceptively high fidelity {predicted_F(p0):.5f} despite a")
    emit(f"  {100*abs(mu0):.0f}% error.  Correcting a single eigenvalue sets r = 0 on the mode")
    emit(f"  carrying 99% of the energy while leaving r ~ {mu0:.2f} on the rest: the MEAN")
    emit("  improves by orders of magnitude but the SPREAD grows, so fidelity DROPS.")
    emit("  Fidelity only recovers once K covers essentially all energy-carrying modes.")
    emit("")
    emit("  CONSEQUENCE FOR THE PAPER: fidelity is a poor headline metric here because")
    emit("  it is blind to the uniform part of the error.  The relative compliance error")
    emit("  (Table 5) is sensitive to it; it is NOT monotone in K, and the useful")
    emit("  statement is the improvement at the K the adaptive algorithm selects.")


# ==========================================================================
# TABLE 8 -- output of the two-stage algorithm on spectra of growing density
# ==========================================================================
def adaptive():
    from spectralSweep import weyl_spectrum
    emit("\n" + "=" * 104)
    emit("TABLE 8  Output of the two-stage algorithm (Sec. 4.3), gamma_max = 2.")
    emit("eps and d are OUTPUTS of the sweep, not inputs.")
    emit("=" * 104)
    specs = []
    e = eigs_1d_poisson(4); e = e / (1.01 * e.max())
    specs.append(("1D", 1.0 / e[0], e, 16))
    e2 = np.sort(eigs_2d_poisson(4)); e2 = e2 / (1.01 * e2.max())
    specs.append(("2D", 1.0 / e2[0], e2, 32))
    kap = 1.0 / e[0]
    # 1D and 2D Poisson realize lam_k ~ k^2 and ~ k; the k^{2/3} regime of a
    # three-dimensional Laplacian has no small test operator, so it is synthetic.
    lam, _ = weyl_spectrum(kap, 3)
    specs.append(("Weyl 3", kap, lam, 16))
    emit(f"  {'spectrum':<9}{'K':>4}{'eps':>8}{'d':>6}{'Keff':>6}"
         f"{'R_K':>11}{'gamma':>7}{'G':>7}")
    for tag, kappa, lam, K in specs:
        r = quiet(adaptive_spectral_correction, ChebIterPolynomial, kappa, lam[:K])
        emit(f"  {tag:<9}{K:>4}{r['epsilon']:>8}{r['d']:>6}{r['K_eff']:>6}"
             f"{r['R_all']:>11.1e}{r['gamma']:>7.2f}{r['G']:>7.1f}")
    emit("\n  As the spectrum crowds, the sweep tightens eps and raises d; every")
    emit("  supplied eigenvalue is retained (2D: K_eff < K from exact degeneracy).")


# ==========================================================================
# TABLE 9 -- the role of the load:  rho = sqrt(E_out) * r_out
# ==========================================================================
def loads():
    emit("\n" + "=" * 104)
    emit("TABLE 9  Relative solution error on the 2D Poisson operator, three loads.")
    emit("REBUTTAL ONLY -- the paper carries the point-load result as two sentences in")
    emit("Sec. 6.1; the full table supports the response on when the method helps.")
    emit("Sec. 4.3 reads only the spectrum, so eps, d and G are identical in every row.")
    emit("=" * 104)
    emit(f"  {'load':<9}{'K':>4}{'eps':>7}{'d':>6}{'Keff':>6}{'G':>7}"
         f"{'E_out':>11}{'r_out':>8}{'rho_0':>11}{'rho_SC':>11}"
         f"{'rho_0 sim':>11}{'rho_SC sim':>11}")
    for load, Ks in (("uniform", (32,)), ("random", (32,)), ("delta", (16, 32))):
        np.random.seed(0)
        S = setup_2d(4, load)
        A, b = S['A'], S['b']
        w, V = np.linalg.eigh(A)
        beta = V.T @ b
        en = (beta / w) ** 2
        en = en / en.sum()
        xex = np.linalg.solve(A, b)
        for K in Ks:
            r = quiet(adaptive_spectral_correction, ChebIterPolynomial,
                      S['kappa'], S['eigs'][:K])
            p0 = ChebIterPolynomial.poly(r['d'], 1.0 / S['kappa'])
            coef = p0.coef.copy(); coef[1::2] += r['dc']
            pSC = Chebyshev(coef)
            n0 = len(p0.coef[1::2])
            lam_ret, _ = quiet(merge_by_resolution, S['eigs'][:K], n0, r['c'])
            res = w * pSC(w) - 1.0
            pin = np.zeros(len(w), bool)
            for lv in lam_ret:
                pin |= np.isclose(w, lv, rtol=0, atol=1e-12)
            Eo = float(en[~pin].sum())
            ro = float(np.sqrt(np.sum(en[~pin] * res[~pin] ** 2) / Eo))
            rho = float(np.sqrt(np.sum(en * res ** 2)))
            rho0 = float(np.sqrt(np.sum(en * (w * p0(w) - 1) ** 2)))
            # cross-check rho against the statevector simulation
            gc.collect()
            sb = quiet(StandardQSVT, A, b, kappa=S['kappa'],
                       target_error=r['epsilon'], polyMethod='ChebIter')
            xb, Pb, _ = quiet(sb.solve)
            rho0_sim = float(np.linalg.norm(sb.tau*np.sqrt(Pb)*xb - xex)
                             / np.linalg.norm(xex))
            del sb, xb; gc.collect()
            ss = quiet(SpectrallyBootstrappedQSVT, A, b, lam_K=S['eigs'][:K],
                       kappa=S['kappa'], target_error=r['epsilon'],
                       polyMethod='ChebIter')
            xs_, Ps, _ = quiet(ss.solve)
            rho_sim = float(np.linalg.norm(ss.tau*np.sqrt(Ps)*xs_ - xex)
                            / np.linalg.norm(xex))
            del ss, xs_; gc.collect()
            emit(f"  {load:<9}{K:>4}{r['epsilon']:>7}{r['d']:>6}{r['K_eff']:>6}"
                 f"{r['G']:>7.1f}{Eo:>11.1e}{ro:>8.2f}{rho0:>11.2e}{rho:>11.2e}"
                 f"{rho0_sim:>11.2e}{rho_sim:>11.2e}")
    emit("\n  rho = sqrt(E_out) * r_out exactly.  The point load spreads energy across")
    emit("  the spectrum: at K = 32 the correction raises rho above the base.")


# ==========================================================================
# TABLE 6 -- 2D cost at matched solution accuracy
# ==========================================================================
def cost2d(m=4, eps=0.2, K=16):
    """What the base costs to reach the accuracy the correction reaches.

    Evaluated from the polynomials: P_succ = ||p(A)b||^2/tau^2 needs no
    circuit, and the simulated and polynomial-level values agree to seven
    significant figures (see the loads() target).
    """
    emit("\n" + "=" * 104)
    emit(f"TABLE 6  2D Poisson cost at MATCHED solution accuracy, eps={eps}, K={K}")
    emit("=" * 104)
    S = setup_2d(m)
    A, b = S['A'], S['b']
    a = 1.0 / S['kappa']
    w, V = np.linalg.eigh(A)
    beta = V.T @ b
    xex = np.linalg.solve(A, b)
    nx = float(np.linalg.norm(xex))
    xs = np.linspace(-1.0, 1.0, 200001)

    def cost(p, d):
        tau = float(np.max(np.abs(p(xs))))
        y = V @ (p(w) * beta)
        P = float(np.linalg.norm(y) ** 2 / tau ** 2)
        return d, tau, P, d / np.sqrt(P), float(np.linalg.norm(y - xex) / nx)

    d0 = ChebIterPolynomial.mindegree(eps, a)
    p0 = ChebIterPolynomial.poly(d0, a)
    n0 = len(p0.coef[1::2])
    lam, keff = quiet(merge_by_resolution, S['eigs'][:K], n0, 0.0)
    dc, _ = quiet(spectral_correction, p0, lam, return_info=True)
    coef = p0.coef.copy(); coef[1::2] += dc
    pSC = Chebyshev(coef)
    sc = cost(pSC, d0)

    d = d0                       # smallest base degree matching sc's rho
    while d < 4001:
        bm = cost(ChebIterPolynomial.poly(d, a), d)
        if bm[4] <= sc[4]:
            break
        d += 2

    emit(f"  {'method':<28}{'d':>6}{'tau':>9}{'P_succ':>9}{'Q':>10}{'rho':>12}")
    emit(f"  {'p_0  (eps=%.2g)'%eps:<28}{cost(p0,d0)[0]:>6}{cost(p0,d0)[1]:>9.1f}"
         f"{cost(p0,d0)[2]:>9.4f}{cost(p0,d0)[3]:>10.1f}{cost(p0,d0)[4]:>12.3e}")
    emit(f"  {'p_SC (eps=%.2g, K=%d)'%(eps,K):<28}{sc[0]:>6}{sc[1]:>9.1f}{sc[2]:>9.4f}"
         f"{sc[3]:>10.1f}{sc[4]:>12.3e}")
    emit(f"  {'p_0  (matched accuracy)':<28}{bm[0]:>6}{bm[1]:>9.1f}{bm[2]:>9.4f}"
         f"{bm[3]:>10.1f}{bm[4]:>12.3e}")
    emit(f"\n  K_eff = {keff};  depth ratio = {bm[0]/sc[0]:.2f}x;  "
         f"query-cost ratio = {bm[3]/sc[3]:.2f}x")


# ==========================================================================
def main():
    args = [a.lower() for a in sys.argv[1:]] or ['2', '3', '4', '5', 'adapt',
                                                 'cost2d', 'loads', 'nm',
                                                 '1', 'scaling', 'ext']
    t0 = time.time()
    for a in args:
        {'1': table1, '2': table2, '3': table3, '4': table4, '5': table5,
         'adapt': adaptive, 'loads': loads, 'cost2d': cost2d,
         'scaling': scaling, 'ext': extension, 'nm': nonmonotonic}[a]()
    emit(f"\nTotal wall time: {time.time()-t0:.0f}s")
    with open("paperTables_output.txt", "w") as f:
        f.write("\n".join(OUT))
    print("\n[written to paperTables_output.txt]")


if __name__ == "__main__":
    main()