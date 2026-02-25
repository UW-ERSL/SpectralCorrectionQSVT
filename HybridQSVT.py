"""
qsvt_perturbation_experiment.py
================================
Robustness experiment: how does the hybrid correction degrade when
eigenvalues are known only approximately?

Perturbed eigenvalues: lambda_hat_k = lambda_k * (1 + delta_k)
                       delta_k ~ U(-eta, eta)

For each noise level eta, we:
  1. Perturb the K smallest eigenvalues
  2. Apply hybrid correction using perturbed eigenvalues
  3. Run QSVT statevector simulation
  4. Report fidelity and compliance error vs exact result

The experiment is repeated n_trials times per eta to account for
randomness in the perturbation, reporting mean +/- std.

Problem: 1D Poisson FD, N=2^m, statevector simulation.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from QSVTPoisson1D import myQSVT, build_1D_problem,eigs_1d_poisson
from PolynomialApproximators import hybrid_correction
from numpy.polynomial.chebyshev import Chebyshev
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
from PolynomialApproximators import MangPolynomial



class myQSVT_Hybrid(myQSVT):
    """
    Extends myQSVT to apply the min-norm hybrid correction before
    computing QSP phase angles.  All other pipeline steps are unchanged.

    Parameters
    ----------
    lam_K : np.ndarray
        K known eigenvalues (singular values of A, already normalised
        to lie in (0, 1]) to be corrected to machine precision.
    rcond : float
        SVD truncation threshold for the Gram system (default 1e-10).
    """

    def __init__(self, A, b, lam_K, rcond=1e-10, **kwargs):
        self.lam_K = np.asarray(lam_K)
        self.rcond = rcond
        super().__init__(A, b, **kwargs)

    def _get_inverse_phases(self, kappa, target_error=None):
        """
        Override: build base polynomial, apply hybrid correction,
        then proceed with normalisation and QSP phase computation.
        """
        from pyqsp.angle_sequence import QuantumSignalProcessingPhases

        a      = 1.0 / kappa
        degree = self.polyMethod.mindegree(target_error, a)
        self.degree = degree

        # ── base polynomial ───────────────────────────────────────────
        p0 = self.polyMethod.poly(degree, a)

        # ── hybrid correction ─────────────────────────────────────────
        c_corr        = hybrid_correction(p0, self.lam_K, rcond=self.rcond)
        coef_H        = p0.coef.copy()
        coef_H[1::2] += c_corr
        poly          = Chebyshev(coef_H)

        # ── normalisation (identical to base class) ───────────────────
        N_sample        = 25 * degree
        x_s             = np.linspace(-1, 1, N_sample)
        M               = (np.max(np.abs(poly(x_s)))
                           / np.cos(np.pi * degree / (2 * N_sample)))
        tau             = M
        poly_normalised = Chebyshev(poly.coef / M)

        max_val = np.max(np.abs(poly_normalised(np.linspace(-1, 1, 2000))))
        if max_val > 0.999:
            scale           = 0.999 / max_val
            poly_normalised = Chebyshev(poly_normalised.coef * scale)
            tau            /= scale

        phases = QuantumSignalProcessingPhases(poly_normalised,
                                               signal_operator="Wx")
        return [float(phi) for phi in phases], tau, None


def run_hybrid_experiment(m: int,
                          eps_loose: float = 0.1,
                          eps_tight: float = 0.001,
                          K_fraction: float = 0.5,
                          function_type: str = "uniform"):
    """
    Run the three-way QSVT comparison for the 1D Poisson problem.

    Parameters
    ----------
    m             : log2(N) — number of qubits for the data register
    eps_loose     : loose tolerance for base and hybrid methods
    eps_tight     : tight tolerance reference method
    K_fraction    : fraction of eigenvalues to correct (default 0.5 = N/2)
    function_type : 'uniform', 'sine', 'delta', or 'random'
    """
    N = 2**m
    K = max(1, int(round(K_fraction * N)))

    print(f"\n{'='*65}")
    print(f"1D Poisson QSVT Hybrid Experiment")
    print(f"m={m}, N={N}, K={K} (={K_fraction:.0%} of N)")
    print(f"eps_loose={eps_loose}, eps_tight={eps_tight}")
    print(f"function_type={function_type}")
    print(f"{'='*65}")

    # ── build problem ─────────────────────────────────────────────────
    A, b   = build_1D_problem(m, function_type=function_type)
    eigs = eigs_1d_poisson(m)[::-1]
    print("Exact eigenvalues:", eigs)
    eigs   = np.linalg.svd(A, compute_uv=False)   # singular values, descending
    print("SVD: ",eigs)
    
    kappa  = 1.0 / float(eigs[-1])
    lam_K  = eigs[-K:][::-1]                       # K smallest, ascending

    u_cl_unnorm = np.linalg.solve(A, b)
    C_true      = float(b @ u_cl_unnorm)
    x_cl        = u_cl_unnorm / np.linalg.norm(u_cl_unnorm)

    print(f"kappa={kappa:.3f}, lam_min={eigs[-1]:.6f}, lam_max={eigs[0]:.6f}")
    print(f"K smallest eigenvalues: {lam_K}")
    print(f"C_true = {C_true:.6f}")

    results = {}

    # ── (1) Mang base, loose epsilon ──────────────────────────────────
    label = f"Mang (eps={eps_loose})"
    print(f"\n--- {label} ---")
    t0 = time.time()
    solver = myQSVT(A, b, kappa=kappa, target_error=eps_loose,
                    polyMethod='Mang')
    x_qsvt, succ_prob, norm_real = solver.solve(stateVector=True)
    elapsed = time.time() - t0
    results[label] = _collect(x_qsvt, x_cl, succ_prob, norm_real,
                               solver, b, C_true, elapsed)

    # ── (2) Mang base, tight epsilon ──────────────────────────────────
    label = f"Mang (eps={eps_tight})"
    print(f"\n--- {label} ---")
    t0 = time.time()
    solver = myQSVT(A, b, kappa=kappa, target_error=eps_tight,
                    polyMethod='Mang')
    x_qsvt, succ_prob, norm_real = solver.solve(stateVector=True)
    elapsed = time.time() - t0
    results[label] = _collect(x_qsvt, x_cl, succ_prob, norm_real,
                               solver, b, C_true, elapsed)

    # ── (3) Hybrid-Mang, loose epsilon, K=N/2 ────────────────────────
    label = f"Hybrid-Mang K={K} (eps={eps_loose})"
    print(f"\n--- {label} ---")
    t0 = time.time()
    solver = myQSVT_Hybrid(A, b, lam_K=lam_K, kappa=kappa,
                           target_error=eps_loose, polyMethod='Mang')
    x_qsvt, succ_prob, norm_real = solver.solve(stateVector=True)
    elapsed = time.time() - t0
    results[label] = _collect(x_qsvt, x_cl, succ_prob, norm_real,
                               solver, b, C_true, elapsed)

    # ── summary ───────────────────────────────────────────────────────
    _print_experiment_summary(results, kappa, K, eps_loose, eps_tight)
    return results, x_cl


def _collect(x_qsvt, x_cl, succ_prob, norm_real, solver, b, C_true, elapsed):
    C_qsvt        = float(b @ x_qsvt) * solver.tau * norm_real
    compliance_err = abs(C_qsvt - C_true) / abs(C_true)
    fidelity       = float(np.abs(np.vdot(x_qsvt, x_cl))**2)
    max_err        = np.max(np.abs(x_qsvt - x_cl)) / np.max(np.abs(x_cl))
    return dict(degree=solver.degree, fidelity=fidelity,
                success_prob=succ_prob, tau=solver.tau,
                compliance_err=compliance_err, max_error=max_err,
                C_qsvt=C_qsvt, C_true=C_true, elapsed=elapsed,
                x_qsvt=x_qsvt)


def _print_experiment_summary(results, kappa, K, eps_loose, eps_tight):
    print(f"\n{'='*85}")
    print(f"{'Method':<35} {'d':>5} {'Fidelity':>10} "
          f"{'Rel J err':>10} {'Succ.prob':>10} {'tau':>7}  {'Time':>7}")
    print(f"{'-'*85}")
    for label, r in results.items():
        print(f"{label:<35} {r['degree']:>5d} {r['fidelity']:>10.6f} "
              f"{r['compliance_err']:>10.2e} {r['success_prob']:>10.4f} "
              f"{r['tau']:>7.4f}  {r['elapsed']:>6.1f}s")
    print(f"{'='*85}")

    # highlight the key message
    methods = list(results.keys())
    d_loose  = results[methods[0]]['degree']
    d_tight  = results[methods[1]]['degree']
    d_hybrid = results[methods[2]]['degree']
    F_loose  = results[methods[0]]['fidelity']
    F_tight  = results[methods[1]]['fidelity']
    F_hybrid = results[methods[2]]['fidelity']

    print(f"\nKey result:")
    print(f"  Hybrid-Mang at eps={eps_loose} (d={d_hybrid}) achieves "
          f"fidelity {F_hybrid:.6f}")
    print(f"  vs Mang at eps={eps_tight}   (d={d_tight})  fidelity {F_tight:.6f}")
    print(f"  vs Mang at eps={eps_loose}   (d={d_loose})  fidelity {F_loose:.6f}")
    print(f"  Circuit depth reduction: {d_tight}/{d_hybrid} = {d_tight/d_hybrid:.2f}x")


# ── single trial ──────────────────────────────────────────────────────────────

def run_single_trial(A, b, kappa, lam_K_exact, x_cl, C_true,
                     eta, eps, rng):
    """
    Run one QSVT trial with eigenvalues perturbed at level eta.

    Returns
    -------
    fidelity, compliance_err, success_prob
    """
    # perturb eigenvalues
    delta      = rng.uniform(-eta, eta, size=len(lam_K_exact))
    lam_K_pert = lam_K_exact * (1.0 + delta)

    # clip to valid range [a, 1]
    a          = 1.0 / kappa
    lam_K_pert = np.clip(lam_K_pert, a + 1e-10, 1.0 - 1e-10)

    try:
        solver = myQSVT_Hybrid(A, b, lam_K=lam_K_pert, kappa=kappa,
                               target_error=eps, polyMethod='Mang')
        x_qsvt, succ_prob, norm_real = solver.solve(stateVector=True)

        C_qsvt         = float(b @ x_qsvt) * solver.tau * norm_real
        compliance_err = abs(C_qsvt - C_true) / abs(C_true)
        fidelity       = float(np.abs(np.vdot(x_qsvt, x_cl))**2)
        return fidelity, compliance_err, succ_prob

    except Exception as e:
        print(f"    Trial failed: {e}")
        return None, None, None


# ── perturbation sweep ────────────────────────────────────────────────────────

def run_perturbation_experiment(m: int = 4,
                                eps: float = 0.2,
                                K_fraction: float = 0.5,
                                eta_values: list = None,
                                n_trials: int = 10,
                                seed: int = 42,
                                function_type: str = "uniform",
                                save_path: str = None):
    """
    Sweep over perturbation levels eta and report mean/std of
    fidelity and compliance error over n_trials random perturbations.

    Parameters
    ----------
    m             : log2(N)
    eps           : base polynomial tolerance
    K_fraction    : fraction of eigenvalues to correct
    eta_values    : list of perturbation levels (default logspace -4 to -1)
    n_trials      : number of random trials per eta
    seed          : random seed for reproducibility
    function_type : RHS type
    save_path     : if given, save figure to this path
    """
    if eta_values is None:
        eta_values = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]

    N = 2**m
    K = max(1, int(round(K_fraction * N)))
    rng = np.random.default_rng(seed)

    print(f"\n{'='*65}")
    print(f"Perturbation robustness experiment")
    print(f"m={m}, N={N}, K={K}, eps={eps}, n_trials={n_trials}")
    print(f"eta_values={eta_values}")
    print(f"{'='*65}")

    # ── build problem ─────────────────────────────────────────────────
    A, b         = build_1D_problem(m, function_type=function_type)
    eigs         = np.linalg.svd(A, compute_uv=False)   # descending
    kappa        = 1.0 / float(eigs[-1])
    lam_K_exact  = eigs[-K:][::-1]                      # K smallest, ascending

    u_cl_unnorm  = np.linalg.solve(A, b)
    C_true       = float(b @ u_cl_unnorm)
    x_cl         = u_cl_unnorm / np.linalg.norm(u_cl_unnorm)

    print(f"kappa={kappa:.3f}, K smallest eigenvalues: {lam_K_exact}")

    # ── reference: exact eigenvalues (eta=0) ─────────────────────────
    results = {}   # eta -> {fidelity: [...], compliance_err: [...]}

    for eta in eta_values:
        print(f"\neta={eta:.0e}  ({n_trials} trials)")
        fids, c_errs, probs = [], [], []

        n_rep = 1 if eta == 0.0 else n_trials
        for trial in range(n_rep):
            F, ce, sp = run_single_trial(
                A, b, kappa, lam_K_exact, x_cl, C_true, eta, eps, rng)
            if F is not None:
                fids.append(F)
                c_errs.append(ce)
                probs.append(sp)
                print(f"  trial {trial+1:2d}: F={F:.6f}  "
                      f"compliance_err={ce:.2e}  P_succ={sp:.4f}")

        results[eta] = dict(
            fidelity_mean  = np.mean(fids),
            fidelity_std   = np.std(fids),
            compliance_mean= np.mean(c_errs),
            compliance_std = np.std(c_errs),
            prob_mean      = np.mean(probs),
        )

    # ── print summary table ───────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"{'eta':>8}  {'Fidelity (mean±std)':>22}  "
          f"{'Rel J err (mean±std)':>24}  {'P_succ':>8}")
    print(f"{'-'*75}")
    for eta, r in results.items():
        fid_s = (f"{r['fidelity_mean']:.6f}"
                 if r['fidelity_std'] < 1e-10
                 else f"{r['fidelity_mean']:.6f} ± {r['fidelity_std']:.2e}")
        cer_s = (f"{r['compliance_mean']:.2e}"
                 if r['compliance_std'] < 1e-14
                 else f"{r['compliance_mean']:.2e} ± {r['compliance_std']:.2e}")
        print(f"{eta:>8.0e}  {fid_s:>22}  {cer_s:>24}  "
              f"{r['prob_mean']:>8.4f}")
    print(f"{'='*75}")

    # ── plot ──────────────────────────────────────────────────────────
    eta_plot = [e for e in eta_values if e > 0]
    F_mean   = [results[e]['fidelity_mean']   for e in eta_plot]
    F_std    = [results[e]['fidelity_std']    for e in eta_plot]
    C_mean   = [results[e]['compliance_mean'] for e in eta_plot]
    C_std    = [results[e]['compliance_std']  for e in eta_plot]
    F_exact  = results[0.0]['fidelity_mean']
    C_exact  = results[0.0]['compliance_mean']

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # fidelity
    ax = axes[0]
    ax.axhline(F_exact, color='k', lw=1.0, linestyle='--',
               label='Exact eigenvalues')
    ax.errorbar(eta_plot, F_mean, yerr=F_std, fmt='o-', color='C1',
                lw=1.5, ms=5, capsize=4, label='Perturbed eigenvalues')
    ax.set_xscale('log')
    ax.set_xlabel(r'Perturbation level $\eta$', fontsize=14)
    ax.set_ylabel(r'Fidelity $F$', fontsize=14)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    # compliance error
    ax = axes[1]
    ax.axhline(C_exact, color='k', lw=1.0, linestyle='--',
               label='Exact eigenvalues')
    ax.errorbar(eta_plot, C_mean, yerr=C_std, fmt='s-', color='C1',
                lw=1.5, ms=5, capsize=4, label='Perturbed eigenvalues')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Perturbation level $\eta$', fontsize=14)
    ax.set_ylabel(r'Rel.\ compliance error', fontsize=14)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(
        rf'Robustness to eigenvalue perturbations: '
        rf'$N={N}$, $K={K}$, $\varepsilon={eps}$',
        fontsize=11)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return results, fig


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    
    results_uniform, x_cl_uniform = run_hybrid_experiment(
        m=4,
        eps_loose=0.2,
        eps_tight=0.001,
        K_fraction=1.0,
        function_type="uniform"
    )
    
    # ── point load ────────────────────────────────────────────────────
    results_delta, x_cl_delta = run_hybrid_experiment(
        m=4,
        eps_loose=0.2,
        eps_tight=0.001,
        K_fraction=1.0,
        function_type="delta"
    )
    plt.show()