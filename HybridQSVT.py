import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, Operator
from numpy.polynomial import Chebyshev
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
from HybridPolynomial import (SunderhaufPolynomial,
                            RemezPolynomial,
                            MangPolynomial, hybrid_correction)

from PoissonFunctions import (build_1d_Poisson, eigs_1d_poisson)
import time


# ==============================================================================
# QSVT linear solver
# ==============================================================================
class myQSVT:
    def __init__(self, A, b, kappa=None, nShots=1000, target_error=None,
                 polyMethod='Remez', sigma_cutoff=1.0,
                 eigenvalues=None, n_factor=1.5):
        """
        Parameters
        ----------
        A            : (N, N) real matrix; all singular values must be in (0, 1).
        b            : (N,) right-hand side; normalised
        kappa        : condition number estimate (if None computed here).
        nShots       : shots for QASM simulator (unused in statevector mode).
        target_error : target L-inf error for the 1/x Chebyshev approximation.
        polyMethod   : polynomial method for 1/x approximation.  Options:
                         'Remez', 'RemezCutoff',
                         'Sunderhauf', 'SunderhaufCutoff',
                         'Mang', 'MangCutoff',
                         'Eigenvalue'  (requires all singular values of A).
        sigma_cutoff : cutoff parameter b for '*Cutoff' methods; ignored otherwise.
        eigenvalues  : (N,) array of ALL singular values of A, required (or
                       auto-computed from A) when polyMethod='Eigenvalue'.
        n_factor     : over-parameterisation ratio for 'Eigenvalue'.
                       n = ceil(n_factor * N), d = 2n-1.  Default 1.5.
        """
        self.A = A
        self.b = b
        self.nShots = nShots
        self.n = int(np.log2(len(b)))
        self.ancilla_qubits = 1

        # ── Polynomial method selection ───────────────────────────────
        if polyMethod == 'Remez':
            self.polyMethod = RemezPolynomial
        elif polyMethod == 'Sunderhauf':
            self.polyMethod = SunderhaufPolynomial
        elif polyMethod == 'Mang':
            self.polyMethod = MangPolynomial
        else:
            raise ValueError(f"Unknown polyMethod '{polyMethod}'.")
 
        if kappa is None:
            s = np.linalg.svd(A, compute_uv=False)
            self.kappa = s[0] / s[-1]
            print(f"Computed κ = {self.kappa:.4f}")
        else:
            self.kappa = kappa

        self.dataOK = True # = self._validate_input()
        self.target_error = target_error

        self.angles, self.tau, self.achieved_error = \
            self._get_inverse_phases(self.kappa, target_error=target_error)

        print(f"Generated {len(self.angles)} phase angles for degree {len(self.angles) - 1}")

    # ------------------------------------------------------------------
    # Phase computation
    # ------------------------------------------------------------------
    def _get_inverse_phases(self, kappa, target_error=None):
        a = 1.0 / kappa

        if target_error is not None:
            degree = self.polyMethod.mindegree(target_error, a)
            self.degree = degree
        else:
            degree = max(self.degree, 1)
            if degree % 2 == 0:
                degree += 1
           

        poly = self.polyMethod.poly(degree, a)
        achieved_error = None # placeholder; self.polyMethod.error_for_degree(degree, a)
   
        N_sample = 25 * degree
        x_s    = np.linspace(-1, 1, N_sample)
        M      = np.max(np.abs(poly(x_s))) / np.cos(np.pi * degree / (2 * N_sample))

        tau            = M               # scaling to recover unnormalised A^{-1}b
        poly_normalised = Chebyshev(poly.coef / M)

        max_val = np.max(np.abs(poly_normalised(np.linspace(-1, 1, 2000))))
        if max_val > 0.999:
            scale           = 0.999 / max_val
            poly_normalised = Chebyshev(poly_normalised.coef * scale)
            tau            /= scale


        phases = QuantumSignalProcessingPhases(poly_normalised, signal_operator="Wx")
        return [float(phi) for phi in phases], tau, achieved_error

    # ------------------------------------------------------------------
    # Block encoding  -- ROTATION form to match pyqsp Wx convention
    # ------------------------------------------------------------------
    def get_block_encoding(self):
        """
        One would use effcient block-encoding constructions for sparse or structured A, but here we
        build the (2N x 2N) block-encoding unitary matching pyqsp's Wx signal:

            U_BE = [[ A,                i*sqrt(I - A A†) ],
                    [ i*sqrt(I - A†A),       A†          ]]

        This ensures the effective 2x2 sub-unitary for each singular value
        sigma_i is exactly W_pyqsp(sigma_i) = [[sigma_i, i*sqrt(1-sigma_i^2)], ...].
        """
        N     = self.A.shape[0]
        I     = np.eye(N)
        A_dag = self.A.conj().T

        sqrt_r = scipy.linalg.sqrtm(I - self.A @ A_dag)
        sqrt_l = scipy.linalg.sqrtm(I - A_dag @ self.A)

        U_matrix = np.block([[self.A,   1j * sqrt_r],
                              [1j * sqrt_l, A_dag  ]])

        err = np.max(np.abs(U_matrix @ U_matrix.conj().T - np.eye(2 * N)))
        if err > 1e-10:
            print(f"Warning: block encoding not unitary, max error = {err:.2e}")


        return Operator(U_matrix)

    # ------------------------------------------------------------------
    # Phase gate on ancilla  (diagonal Rz, NOT Rx)
    # ------------------------------------------------------------------
    def _apply_projector_phase(self, circuit, phi, anc_qubit):
        """
        Apply P(phi) = diag(e^{i*phi}, e^{-i*phi}) on the ancilla.

        Qiskit Rz(theta) = diag(e^{-i*theta/2}, e^{+i*theta/2})
        => Rz(-2*phi)    = diag(e^{+i*phi},      e^{-i*phi})     ✓
        """
        circuit.rz(-2.0 * phi, anc_qubit)   # Z-rotation, NOT X-rotation

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------
    def construct_qsvt_circuit(self):
        """
        QSVT sequence: P(phi_0), U_BE, P(phi_1), U_BE, ..., U_BE, P(phi_d)

        Gate appended as list(q_data) + list(q_anc) so that Qiskit places
        q_anc as the most-significant-bit (MSB) block selector, matching
        the mathematical block-encoding convention.
        """
        q_anc  = QuantumRegister(self.ancilla_qubits, 'anc')
        q_data = QuantumRegister(self.n, 'b')
        c      = ClassicalRegister(self.n + self.ancilla_qubits, 'meas')
        qc     = QuantumCircuit(q_anc, q_data, c)

        qc.prepare_state(Statevector(self.b), q_data)
        qc.barrier()

        U_op   = self.get_block_encoding()
        U_gate = U_op.to_instruction()

        for i in range(len(self.angles) - 1):
            self._apply_projector_phase(qc, self.angles[i], q_anc[0])
            qc.append(U_gate, list(q_data) + list(q_anc))

        self._apply_projector_phase(qc, self.angles[-1], q_anc[0])
        qc.barrier()
        qc.measure(range(qc.num_qubits), range(qc.num_qubits))

        print(f"Circuit width: {qc.width()}, depth: {qc.depth()}")
        return qc

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    def _validate_input(self):
        s = np.linalg.svd(self.A, compute_uv=False)
        if np.any(s >= 1.0):
            print("Warning: all singular values must be strictly < 1.")
            return False
        return True

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self, stateVector=True):
        """
        Run QSVT and return the unit-normalised solution direction and success
        probability.

        Returns
        -------
        u_dir : (N,) ndarray
            Unit-normalised solution direction  Re(sv[0::2]) / ||Re(sv[0::2])||,
            proportional to p(A)|b>.
        success_prob : float
            Post-selection probability P = ||sv[0::2]||^2  (complex norm, includes
            both the real/target part and the imaginary QSP-completion part).
        norm_real : float
            ||Re(sv[0::2])||  — the real-part norm only.  Use this (not
            sqrt(success_prob)) to recover the physical compliance:

                C_qsvt = (b @ u_dir) * solver.tau * norm_real
                C_true = b @ np.linalg.solve(A, b)
                rel_compliance_err = abs(C_qsvt - C_true) / C_true

            Derivation: p_norm = p / tau, so Re(sv[0::2]) = p_norm(A)|b> / ||b||
            (with ||b||=1), giving ||p(A)b|| = tau * norm_real.
            sqrt(success_prob) overestimates this because success_prob includes
            the imaginary QSP-completion polynomial, which is orthogonal to the
            target but lives in the same ancilla=0 subspace.

        Statevector index: k = data_idx * 2 + anc_bit
        Ancilla=0 subspace: sv.data[0::2]  (even indices, correct data order).
        Solution direction: real part of the extracted amplitudes.
        """
        if not self.dataOK:
            return None

        qc = self.construct_qsvt_circuit()

        if stateVector:
            print("Running statevector simulation...")
            qc_sv = qc.copy()
            qc_sv.remove_final_measurements()
            sv = Statevector.from_instruction(qc_sv)
            u_qsvt = sv.data[0::2]          # ancilla=0 => even indices
            success_prob = float(np.sum(np.abs(u_qsvt)**2))
        else:
            print("Running QASM simulation...")
            backend = Aer.get_backend('qasm_simulator')
            t_qc    = transpile(qc, backend)
            counts  = backend.run(t_qc, shots=self.nShots).result().get_counts()

            # Qiskit bitstring: rightmost char = qubit 0 = ancilla
            success_counts = {k: v for k, v in counts.items() if k[-1] == '0'}
            total_success  = sum(success_counts.values())
            if total_success == 0:
                print("ERROR: no shots with ancilla=0.")
                return np.zeros(2**self.n)

            success_prob = total_success / self.nShots
            u_qsvt = np.zeros(2**self.n, dtype=complex)
            for bitstr, count in success_counts.items():
                idx          = int(bitstr[:-1], 2)
                u_qsvt[idx]  = np.sqrt(count / total_success)

        # Solution direction is in the REAL PART (imaginary is the QSP completion).
        # The ancilla=0 subspace sv[0::2] has complex amplitudes:
        #   Re(sv[0::2]) ~ p_norm(A)|b>   (target polynomial)
        #   Im(sv[0::2]) ~ completion poly (orthogonal, irrelevant)
        # success_prob = ||sv[0::2]||^2 = ||Re||^2 + ||Im||^2  (COMPLEX norm).
        # For physical compliance recovery we need ||Re(sv[0::2])|| = norm_real,
        # because ||p(A)b|| = tau * norm_real * ||b||  (with ||b||=1).
        # Using sqrt(success_prob) would overestimate this by the imaginary contribution.
        u_real    = u_qsvt.real
        norm_real = float(np.linalg.norm(u_real))
        if norm_real < 1e-12:
            print("ERROR: real part of extracted state has near-zero norm.")
            return None
        return u_real / norm_real, success_prob, norm_real



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
    A, b   = build_1d_Poisson(m, function_type=function_type)
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
    A, b         = build_1d_Poisson(m, function_type=function_type)
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


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":

    A = np.array([[0.5, 0.1], [0.1, 0.3]])
    b = np.array([1.0, 0.0])
    solver = myQSVT(A, b, polyMethod='Remez', target_error=0.01)
    u_dir, success_prob, norm_real = solver.solve(stateVector=True)
    print(f"QSVT solution direction: {u_dir}, success probability: {success_prob:.4f}, norm_real: {norm_real:.4f}")
