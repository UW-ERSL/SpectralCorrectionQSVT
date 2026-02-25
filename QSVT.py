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
from PolynomialApproximators import (SunderhaufPolynomial,
                            RemezPolynomial,
                            MangPolynomial)
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


# ==============================================================================
# Helpers
# ==============================================================================

def build_1D_problem(m: int, function_type="uniform"):
    """Build the 1D Poisson system for a given m (N = 2^m grid points)."""
    N = 2**m
    A = (np.diag(np.full(N, 2.0))
         + np.diag(np.full(N - 1, -1.0), k=1)
         + np.diag(np.full(N - 1, -1.0), k=-1))
    s_max = np.linalg.svd(A, compute_uv=False)[0]
    A = A / (s_max*1.01 )          # rescale: max sv safely < 1

    if function_type == "uniform":
        b = np.ones(N) / np.sqrt(N)     # uniform load, unit norm
    elif function_type == "delta":
        b = np.zeros(N)
        b[N // 2] = 1.0
    elif function_type == "random":
        b = np.random.rand(N)
        b /= np.linalg.norm(b)          # random load, unit norm
    elif function_type == "sine":
        K = 1
        x = np.linspace(0, 1, N)
        b = np.sin(K*np.pi * x) 
        b /= np.linalg.norm(b)          # sine load, unit norm
    else:
        raise ValueError(f"Unknown function_type: {function_type}")
    return A, b

def build_2d_poisson(m, function_type="uniform"):
    N1 = 2**m
    h  = 1.0 / (N1 + 1)
    # 1D tridiagonal
    T  = (2.0 * np.eye(N1) - np.diag(np.ones(N1-1), 1)
                            - np.diag(np.ones(N1-1), -1)) / h**2
    I  = np.eye(N1)
    A  = np.kron(T, I) + np.kron(I, T)   # N^2 x N^2

    N = A.shape[0]
    if function_type == "uniform":
        b = np.ones(N) / np.sqrt(N)     # uniform load, unit norm
    elif function_type == "delta":
        b = np.zeros(N)
        b[N // 2] = 1.0
    elif function_type == "random":
        b = np.random.rand(N)
        b /= np.linalg.norm(b)          # random load, unit norm
    elif function_type == "sine":
        K = 1
        x = np.linspace(0, 1, N)
        b = np.sin(K*np.pi * x) 
        b /= np.linalg.norm(b)          # sine load, unit norm
    else:
        raise ValueError(f"Unknown function_type: {function_type}")
    return A

def eigs_1d_poisson(m):
    N1  = 2**m
    h   = 1.0 / (N1 + 1)
    k   = np.arange(1, N1 + 1)
    lam = 4.0 / h**2 * np.sin(k * np.pi / (2*(N1+1)))**2
    lam = np.sort(lam)
    lam = lam / lam.max()   # normalise to [0,1]
    return lam

def eigs_2d_poisson(m):
    N1   = 2**m
    h    = 1.0 / (N1 + 1)
    k    = np.arange(1, N1 + 1)
    lam1 = 4.0 / h**2 * np.sin(k * np.pi / (2*(N1+1)))**2
    # tensor product: all pairwise sums
    lam2d = (lam1[:, None] + lam1[None, :]).ravel()
    return np.sort(lam2d)

def run_comparison(A, b, kappa=None, target_error=0.1, sigma_cutoff=0.5,
                   methods="Remez", stateVector=True,
                   eigenvalues=None, n_factor=1.5):
    """
    Run myQSVT for each polynomial method and collect results.

    Parameters
    ----------
    A            : system matrix (2D numpy array).
    b            : right-hand side vector (1D numpy array).
    kappa        : condition number of A (if None, computed internally).
    target_error : target L-inf error for 1/x approximation.
    sigma_cutoff : spectral cutoff b for '*Cutoff' methods; ignored otherwise.
    methods      : list of method name strings.
    stateVector  : use statevector (True) or QASM simulator (False).
    eigenvalues  : all singular values of A, passed to 'Eigenvalue' method.
                   If None and 'Eigenvalue' is in methods, computed from A.
    n_factor     : over-parameterisation ratio for 'Eigenvalue' method.

    Returns
    -------
    results : dict keyed by method name, each value a dict with keys
              degree, fidelity, success_prob, x_qsvt, solver.
    x_cl    : classical solution (normalised), for comparison.
    """
    m = int(np.log2(len(b)))

    if kappa is None:
        kappa = np.linalg.cond(A)

    # Pre-compute eigenvalues once if any Eigenvalue method is requested
    eigs_for_solver = eigenvalues
    if eigs_for_solver is None and "Eigenvalue" in methods:
        eigs_for_solver = np.linalg.svd(A, compute_uv=False)

    # Classical solution
    u_cl_unnorm = np.linalg.solve(A, b)
    C_true      = float(b @ u_cl_unnorm)
    x_cl        = u_cl_unnorm / np.linalg.norm(u_cl_unnorm)

    results = {}
    for method in methods:
        print(f"\n{chr(8212)*50}")
        print(f"Method: {method}")
        print(f"{chr(8212)*50}")
        try:
            start_time = time.time()
            solver = myQSVT(A, b, kappa=kappa,
                            target_error=target_error,
                            polyMethod=method,
                            sigma_cutoff=sigma_cutoff,
                            eigenvalues=eigs_for_solver,
                            n_factor=n_factor)
            x_qsvt, succ_prob, norm_real = solver.solve(stateVector=stateVector)
            elapsed = time.time() - start_time

            if x_qsvt is None:
                results[method] = dict(degree=solver.degree, fidelity=None,
                                       success_prob=succ_prob, x_qsvt=None,
                                       solver=solver, compliance_err=None,
                                       max_error=None, elapsed_time=elapsed)
                continue

            # ── Physical compliance recovery ─────────────────────────────
            # solve() returns:
            #   x_qsvt   = Re(sv[0::2]) / norm_real   (unit direction)
            #   succ_prob = ||sv[0::2]||^2             (complex: Re^2 + Im^2)
            #   norm_real = ||Re(sv[0::2])||           (real-part norm only)
            #
            # The circuit encodes p_norm(A)|b> in Re(sv[0::2]), so:
            #   ||p(A)b|| = tau * norm_real   (since p = tau * p_norm, ||b||=1)
            #
            # Physical compliance: C_qsvt = b^T p(A)b
            #   = (b @ x_qsvt) * ||p(A)b||
            #   = (b @ x_qsvt) * tau * norm_real
            #
            # Using sqrt(succ_prob) instead would overestimate ||p(A)b|| because
            # succ_prob includes the imaginary QSP-completion part.
            C_qsvt        = float(b @ x_qsvt) * solver.tau * norm_real
            compliance_err = abs(C_qsvt - C_true) / abs(C_true)

            fidelity = float(np.abs(np.vdot(x_qsvt, x_cl))**2)
            max_err  = np.max(np.abs(x_qsvt - x_cl)) / np.max(np.abs(x_cl))

            results[method] = dict(degree=solver.degree, fidelity=fidelity,
                                   success_prob=succ_prob, x_qsvt=x_qsvt,
                                   solver=solver, compliance_err=compliance_err,
                                   max_error=max_err, elapsed_time=elapsed,
                                   C_qsvt=C_qsvt, C_true=C_true)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results[method] = dict(degree=solver.degree, fidelity=None,
                                   success_prob=None, x_qsvt=None,
                                   solver=solver, compliance_err=None,
                                   max_error=None, elapsed_time=elapsed)

    return results, x_cl


def print_summary(results):
    """Print a formatted comparison table to stdout.

    Columns
    -------
    Rel U_err   : ||u_qsvt - u_class|| / ||u_class||  (normalised direction error)
    Rel J_err   : |C_qsvt - C_true| / C_true          (physical compliance error)
    C_ratio     : C_qsvt / C_true                     (physical compliance ratio;
                  < 1 means compliance underestimated, which is always the case
                  when the cutoff removes low-frequency modes)
    Succ. prob. : post-selection probability P = ||p(A)b||^2 / (tau^2 ||b||^2)
    """
    hdr = (f"\n{'='*85}\n"
           f"{'Method':<18} {'Degree':>6} {'Rel U_err':>10} "
           f"{'Rel J_err':>10} {'C_ratio':>8} {'Succ.prob':>10}  {'Time':>7}\n"
           f"{chr(45)*85}")
    print(hdr)
    for method, r in results.items():
        deg     = r.get("degree")
        prob    = r.get("success_prob")
        c_err   = r.get("compliance_err")
        max_err = r.get("max_error")
        elapsed = r.get("elapsed_time")
        C_q     = r.get("C_qsvt")
        C_t     = r.get("C_true")
        c_ratio = (C_q / C_t) if (C_q is not None and C_t) else None

        d_s   = str(deg)            if deg      is not None else "FAILED"
        p_s   = f"{prob:.4f}"       if prob     is not None else "—"
        ce_s  = f"{c_err:.4f}"      if c_err    is not None else "—"
        me_s  = f"{max_err:.4f}"    if max_err  is not None else "—"
        cr_s  = f"{c_ratio:.4f}"    if c_ratio  is not None else "—"
        e_s   = f"{elapsed:.2f}s"   if elapsed  is not None else "—"
        print(f"{method:<18} {d_s:>6} {me_s:>10} {ce_s:>10} {cr_s:>8} {p_s:>10}  {e_s:>7}")
    print(f"{'='*85}")


def plot_comparison(results, x_cl, m, target_error):
    """
    Two-panel figure:
      Left  — normalised solution amplitude for each method vs classical.
      Right — bar chart of polynomial degree (quantum gate cost proxy).
    """
    colors  = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "v"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: solution vectors ────────────────────────────────────────
    ax = axes[0]
    ax.plot(x_cl, "k--", lw=2, label="Classical", zorder=10)
    for idx, (method, r) in enumerate(results.items()):
        if r["x_qsvt"] is None:
            continue
        fid = r["fidelity"]
        ax.plot(r["x_qsvt"],
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                lw=1.5, ms=6,
                label=f"{method}  (F={fid:.6f})")
    ax.set_xlabel("Grid index")
    ax.set_ylabel("Normalised amplitude")
    ax.set_title(f"Solution comparison  (m={m}, eps={target_error:.0e})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Right: degree bar chart ───────────────────────────────────────
    ax = axes[1]
    valid   = {k: v for k, v in results.items() if v["degree"] is not None}
    names   = list(valid.keys())
    degrees = [valid[k]["degree"] for k in names]
    bars = ax.bar(names, degrees,
                  color=[colors[i % len(colors)] for i in range(len(names))],
                  edgecolor="black", linewidth=0.8)
    for bar, deg in zip(bars, degrees):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(degrees) * 0.01,
                str(deg), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Polynomial degree")
    ax.set_title("Degree comparison (gate cost proxy)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":

    # ── Experiment configuration ──────────────────────────────────────────────
    m             = 4                                    # N = 2^m grid points
    J_tolerance   = 0.025                               # target physical compliance tolerance
    target_error  = J_tolerance / 2                     # L-inf target for 1/x polynomial
    stateVector   = True                                # False => QASM shot simulator
    function_type = "uniform"                             # "uniform" | "delta" | "random" | "sine"
    sigma_cutoff  = 0.5                                 # spectral cutoff for *Cutoff methods

    A, b = build_1D_problem(m, function_type=function_type)

    # Eigenvalues precomputed once; corrected kappa uses sigma_min directly
    # so that a = 1/kappa = sigma_min = lambda_1 (no 1% undershoot).
    eigs  = np.linalg.svd(A, compute_uv=False)
    kappa = 1.0 / float(eigs[-1])

    # True physical compliance (scale reference for all methods)
    C_true = float(b @ np.linalg.solve(A, b))

    print(f"{'='*60}")
    print(f"1D Poisson QSVT  m={m} (N={2**m})")
    print(f"κ (exact) = {kappa:.4f}   J_tol = {J_tolerance:.4f}")
    print(f"target_error = {target_error:.4f}   sigma_cutoff = {sigma_cutoff:.3f}")
    print(f"function_type = {function_type}")
    print(f"C_true = b^T A^{{-1}} b = {C_true:.6f}")

    methods = ["Remez", "Sunderhauf", "Mang"]

    # ── Run ───────────────────────────────────────────────────────────────────
    results, x_cl = run_comparison(A, b,
        kappa=kappa, target_error=target_error, sigma_cutoff=sigma_cutoff,
        methods=methods, stateVector=stateVector,
        eigenvalues=eigs, n_factor=1.5,
    )

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"1D Poisson QSVT  m={m} (N={2**m})")
    print(f"κ = {kappa:.4f}   target_error = {target_error:.4f}   sigma_cutoff = {sigma_cutoff:.3f}")
    print(f"function_type = {function_type}   C_true = {C_true:.6f}")
    print_summary(results)
    plot_comparison(results, x_cl, m=m, target_error=target_error)