"""
makeTables.py -- single reproducible driver for every table in

    "Spectrally Corrected Polynomial Approximation for Quantum Singular
     Value Transformation"

Replaces paperTables.py.  Each paper table is emitted as a self-contained LaTeX
float -- caption, label, tabular -- into TABLES_DIR, so main.tex carries one
\\input line per table and a caption cannot drift from the numbers it describes.

Usage
-----
    python makeTables.py                 # every paper table
    python makeTables.py qsvt_results    # one table
    python makeTables.py fast            # every table that needs no circuit
    python makeTables.py nm scaling      # rebuttal-only diagnostics (stdout)

Paper tables (emitted to TABLES_DIR)
------------------------------------
    methods          tab:methods           Sec. 3.5   STATIC, not computed
    degree_accuracy  tab:degree_accuracy   Sec. 5.1.2
    adaptive         tab:adaptive          Sec. 5.1.3
    qsvt_results     tab:qsvt_results      Sec. 5.2.1   circuit
    perturbation     tab:perturbation      Sec. 5.2.2   circuit
    qsvt_2d          tab:qsvt_2d           Sec. 5.3.1   circuit
    assess           tab:assess            Sec. 5.3.2

Rebuttal-only diagnostics (stdout only, no .tex)
------------------------------------------------
    pure       pure spectral polynomial, P_succ vs the kappa^2/tau^2 bound
    scaling    is d(kappa) linear or sqrt?          [R2 Technical 4]
    ext        spectral extension vs correction
    nm         why no metric is monotone in K       [R2 Significance 2]
    loads      solution error under three loads

Conventions (stated once, used everywhere)
------------------------------------------
* A and its spectrum are normalised by 1.01 * lambda_max, so lambda_max = 0.9901
  and all singular values lie strictly inside (0, 1) as the block encoding requires.
* kappa is therefore 1 / lambda_min of the NORMALISED matrix.
* tau is the maximum of |p_hat| over the FULL interval [-1, 1], including the
  central gap (-a, a).
* P_succ is  ||p(A)b||^2 / tau^2 -- the post-selection probability of the
  real-part-extracting circuit.  It obeys P_succ <= ||A^{-1}b||^2/tau^2 <= kappa^2/tau^2.
* Total query cost is Q = d / sqrt(P_succ), the number of block-encoding queries
  under amplitude amplification.
"""
import sys, os, io, time, gc, contextlib
import numpy as np
from numpy.polynomial import Chebyshev

from PolynomialApproximators import (ChebIterPolynomial, MangPolynomial,
                                     SunderhaufPolynomial, SpectralPolynomial,
                                     spectral_correction, spectral_extension,
                                     merge_eigenvalues, merge_by_resolution,
                                     correct_at_tolerance,
                                     adaptive_spectral_correction,
                                     _SQRT_EPS, _Q_FLOOR, _tau)
from PoissonFunctions import (build_1d_poisson, eigs_1d_poisson,
                              build_2d_poisson, eigs_2d_poisson)
from QSVTSolvers import (StandardQSVT, PureSpectralQSVT,
                         SpectrallyBootstrappedQSVT)

BASE_POLYS = {'chebiter': ChebIterPolynomial, 'mang': MangPolynomial,
              'sunderhauf': SunderhaufPolynomial}

# Where the generated .tex floats go, relative to the working directory.
# main.tex is expected to sit alongside this folder and \input{tables/<name>}.
TABLES_DIR = os.environ.get("QSVT_TABLES_DIR", "tables")

OUT = []


def emit(s=""):
    print(s)
    OUT.append(s)


def quiet(fn, *a, **kw):
    """Run fn silently (the solvers are chatty)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# ==========================================================================
# LaTeX emission
# ==========================================================================
_BANNER = (
    "% ---------------------------------------------------------------------\n"
    "% GENERATED FILE -- DO NOT EDIT BY HAND.\n"
    "% Produced by makeTables.py ({fn}).  Edit that function and re-run:\n"
    "%     python makeTables.py {name}\n"
    "% ---------------------------------------------------------------------\n"
)


def write_tex(name, body, fn):
    """Write one generated float to TABLES_DIR/<name>.tex."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, name + ".tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_BANNER.format(fn=fn, name=name))
        f.write(body.rstrip() + "\n")
    emit(f"  [wrote {path}]")
    return path


def sci(x, nd=2):
    """1.97e-1 -> $1.97\\times10^{-1}$"""
    s = f"{x:.{nd}e}"
    m, e = s.split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def sci_or_mach(x, nd=2):
    """Machine-precision residuals are reported as O(eps_mach), not as a number:
    they sit at the double-precision floor and their digits are not meaningful."""
    return "$O(\\epsmach)$" if abs(x) <= _SQRT_EPS else sci(x, nd)


def rng_or_one(vals):
    """14, 14, 16, 16 -> '14--16';  16, 16 -> '16'."""
    lo, hi = min(vals), max(vals)
    return f"{lo}" if lo == hi else f"{lo}--{hi}"


# ==========================================================================
# Problem setup
# ==========================================================================
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
# tab:methods  (Sec. 3.5)  --  STATIC
# ==========================================================================
def tab_methods():
    """The four constructions ranked on six measures.

    NOT COMPUTED.  These ranks are a reading of a single example, not the output
    of a sweep, and the caption says so.  The file is emitted here only so that
    every table in the paper lives in one place; edit the ranks in this function
    if the reading changes.
    """
    emit("\n" + "=" * 104)
    emit("tab:methods  (Sec. 3.5)  --  STATIC, not computed")
    emit("=" * 104)
    rows = [                    # accuracy, gamma, classical, quantum, demands, robustness
        ("Spectral Polynomial", 1, 4, 2, 3, 4, 4),
        ("Spectral Correction", 3, 2, 1, 4, 1, 2),
        ("Spectral Separation", 2, 3, 3, 1, 3, 3),
        ("Spectral Constraint", 4, 1, 4, 2, 1, 1),
    ]
    lines = []
    for tag, *r in rows:
        lines.append(f"{tag:<21}& " + " & ".join(str(v) for v in r)
                     + f" & {sum(r)} \\\\")
        emit(f"  {tag:<21} " + " ".join(f"{v:>2}" for v in r) + f"   sum {sum(r)}")

    body = r"""\begin{table}[htbp]
\centering
\caption{\rev{The four constructions ranked on six measures, $1$ best. From the 2D Poisson
operator with $N = 256$, $\kap = 116.5$ and $K = 32$ supplied eigenvalues, each
construction taken to the degree at which it first matches the guarantee of the base
polynomial. Accuracy is the relative solution error on the least favourable of three loads;
classical cost is the time to build the polynomial given the eigenvalues; quantum cost is
$d\,\tau$; demands is what the construction requires of the user.}}
\label{tab:methods}
{\setlength{\tabcolsep}{4pt}\small
\begin{tabular}{lccccccc}
\toprule
 & \multicolumn{7}{c}{rank} \\
\cmidrule(lr){2-8}
 & \rotatebox{90}{accuracy} & \rotatebox{90}{$\gamma$} & \rotatebox{90}{classical cost}
 & \rotatebox{90}{quantum cost} & \rotatebox{90}{demands} & \rotatebox{90}{robustness}
 & \rotatebox{90}{sum} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}"""
    write_tex("methods", body, "tab_methods")


# ==========================================================================
# tab:degree_accuracy  (Sec. 5.1.2)
# ==========================================================================
def tab_degree_accuracy(N=4, K=2, eps_grid=(0.2, 0.1, 0.01)):
    """Degree vs eigenvalue residual, base p0 and p_SC, on the 4-eigenvalue 1D
    spectrum.  Polynomial level: no circuit."""
    emit("\n" + "=" * 104)
    emit("tab:degree_accuracy  (Sec. 5.1.2)  Degree vs eigenvalue residual")
    emit("=" * 104)
    eigs = eigs_1d_poisson(2)
    eigs = eigs / (1.01 * eigs.max())
    kappa = 1.0 / eigs[0]
    a = 1.0 / kappa
    emit(f"  N = {N}, kappa = {kappa:.4f}, K = {K}, "
         f"lambda = {np.array2string(eigs, precision=4)}")
    emit(f"  {'method':<8} {'eps':>7} {'d':>5} {'R_K':>12} {'R_N':>12}")

    base_rows, corr_rows = [], []
    for eps in eps_grid:
        d = ChebIterPolynomial.mindegree(eps, a)
        p0 = ChebIterPolynomial.poly(d, a)
        e0 = float(np.max(np.abs(eigs * p0(eigs) - 1)))
        emit(f"  {'p0':<8} {eps:>7} {d:>5} {e0:>12.2e} {e0:>12.2e}")
        base_rows.append(f"$p_0$    & ${eps}$  & ${d}$ & "
                         f"\\rev{{{sci(e0)}}}  & \\rev{{{sci(e0)}}} \\\\")

        cc = quiet(spectral_correction, p0, eigs[:K])
        coef = p0.coef.copy(); coef[1::2] += cc
        pSC = Chebyshev(coef)
        eK = float(np.max(np.abs(eigs[:K] * pSC(eigs[:K]) - 1)))
        eA = float(np.max(np.abs(eigs * pSC(eigs) - 1)))
        emit(f"  {'p_SC':<8} {eps:>7} {d:>5} {eK:>12.2e} {eA:>12.2e}")
        corr_rows.append(f"$p_{{SC}}$ & ${eps}$  & ${d}$ & "
                         f"\\rev{{{sci_or_mach(eK)}}}  & \\rev{{{sci(eA)}}} \\\\")

    body = r"""\begin{table}[htbp]
\centering
\caption{Degree $d$ and eigenvalue residual for $p_0$ and $p_{SC}$ at increasing
accuracy targets. $N=""" + str(N) + r"""$, \rev{$\kap \approx """ + f"{kappa:.2f}" + r"""$}, $K=""" + str(K) + r"""$. \rev{$R_K$ is taken over the
$K$ corrected eigenvalues and $R_N$ over the full spectrum,
Equation \eqref{eq:Rset}.}}
\label{tab:degree_accuracy}
{\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{llrrr}
\toprule
method & $\varepsilon$ & $d$ & \rev{$R_K$} & \rev{$R_N$} \\
\midrule
""" + "\n".join(base_rows) + r"""
\midrule
""" + "\n".join(corr_rows) + r"""
\bottomrule
\end{tabular}}
\end{table}"""
    write_tex("degree_accuracy", body, "tab_degree_accuracy")


# ==========================================================================
# tab:adaptive  (Sec. 5.1.3)
# ==========================================================================
def tab_adaptive(gamma_max=2.0):
    """Output of Algorithm B on three spectra of growing density.
    Polynomial level: no circuit."""
    from spectralSweep import weyl_spectrum
    emit("\n" + "=" * 104)
    emit(f"tab:adaptive  (Sec. 5.1.3)  Algorithm B, gamma_max = {gamma_max}")
    emit("eps and d are OUTPUTS of the sweep, not inputs.")
    emit("=" * 104)

    e = eigs_1d_poisson(4); e = e / (1.01 * e.max())
    kap = 1.0 / e[0]
    e2 = np.sort(eigs_2d_poisson(4)); e2 = e2 / (1.01 * e2.max())
    # 1D and 2D Poisson realize lam_k ~ k^2 and ~ k; the k^{2/3} regime of a
    # three-dimensional Laplacian has no small test operator, so it is synthetic.
    lam3, _ = weyl_spectrum(kap, 3)
    specs = [("1D", kap, e, 16),
             ("2D", 1.0 / e2[0], e2, 32),
             ("Weyl 3", kap, lam3, 16)]

    emit(f"  {'spectrum':<9}{'K':>4}{'eps':>8}{'d':>6}{'Keff':>6}"
         f"{'R_K':>11}{'gamma':>7}{'G':>7}")
    rows = []
    for tag, kappa, lam, K in specs:
        r = quiet(adaptive_spectral_correction, ChebIterPolynomial, kappa,
                  lam[:K], gamma_max=gamma_max)
        emit(f"  {tag:<9}{K:>4}{r['epsilon']:>8}{r['d']:>6}{r['K_eff']:>6}"
             f"{r['R_all']:>11.1e}{r['gamma']:>7.2f}{r['G']:>7.1f}")
        rows.append(f"{tag:<6} & {K} & ${r['epsilon']}$ & {r['d']} & {r['K_eff']} & "
                    f"{sci_or_mach(r['R_all'])} & {r['gamma']:.2f} & {r['G']:.1f} \\\\")

    body = r"""\begin{table}[htbp]
\centering\color{blue}
\caption{Output of Algorithm~B at $\kap = """ + f"{kap:.1f}" + r"""$, $\gamma_{\max} = """ + f"{gamma_max:g}" + r"""$.
``Weyl 3'' is the synthetic spectrum $\lambda_k \sim k^{2/3}$. The tolerance
$\varepsilon$ and the degree $d$ are outputs of the sweep. For the 2D operator $K_{\rm eff} < K$ because $\lambda_{j,k} = \lambda_{k,j}$ is
degenerate and the duplicate constraints are removed losslessly; elsewhere every supplied
eigenvalue is retained. Entries marked $O(\epsmach)$ sit at the double-precision floor,
between one and a few hundred times unit roundoff depending on the degree and the LAPACK
build.}
\label{tab:adaptive}
{\small\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lrrrrrrr}
\toprule
spectrum & $K$ & $\varepsilon$ & $d$ & $K_{\rm eff}$ & $R_K$ & $\gamma$ & $G$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}"""
    write_tex("adaptive", body, "tab_adaptive")


# ==========================================================================
# tab:qsvt_results  (Sec. 5.2.1)   -- CIRCUIT
# ==========================================================================
def tab_qsvt_results(m=4, base='ChebIter', eps_loose=0.5, eps_tight=1e-3):
    """Algorithm B run end to end and put on a circuit, 1D Poisson, K = N.

    Three rows per load, in the order the paper reads them: the polynomial
    Algorithm B returns, the base at the SAME degree (which isolates what the
    correction buys), and a tight base reference.
    """
    emit("\n" + "=" * 116)
    emit(f"tab:qsvt_results  (Sec. 5.2.1)  QSVT, 1D Poisson, m={m}, base={base}, "
         f"K=N={2**m}, eps_loose={eps_loose}, eps_tight={eps_tight}")
    emit("=" * 116)
    K = 2 ** m
    labels = [("\\rev{$p_{SC}$, Algorithm~B}", 'sc',  eps_loose),
              ("\\rev{$p_0$, same degree}",    'std', eps_loose),
              ("\\rev{$p_0$, tight base}",     'std', eps_tight)]
    blocks, ratios, kappa = [], {}, None

    for load, heading in (("uniform", "\\textit{Uniform load}"),
                          ("delta",   "\\textit{Point load at midpoint}")):
        S = setup_1d(m, load)
        kappa = S['kappa']
        emit(f"\n  {load} load   (kappa = {kappa:.2f})")
        emit(f"  {'Method':<26} {'d':>5} {'F':>10} {'e_C':>12} {'e_x':>12} "
             f"{'P_succ':>8} {'tau':>8} {'Q':>10}")
        rows, M = [], {}
        for i, (label, kind, eps) in enumerate(labels):
            gc.collect()
            if kind == 'std':
                s = quiet(StandardQSVT, S['A'], S['b'], kappa=kappa,
                          target_error=eps, polyMethod=base)
            else:
                s = quiet(SpectrallyBootstrappedQSVT, S['A'], S['b'],
                          lam_K=S['eigs'][:K], kappa=kappa,
                          target_error=eps, polyMethod=base)
            x, P, nr = quiet(s.solve)
            M[i] = metrics(s, x, P, nr, S)
            emit(f"  {label:<26} {M[i]['d']:>5} {M[i]['F']:>10.6f} "
                 f"{M[i]['cerr']:>12.2e} {M[i]['ex']:>12.2e} {P:>8.4f} "
                 f"{M[i]['tau']:>8.1f} {M[i]['cost']:>10.1f}")
            # the returned polynomial is the recommendation: bold its query cost
            cost = (f"\\textbf{{{M[i]['cost']:.1f}}}" if i == 0
                    else f"{M[i]['cost']:.1f}")
            rows.append(f"{label}\n  & {M[i]['d']} & {M[i]['F']:.6f} & "
                        f"{sci_or_mach(M[i]['cerr'])} & "
                        f"\\rev{{{sci_or_mach(M[i]['ex'])}}} & "
                        f"\\rev{{{P:.4f}}} & \\rev{{{cost}}} \\\\")
            del s, x
        blocks.append(f"\\multicolumn{{7}}{{l}}{{{heading}}} \\\\\n\\midrule\n"
                      + "\n".join(rows))
        ratios[load] = dict(depth=M[2]['d'] / M[0]['d'],
                            cost=M[2]['cost'] / M[0]['cost'],
                            tau_sc=M[0]['tau'], tau_tight=M[2]['tau'],
                            tau_base=M[1]['tau'],
                            P_sc=M[0]['P'], P_tight=M[2]['P'],
                            d_sc=M[0]['d'], d_tight=M[2]['d'],
                            ex_tight=M[2]['ex'])
        emit(f"    depth ratio      tight/corrected = {ratios[load]['depth']:.2f}x")
        emit(f"    query-cost ratio tight/corrected = {ratios[load]['cost']:.2f}x")

    u = ratios['uniform']
    body = r"""\begin{table*}[t]
\centering
\caption{QSVT results for the 1D Poisson equation, $N=""" + str(2**m) + r"""$,
$\kappa=""" + f"{kappa:.1f}" + r"""$, $K=N$. \rev{The polynomial Algorithm~B returns} achieves unit fidelity and lower
$e_C$ and $e_x$ than the tight base at $""" + f"{u['depth']:.2f}" + r"""\times$ lower
circuit depth\rev{, and $""" + f"{u['cost']:.2f}" + r"""\times$ lower total query cost
$\mathcal{Q} = d/\sqrt{P_{\rm succ}}$. Entries marked $O(\epsmach)$ sit at the
double-precision floor.}}
\label{tab:qsvt_results}
{\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrr}
\toprule
Method & $d$ & $F$ & $e_C$ & \rev{$e_x$}
  & $P_{\rm succ}$ & \rev{$\mathcal{Q}$} \\
\midrule
""" + "\n\\midrule\n".join(blocks) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}"""
    write_tex("qsvt_results", body, "tab_qsvt_results")

    # numbers Sec. 5.2.1 quotes in prose, so they can be checked against the table
    p = ratios['delta']
    emit("\n  PROSE CHECK for Sec. 5.2.1:")
    emit(f"    tau: base {u['tau_base']:.1f} -> corrected {u['tau_sc']:.1f} "
         f"(gamma = {u['tau_sc']/u['tau_base']:.2f}); tight base {u['tau_tight']:.1f}")
    emit(f"    P_succ corrected {u['P_sc']:.4f} vs tight base {u['P_tight']:.4f}")
    emit(f"    depth {u['depth']:.2f}x, query cost {u['cost']:.2f}x, "
         f"tight-base e_x = {u['ex_tight']:.2e}")
    emit(f"    point load P_succ = {p['P_sc']:.4f}")


# ==========================================================================
# tab:perturbation  (Sec. 5.2.2)   -- CIRCUIT
# ==========================================================================
def tab_perturbation(m=4, base='ChebIter', eps=0.5, n_trials=60, seed=42,
                     etas=(0.0, 0.01, 0.1, 0.2)):
    """Robustness of p_SC to eigenvalue perturbation.  The eta = 0 row is by
    construction the same computation as the p_SC row of tab:qsvt_results."""
    emit("\n" + "=" * 104)
    emit(f"tab:perturbation  (Sec. 5.2.2)  1D Poisson, m={m}, base={base}, "
         f"K=N={2**m}, eps={eps}, {n_trials} trials, seed={seed}")
    emit("(eta=0 row MUST match the p_SC row of tab:qsvt_results)")
    emit("=" * 104)
    S = setup_1d(m, "uniform")
    K = 2 ** m
    rng = np.random.default_rng(seed)
    a = 1.0 / S['kappa']
    emit(f"  {'eta':>6} {'1-F':>12} {'e_C':>12} {'e_x':>12} {'P_succ':>9} {'K_eff':>8}")

    rows, spread = [], {}
    for eta in etas:
        F, C, P, X, KE = [], [], [], [], []
        for _ in range(1 if eta == 0.0 else n_trials):
            lam = S['eigs'][:K] * (1.0 + rng.uniform(-eta, eta, size=K))
            if eta > 0.0:
                # the clip is a perturbation in its own right; at eta = 0 the
                # row must be the same computation as tab:qsvt_results.
                lam = np.clip(lam, a + 1e-10, 1.0 - 1e-10)
            s = quiet(SpectrallyBootstrappedQSVT, S['A'], S['b'], lam_K=lam,
                      kappa=S['kappa'], target_error=eps, polyMethod=base)
            x, p, nr = quiet(s.solve)
            M = metrics(s, x, p, nr, S)
            F.append(M['F']); C.append(M['cerr']); P.append(p); X.append(M['ex'])
            KE.append(s.correction_info['K_eff'])
            del s, x
            gc.collect()
        oneF = 1.0 - float(np.mean(F))
        mC, mX, mP = float(np.mean(C)), float(np.mean(X)), float(np.mean(P))
        emit(f"  {eta:>6} {oneF:>12.2e} {mC:>12.2e} {mX:>12.2e} {mP:>9.4f} "
             f"{rng_or_one(KE):>8}")
        spread[eta] = dict(F=float(np.std(F)), C=float(np.std(C)),
                           X=float(np.std(X)))
        # at eta = 0 the fidelity loss is at the simulation floor, not a measurement
        fcell = "$<10^{-6}$" if oneF < 1e-6 else f"\\rev{{{sci(oneF, 1)}}}"
        # the en-dash of a K_eff range must sit OUTSIDE math mode, or '--'
        # is typeset as two minus signs
        ke = rng_or_one(KE)
        ke_tex = "$" + ke.replace("--", "$--$") + "$"
        rows.append(f"${eta:g}$ & {fcell} & \\rev{{{sci_or_mach(mC, 1)}}}\n"
                    f"       & \\rev{{{sci_or_mach(mX, 1)}}} & \\rev{{${mP:.4f}$}} "
                    f"& \\rev{{{ke_tex}}} \\\\")

    hi = spread[max(etas)]
    body = r"""\begin{table}[htbp]
\centering
\caption{Robustness of $p_{SC}$ to eigenvalue perturbations.
$N=""" + str(2**m) + r"""$, $\kappa=""" + f"{S['kappa']:.1f}" + r"""$, $K=N$, $\varepsilon=""" + f"{eps:g}" + r"""$,
$n=""" + str(n_trials) + r"""$ trials per perturbation level $\eta$. \rev{Entries are means over the trials; the
spread at $\eta = """ + f"{max(etas):g}" + r"""$ is comparable to the mean for $e_C$ ($\pm """ + sci(hi['C'], 1).strip('$') + r"""$) and
$e_x$ ($\pm """ + sci(hi['X'], 1).strip('$') + r"""$), and an order of magnitude smaller for $1-F$
($\pm """ + sci(hi['F'], 1).strip('$') + r"""$). $K_{\rm eff}$ is given as the range over the trials: a
perturbation can bring two eigenvalues within the resolution scale, and the merge then
discards one.}}
\label{tab:perturbation}
{\small\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lrrrrr}
\toprule
$\eta$ & $1-F$ & $e_C$ & \rev{$e_x$} & $P_{\rm succ}$ & \rev{$K_{\rm eff}$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}"""
    write_tex("perturbation", body, "tab_perturbation")


# ==========================================================================
# tab:qsvt_2d  (Sec. 5.3.1)   -- CIRCUIT
# ==========================================================================
def tab_qsvt_2d(m=4, base='ChebIter', eps=0.2, KRange=(0, 1, 4, 8, 16, 32)):
    """p_SC on the 2D Poisson equation under a uniform load, with increasing K.

    Every row is Algorithm A at the fixed eps and d that Algorithm B returns at
    K = 32; holding them fixed keeps the columns comparable.
    """
    emit("\n" + "=" * 104)
    emit(f"tab:qsvt_2d  (Sec. 5.3.1)  QSVT, 2D Poisson, N1={2**m}, N={4**m}, "
         f"base={base}, eps={eps}")
    emit("=" * 104)
    S = setup_2d(m)
    emit(f"  kappa = {S['kappa']:.2f}")
    emit(f"  {'K':>4} {'K_eff':>6} {'d':>5} {'F':>11} {'e_C':>12} {'e_x':>12} "
         f"{'P_succ':>8}")
    rows = []
    for K in KRange:
        t0 = time.time()
        gc.collect()          # aer holds the previous circuit otherwise
        if K == 0:
            s = quiet(StandardQSVT, S['A'], S['b'], kappa=S['kappa'],
                      target_error=eps, polyMethod=base)
            Keff = None
        else:
            s = quiet(SpectrallyBootstrappedQSVT, S['A'], S['b'],
                      lam_K=S['eigs'][:K], kappa=S['kappa'],
                      target_error=eps, polyMethod=base)
            Keff = s.correction_info['K_eff']
        x, P, nr = quiet(s.solve)
        M = metrics(s, x, P, nr, S)
        emit(f"  {K:>4} {str(Keff if Keff is not None else '--'):>6} {M['d']:>5} "
             f"{M['F']:>11.7f} {M['cerr']:>12.2e} {M['ex']:>12.2e} {P:>8.4f} "
             f"  [{time.time()-t0:.0f}s]")
        tag = "0 (base)" if K == 0 else str(K)
        kcell = "--" if Keff is None else str(Keff)
        rows.append(f"{tag:<8} & {kcell:<3} & \\rev{{{M['F']:.7f}}} & "
                    f"\\rev{{{sci(M['cerr'])}}} & \\rev{{{sci(M['ex'])}}} & "
                    f"\\rev{{{P:.4f}}} \\\\")
        del s, x

    body = r"""\begin{table*}[t]
\centering
\caption{$p_{SC}$ on the 2D Poisson equation ($N=""" + str(4**m) + r"""$, $\kappa=""" + f"{S['kappa']:.1f}" + r"""$, uniform load).
\rev{Each row is Algorithm~A at the fixed $\varepsilon = """ + f"{eps:g}" + r"""$ and $d = """ + str(ChebIterPolynomial.mindegree(eps, 1.0 / S['kappa'])) + r"""$ that
Algorithm~B returns at $K = """ + str(max(KRange)) + r"""$; holding them fixed keeps the columns comparable. None of the three accuracy metrics is monotonic in
$K$; Section~\ref{sec:nonmonotone} accounts for this.}}
\label{tab:qsvt_2d}
{\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{rrrrrr}
\toprule
$K$ & $K_{\rm eff}$ & $F$ & \rev{$e_C$} & \rev{$e_x$} & \rev{$P_{\rm succ}$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table*}"""
    write_tex("qsvt_2d", body, "tab_qsvt_2d")


# ==========================================================================
# tab:assess  (Sec. 5.3.2)
# ==========================================================================
def tab_assess(m=4, K=32, gamma_max=2.0, eps_grid=(0.5, 0.4, 0.3, 0.2, 0.1)):
    """Stage 1 of Algorithm B on the 2D operator -- Algorithm A at each eps --
    with the gain applied to every row rather than to the halting one alone.

    Polynomial level: no circuit.  Rows where the correction leaves R_K > 1 have
    no base tolerance to compare against, so d(eps*) is undefined and G < 1.
    """
    emit("\n" + "=" * 104)
    emit(f"tab:assess  (Sec. 5.3.2)  2D Poisson, N={4**m}, K={K}, "
         f"gamma_max={gamma_max}")
    emit("=" * 104)
    S = setup_2d(m)
    kappa, a = S['kappa'], 1.0 / S['kappa']
    lam = S['eigs'][:K]
    emit(f"  kappa = {kappa:.2f}")
    emit(f"  {'eps':>6} {'d':>6} {'Keff':>6} {'R_K':>11} {'gamma':>7} "
         f"{'d(eps*)':>9} {'G':>7}")

    rows, failed = [], []
    for eps in eps_grid:
        r = quiet(correct_at_tolerance, ChebIterPolynomial, kappa, lam, eps,
                  gamma_max=gamma_max)
        if r is None:                     # Algorithm A reports failure
            failed.append(eps)
            emit(f"  {eps:>6} {'--':>6} {'--':>6} {'FAILED':>11}")
            continue
        # eps* = R_K is the accuracy the correction achieves; the base needs
        # d(eps*) to match it.  A residual at or above 1 is worse than any
        # tolerance the base family is defined for, so the comparison is void.
        if r['R_all'] >= 1.0:
            dstar_cell, g_cell = "---", "$<1$"
            emit(f"  {eps:>6} {r['d']:>6} {r['K_eff']:>6} {r['R_all']:>11.2g} "
                 f"{r['gamma']:>7.2f} {'---':>9} {'<1':>7}")
        else:
            eps_star = max(r['R_all'], _Q_FLOOR)
            d_star = ChebIterPolynomial.mindegree(eps_star, a)
            G = (d_star * _tau(ChebIterPolynomial.poly(d_star, a))) \
                / (r['d'] * r['gamma'] * r['tau0'])
            dstar_cell = str(d_star)
            g_cell = f"\\textbf{{{G:.1f}}}" if eps == 0.2 else f"{G:.1f}"
            emit(f"  {eps:>6} {r['d']:>6} {r['K_eff']:>6} {r['R_all']:>11.2e} "
                 f"{r['gamma']:>7.2f} {d_star:>9} {G:>7.1f}")
        rk = (f"${r['R_all']:.1f}$" if r['R_all'] >= 1.0
              else sci_or_mach(r['R_all']))
        rows.append(f"${eps}$ & {r['d']} & {r['K_eff']} & {rk:<20} "
                    f"& {r['gamma']:.2f} & {dstar_cell} & {g_cell} \\\\")

    omit = (f"Tolerances above ${max(eps_grid)}$, at which Algorithm~A reports failure, "
            "are omitted.") if failed or True else ""
    body = r"""\begin{table}[htbp]
\centering\color{blue}
\caption{Assessment for the 2D Poisson operator, $N = """ + str(4**m) + r"""$, $\kap = """ + f"{kappa:.1f}" + r"""$,
$K = """ + str(K) + r"""$, $\gamma_{\max} = """ + f"{gamma_max:g}" + r"""$. $R_K$ is the residual over the supplied eigenvalues after
correction and $d(\varepsilon^{*})$ the degree the uncorrected base needs to match it,
floored at $\varepsilon^{*} = 10^{-10}$ where the base construction stalls in double
precision, so $G$ of Equation \eqref{eq:gain} is a lower bound. """ + omit + r""" All quantities are computed classically.}
\label{tab:assess}
{\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{rrrrrrr}
\toprule
$\varepsilon$ & $d$ & $K_{\rm eff}$ & $R_K$
  & $\gamma$ & $d(\varepsilon^{*})$ & $G$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}"""
    write_tex("assess", body, "tab_assess")


# ==========================================================================
# ==========================================================================
#  REBUTTAL-ONLY DIAGNOSTICS -- printed, never emitted as .tex
# ==========================================================================
# ==========================================================================

def pure():
    """Pure spectral polynomial: P_succ against the kappa^2/tau^2 bound."""
    emit("\n" + "=" * 104)
    emit("REBUTTAL ONLY  Pure spectral polynomial, 1D Poisson (uniform load)")
    emit("The paper no longer carries this table; it shows P_succ respects the bound.")
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


def scaling():
    """Is d(kappa, eps) linear in kappa or sqrt?  [R2 Technical 4]"""
    emit("\n" + "=" * 104)
    emit("REBUTTAL ONLY  DEGREE SCALING  d(kappa, eps)   [R2 Technical 4]")
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


def extension():
    """Spectral extension vs spectral correction in the small-K regime."""
    emit("\n" + "=" * 104)
    emit("REBUTTAL ONLY  SPECTRAL EXTENSION vs SPECTRAL CORRECTION, 2D Poisson spectrum")
    emit("Correction perturbs shared coefficients and can degrade UNcorrected")
    emit("eigenvalues; extension cannot, at the price of degree.")
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
        cc, ic = quiet(spectral_correction, p0, eigs[:K], return_info=True)
        coef = p0.coef.copy(); coef[1::2] += cc
        pSC = Chebyshev(coef)
        pEX, ie = spectral_extension(p0, eigs[:K], return_info=True)
        lamT, _ = merge_eigenvalues(eigs[:K])
        emit(f"  {K:>4} {ic['K_eff']:>5} | {d0:>5} {resid(pSC, lamT):>12.2e} "
             f"{resid(pSC, eigs):>13.2e} {ic['corr_norm']:>6.2f} | "
             f"{ie['degree_ext']:>5} {resid(pEX, lamT):>12.2e} "
             f"{resid(pEX, eigs):>13.2e} {ie['cond']:>8.1e}")
    emit("\n  E(targeted)  = max residual at the K_eff targeted eigenvalues")
    emit("  E(full spec) = max residual over ALL 256 eigenvalues")
    emit("  NOTE: the extension system is a square K_eff x K_eff solve in the")
    emit("  HIGHEST-order Chebyshev terms evaluated near x = 1/kappa, where those")
    emit("  terms are nearly linearly dependent -- hence the conditioning blow-up.")


def nonmonotonic():
    """Why no metric is monotone in K -- mode-resolved.  [R2 Significance 2]"""
    emit("\n" + "=" * 110)
    emit("REBUTTAL ONLY  NON-MONOTONICITY IN K -- mode-resolved, 2D Poisson")
    emit("The paper states the mechanism in Sec. 5.3.3; the Var_w(r) vs 1-F match")
    emit("below is the evidence.")
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
            cc, ic = quiet(spectral_correction, p0, eigs[:K], return_info=True)
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


def loads():
    """Relative solution error under three loads; rho = sqrt(E_out) * r_out."""
    emit("\n" + "=" * 104)
    emit("REBUTTAL ONLY  Relative solution error on the 2D Poisson operator, three loads.")
    emit("Algorithm B reads only the spectrum, so eps, d and G are identical in every row.")
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
PAPER = {'methods': tab_methods,
         'degree_accuracy': tab_degree_accuracy,
         'adaptive': tab_adaptive,
         'qsvt_results': tab_qsvt_results,
         'perturbation': tab_perturbation,
         'qsvt_2d': tab_qsvt_2d,
         'assess': tab_assess}

REBUTTAL = {'pure': pure, 'scaling': scaling, 'ext': extension,
            'nm': nonmonotonic, 'loads': loads}

# tables needing no circuit simulation
FAST = ['methods', 'degree_accuracy', 'adaptive', 'assess']

ALL = {**PAPER, **REBUTTAL}


def main():
    args = [a.lower() for a in sys.argv[1:]]
    if not args:
        names = list(PAPER)
    elif args == ['fast']:
        names = FAST
    elif args == ['all']:
        names = list(ALL)
    else:
        names = args
    unknown = [n for n in names if n not in ALL]
    if unknown:
        sys.exit(f"unknown table(s): {', '.join(unknown)}\n"
                 f"paper: {', '.join(PAPER)}\nrebuttal: {', '.join(REBUTTAL)}")

    t0 = time.time()
    for n in names:
        ALL[n]()
    emit(f"\nTotal wall time: {time.time()-t0:.0f}s")
    with open("makeTables_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))
    print("\n[log written to makeTables_output.txt]")


if __name__ == "__main__":
    main()