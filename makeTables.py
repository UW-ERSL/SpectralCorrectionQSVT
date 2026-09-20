"""
makeTables.py -- single reproducible driver for every table in

    "Spectrally Constrained Polynomials for Quantum Singular Value Transformation"

Each paper table is emitted as a self-contained LaTeX float (caption, label,
tabular) into TABLES_DIR, so main.tex carries one \\input line per table and a
caption cannot drift from the numbers it describes.  The numbers the prose quotes
are written to TABLES_DIR/summary.json by the same run.

Usage
-----
    python makeTables.py                  # every paper table
    python makeTables.py ladder spectra   # selected tables
    python makeTables.py cscan            # rebuttal-only diagnostic (stdout)

Paper tables (emitted to TABLES_DIR)
------------------------------------
    ladder     tab:ladder     Sec. 5.2   2D Poisson, degree multiplier sweep
    spectra    tab:spectra    Sec. 5.3   1D, 2D (K sweep), 3D Poisson at mu = 1.5
    eigerror   tab:eigerror   Sec. 5.4   2D Poisson, perturbed eigenvalues
    inclusion  tab:inclusion  Sec. 6     stiff inclusion, coarse-mesh eigenvalues

Rebuttal-only diagnostic (stdout only)
--------------------------------------
    cscan      merge scale c = 0, 1/4, 1/2, 1 at mu = 1.5 on 2D Poisson

Conventions (stated once, used everywhere)
------------------------------------------
* Every polynomial is built by constrainedMinMax.spectral_constraint with
  eps = 0.2, c = 1/2, mu = 1.5, ChebIter base.
* Poisson operators are normalised by alpha = 1.01 lambda_max, so kappa = 117.6
  for n = 16 in every dimension.  The inclusion is normalised by its Gershgorin
  bound, the quantity a practitioner has.
* Everything is closed form in the eigenbasis: the post-selected QSVT output is
  exactly p(A) b, so e_x computed from p at the eigenvalues is the circuit's e_x.
* e_x = ||p(A)b - A^{-1}b|| / ||A^{-1}b||, on three unit loads: uniform, a unit
  point load at the central node, and a WHITE load with equal amplitude in every
  eigenmode.  The white load replaces a Gaussian random vector: for one draw, e_x
  is set by the draw's few lowest modal amplitudes, and G on 2D Poisson ranged
  from 0.79 to 5.55 over 40 seeds.  The white load's e_x is the ratio of
  expectations E||p(A)b - A^{-1}b||^2 / E||A^{-1}b||^2 over isotropic random b,
  so it is the seed-free version of the same quantity (2.85 against a median of
  2.48 over those 40 seeds).
* G = e_x(p0 at d) / e_x(p at d): accuracy ratio at EQUAL DEGREE, per load.
  gamma = tau(p)/tau(p0) is reported beside it, since equal degree is not
  equal query cost when gamma > 1.
"""
import sys, os, json, time
import numpy as np

from PolynomialApproximators import ChebIterPolynomial
from constrainedMinMax import spectral_constraint, C_MERGE, MU_DEGREE

TABLES_DIR = os.environ.get("QSVT_TABLES_DIR", "tables")
EPS = 0.2
BASE = ChebIterPolynomial
LOADS = ("uniform", "point", "white")
SUMMARY = {}


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
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, name + ".tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_BANNER.format(fn=fn, name=name))
        f.write(body.rstrip() + "\n")
    print(f"  [wrote {path}]")


def sci(x, nd=1):
    """1.97e-1 -> $1.97\\times10^{-1}$"""
    m, e = f"{x:.{nd}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def num(x):
    """Gain / ratio cell: three significant figures, math mode, sci above 1e4."""
    if x >= 1e4:
        return sci(x, 1)
    if x >= 100:
        return f"${x:.0f}$"
    if x >= 10:
        return f"${x:.1f}$"
    return f"${x:.2f}$"


def load_cells(G):
    return " & ".join(num(G[L]) for L in LOADS)


# ==========================================================================
# Operators, loads, errors
# ==========================================================================
def poisson_modes(dim, n):
    """Eigenvalues and modal load amplitudes of the dim-dimensional 5/7-point
    Dirichlet Laplacian on n^dim interior nodes, without forming the matrix.

    The eigenvectors are tensor products of the 1D sine modes, so the uniform and
    the central point load factorise; the white load is flat in the eigenbasis.
    """
    h = 1.0 / (n + 1)
    k = np.arange(1, n + 1)
    l1 = 4.0 / h ** 2 * np.sin(k * np.pi * h / 2) ** 2
    S = np.sqrt(2 * h) * np.sin(np.outer(k, k) * np.pi * h)     # S[mode, node]
    u1, p1 = S @ np.ones(n), S[:, n // 2]
    lam, u, p = l1, u1, p1
    for _ in range(dim - 1):
        lam = np.add.outer(lam, l1).ravel()
        u = np.outer(u, u1).ravel()
        p = np.outer(p, p1).ravel()
    N = lam.size
    beta = {"uniform": u / np.sqrt(N), "point": p,
            "white": np.ones(N) / np.sqrt(N)}
    o = np.argsort(lam, kind="stable")
    alpha = 1.01 * lam.max()
    return lam[o] / alpha, {L: b[o] for L, b in beta.items()}


def ex(p, lam, beta):
    """Relative solution error of p(A)b against A^{-1}b, in the eigenbasis."""
    w = (beta / lam) ** 2
    r = lam * p(lam) - 1.0
    return float(np.sqrt(np.sum(w * r * r) / np.sum(w)))


def gains(out, lam, beta):
    """G per load, and the two e_x values it is formed from."""
    G, E = {}, {}
    for L in LOADS:
        e1, e0 = ex(out["p"], lam, beta[L]), ex(out["p0"], lam, beta[L])
        G[L], E[L] = e0 / max(e1, 1e-300), (e0, e1)
    return G, E


def run(kappa, lam_supplied, lam_true, beta, **kw):
    t0 = time.time()
    out = spectral_constraint(BASE, kappa, lam_supplied, epsilon=EPS, **kw)
    G, E = gains(out, lam_true, beta)
    out.update(G=G, E=E, secs=time.time() - t0)
    return out


def show(tag, o):
    print(f"  {tag:<22} d={o['d']:>5} ({o['mu']:.2f}x)  K_eff={o['K_eff']:>3}  "
          f"t={o['t']:.4f} (t0 {o['t0']:.4f}) {'ok ' if o['certified'] else 'NO '} "
          f"gamma={o['gamma']:.3f}  G=" +
          " / ".join(f"{o['G'][L]:.3g}" for L in LOADS) + f"  [{o['secs']:.0f}s]")


# ==========================================================================
# tab:ladder  (Sec. 5.2)
# ==========================================================================
def tab_ladder(n=16, K=32, mults=(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0)):
    """2D Poisson: the certificate and the gain as the degree is raised."""
    print(f"\ntab:ladder  2D Poisson n={n}, K={K}, c={C_MERGE}")
    lam, beta = poisson_modes(2, n)
    kappa = 1.0 / lam[0]
    rows, S = [], []
    for m in mults:
        o = run(kappa, lam[:K], lam, beta, mu=m)
        show(f"mu={m}", o)
        S.append(dict(mu=m, d=o["d"], K_eff=o["K_eff"], t=o["t"], gamma=o["gamma"],
                      G=o["G"], certified=o["certified"]))
        cells = [f"${m:.2f}$", f"${o['d']}$", f"${o['K_eff']}$", f"${o['t']:.3f}$",
                 f"${o['gamma']:.2f}$", load_cells(o["G"])]
        if m == MU_DEGREE:                      # the operating point, marked
            cells[0] = cells[0][:-1] + r"^{*}$"
        rows.append(" & ".join(cells) + r" \\")
    t0, d0 = o["t0"], o["d0"]
    SUMMARY["ladder"] = dict(n=n, N=n * n, K=K, kappa=kappa, d0=d0, t0=t0, rows=S)

    body = r"""\begin{table}[htbp]
\centering
\caption{The constrained polynomial on the 2D Poisson operator ($N = """ + str(n * n) + r"""$,
$\kap = """ + f"{kappa:.1f}" + r"""$, $K = """ + str(K) + r"""$) as its degree is raised from the base degree
$d_0 = d(""" + f"{EPS:g}" + r""") = """ + str(d0) + r"""$. Read the certificate $t$ against the base's
$t_0 = """ + f"{t0:.3f}" + r"""$: rows with $t > t_0$ carry a weaker guarantee than the base polynomial
the caller already has. $G$ compares solution errors with the base \emph{at the same degree}, so it
excludes the accuracy the extra degree buys the base. The asterisk marks $\mu = """ + f"{MU_DEGREE:g}" + r"""$, the
operating point used everywhere else.}
\label{tab:ladder}
{\small\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{rrrrrrrr}
\toprule
 & & & & & \multicolumn{3}{c}{$G$} \\
\cmidrule(lr){6-8}
$\mu$ & $d$ & $K_{\rm eff}$ & $t$ & $\gamma$ & unif. & point & white \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}"""
    write_tex("ladder", body, "tab_ladder")


# ==========================================================================
# tab:spectra  (Sec. 5.3)
# ==========================================================================
def tab_spectra(n=16, cases=((1, 16), (2, 8), (2, 16), (2, 32), (2, 64),
                             (3, 16), (3, 32))):
    """1D, 2D and 3D Poisson with the same kappa, at mu = 1.5."""
    print(f"\ntab:spectra  Poisson n={n}, mu={MU_DEGREE}, c={C_MERGE}")
    rows, S, cache = [], [], {}
    prev = None
    for dim, K in cases:
        if dim not in cache:
            cache[dim] = poisson_modes(dim, n)
        lam, beta = cache[dim]
        kappa = 1.0 / lam[0]
        o = run(kappa, lam[:K], lam, beta)
        distinct = len(np.unique(np.round(lam[:K], 12)))
        show(f"{dim}D K={K}", o)
        S.append(dict(dim=dim, N=lam.size, K=K, distinct=distinct, kappa=kappa,
                      d=o["d"], K_eff=o["K_eff"], t=o["t"], t0=o["t0"],
                      gamma=o["gamma"], G=o["G"], certified=o["certified"]))
        if prev is not None and prev != dim:
            rows.append(r"\midrule")
        prev = dim
        rows.append(f"{dim}D & ${lam.size}$ & ${K}$ & ${distinct}$ & ${o['K_eff']}$ & "
                    f"${o['t']:.3f}$ & ${o['gamma']:.2f}$ & {load_cells(o['G'])} \\\\")
    SUMMARY["spectra"] = dict(n=n, d=o["d"], d0=o["d0"], t0=o["t0"], rows=S)

    body = r"""\begin{table*}[t]
\centering
\caption{The constrained polynomial on Poisson operators of dimension one to three, each with
$n = """ + str(n) + r"""$ nodes per direction and hence the same $\kap = """ + f"{S[0]['kappa']:.1f}" + r"""$, at
$d = """ + str(o['d']) + r"""$ ($\mu = """ + f"{MU_DEGREE:g}" + r"""$). The $K$ smallest eigenvalues are supplied;
``distinct'' counts the different values among them, and $K_{\rm eff}$ the number retained after merging,
so a gap between the two is a loss and a gap between $K$ and ``distinct'' is not. Every certificate
is below the base's $t_0 = """ + f"{o['t0']:.3f}" + r"""$. $G$ is the solution-error ratio against the base
at the same degree; a value below one means the base is the more accurate polynomial for that load.
In 1D, $K = N$ and the constrained polynomial is exact on the whole spectrum, so $G$ there measures
only the base's error.}
\label{tab:spectra}
{\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrrrrrr}
\toprule
 & & & & & & & \multicolumn{3}{c}{$G$} \\
\cmidrule(lr){8-10}
 & $N$ & $K$ & distinct & $K_{\rm eff}$ & $t$ & $\gamma$ & uniform & point & white \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table*}"""
    write_tex("spectra", body, "tab_spectra")


# ==========================================================================
# tab:eigerror  (Sec. 5.4)
# ==========================================================================
def tab_eigerror(n=16, K=32, etas=(0.0, 0.01, 0.05, 0.10), trials=10, seed=42):
    """2D Poisson with supplied eigenvalues lam_k (1 + eta u_k), u_k ~ U(-1,1)."""
    print(f"\ntab:eigerror  2D Poisson n={n}, K={K}, {trials} trials, seed={seed}")
    lam, beta = poisson_modes(2, n)
    kappa = 1.0 / lam[0]
    rng = np.random.default_rng(seed)
    rows, S = [], []
    for eta in etas:
        outs = []
        for _ in range(1 if eta == 0 else trials):
            lh = lam[:K] * (1.0 + eta * rng.uniform(-1, 1, K))
            outs.append(run(kappa, lh, lam, beta))
        ke = [o["K_eff"] for o in outs]
        tmax = max(o["t"] for o in outs)
        ncert = sum(o["certified"] for o in outs)
        med = {L: float(np.median([o["G"][L] for o in outs])) for L in LOADS}
        mn = {L: float(min(o["G"][L] for o in outs)) for L in LOADS}
        print(f"  eta={eta:<5} K_eff {min(ke)}-{max(ke)}  t_max={tmax:.4f}  "
              f"certified {ncert}/{len(outs)}  G median " +
              " / ".join(f"{med[L]:.3g}" for L in LOADS) + "  min " +
              " / ".join(f"{mn[L]:.3g}" for L in LOADS))
        S.append(dict(eta=eta, K_eff=[min(ke), max(ke)], t_max=tmax,
                      certified=ncert, trials=len(outs), G_median=med, G_min=mn))
        kcell = f"${min(ke)}$" if min(ke) == max(ke) else f"${min(ke)}$--${max(ke)}$"
        if eta == 0:
            rows.append(f"${eta * 100:g}\\%$ & {kcell} & ${tmax:.3f}$ & -- & "
                        f"{load_cells(med)} \\\\")
        else:
            rows.append(f"${eta * 100:g}\\%$ & {kcell} & ${tmax:.3f}$ & median & "
                        f"{load_cells(med)} \\\\")
            rows.append(f" & & & min & {load_cells(mn)} \\\\")
    allcert = all(r["certified"] == r["trials"] for r in S)
    certtxt = ("every draw certifies" if allcert else
               "draws that fail to certify are counted in the summary file")
    SUMMARY["eigerror"] = dict(n=n, K=K, trials=trials, seed=seed, d=outs[0]["d"],
                               t0=outs[0]["t0"], rows=S)

    body = r"""\begin{table}[htbp]
\centering
\caption{The constrained polynomial on the 2D Poisson operator ($K = """ + str(K) + r"""$,
$d = """ + str(outs[0]['d']) + r"""$) when each supplied eigenvalue carries a relative error drawn uniformly from
$[-\eta, \eta]$, over """ + str(trials) + r""" draws per level. $t$ is the largest certificate over the draws,
to be read against the base's $t_0 = """ + f"{outs[0]['t0']:.3f}" + r"""$; """ + certtxt + r""". $G$ is given as the median
and the smallest over the draws; a value below one is a draw in which the base polynomial
was the more accurate. The error changes how much the constraints help, not whether the guarantee
holds.}
\label{tab:eigerror}
{\small\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{rrrlrrr}
\toprule
 & & & & \multicolumn{3}{c}{$G$} \\
\cmidrule(lr){5-7}
$\eta$ & $K_{\rm eff}$ & $t$ & & unif. & point & white \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}"""
    write_tex("eigerror", body, "tab_eigerror")


# ==========================================================================
# tab:inclusion  (Sec. 6)
# ==========================================================================
def inclusion_operator(n, rho, frac=1.0 / 3.0):
    """5-point FD for -div(c grad u) on the unit square, c = rho on a centred
    square inclusion of LINEAR fraction `frac` counted in elements: the centred
    round(frac*(n+1)) nodes per direction.  With frac = 1/3 and n+1 a multiple of
    3 (n = 8, 11, 14, 17) the volume fraction is exactly 1/9 on every mesh."""
    m = int(round(frac * (n + 1)))
    lo = (n - m) // 2
    h = 1.0 / (n + 1)
    ins = np.zeros(n, bool)
    ins[lo:lo + m] = True
    c = np.where(np.outer(ins, ins), rho, 1.0)
    A = np.zeros((n * n, n * n))
    for i in range(n):
        for j in range(n):
            p, diag = i * n + j, 0.0
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                inside = 0 <= ii < n and 0 <= jj < n
                cf = 0.5 * (c[i, j] + c[ii, jj]) if inside else c[i, j]
                diag += cf
                if inside:
                    A[p, ii * n + jj] = -cf
            A[p, p] = diag
    return A / h ** 2


def tab_inclusion(nfine=17, rho=4.0, K=32, ncs=(8, 11, 14), safety=1.05):
    """Stiff inclusion: eigenvalues and kappa from a coarse mesh, alpha from
    Gershgorin on the fine matrix.  No eigensolve on the fine operator is used to
    build any row but the last.

    kappa_hat = safety * alpha / lambda_1(coarse).  The safety factor is needed.
    For this operator lambda_1 DECREASES under refinement (20.70, 20.57, 20.50,
    20.46 on n = 8, 11, 14, 17), so a coarse mesh OVERestimates lambda_1 and
    alpha/lambda_1(coarse) is 0.2-1.2% BELOW the true kappa: [a, 1] would then
    miss the smallest eigenvalue and the certificate would not cover it.  Until
    2026-09-17 this table normalised the coarse estimate by the Gershgorin bound
    and the true kappa by 1.01 lambda_max, which differ by 5%, and that mismatch
    made kappa_hat look 5% HIGH.  A 5% factor covers the observed 1.2% and costs
    5% in degree.  The Collatz-Wielandt lower bound min_i (Ax)_i/x_i was tried with
    sine and prolonged coarse eigenvectors and is negative at the interface nodes.
    """
    print(f"\ntab:inclusion  rho={rho}, fine {nfine}x{nfine}, K={K}")
    A = inclusion_operator(nfine, rho)
    w_raw, V = np.linalg.eigh(A)
    alpha = float(np.max(np.abs(A).sum(axis=1)))          # Gershgorin, >= ||A||
    w = w_raw / alpha
    N = nfine * nfine
    pt = np.zeros(N); pt[(nfine // 2) * nfine + nfine // 2] = 1.0
    beta = {"uniform": V.T @ (np.ones(N) / np.sqrt(N)), "point": V.T @ pt,
            "white": np.ones(N) / np.sqrt(N)}
    kappa_true = 1.0 / w[0]
    print(f"  N={N}, alpha/lambda_max = {alpha / w_raw.max():.4f}, "
          f"true kappa = {kappa_true:.1f}")

    srcs = []
    for nc in ncs:
        wc = np.sort(np.linalg.eigvalsh(inclusion_operator(nc, rho)))
        srcs.append((f"${nc}\\times{nc}$", f"{nc}x{nc}", nc * nc,
                     wc[:K] / alpha, safety * alpha / wc[0]))
    srcs.append((r"\emph{exact}", "exact", N, w[:K], kappa_true))

    rows, S = [], []
    for tex, tag, dof, lam_s, kap in srcs:
        err = 100 * float(np.max(np.abs(lam_s - w[:K]) / w[:K]))
        covers = 1.0 / kap <= w[0] * (1 + 1e-12)
        o = run(kap, lam_s, w, beta)
        show(tag, o)
        print(f"      err={err:.1f}%  kappa_hat={kap:.1f}  interval covers spectrum: {covers}")
        S.append(dict(source=tag, dof=dof, err=err, kappa=kap, covers=bool(covers),
                      d0=o["d0"], d=o["d"], K_eff=o["K_eff"], t=o["t"], t0=o["t0"],
                      gamma=o["gamma"], G=o["G"],
                      E={L: list(o["E"][L]) for L in LOADS},
                      certified=o["certified"], secs=o["secs"]))
        if tag == "exact":
            rows.append(r"\midrule")
        rows.append(f"{tex} & ${dof}$ & ${err:.1f}$ & ${kap:.0f}$ & ${o['d']}$ & "
                    f"${o['K_eff']}$ & ${o['t']:.3f}$ & ${o['t0']:.3f}$ & "
                    f"${o['gamma']:.2f}$ & {load_cells(o['G'])} \\\\")
    SUMMARY["inclusion"] = dict(nfine=nfine, N=N, rho=rho, K=K,
                                alpha_over_lmax=alpha / w_raw.max(),
                                kappa_true=kappa_true, rows=S)

    body = r"""\begin{table*}[t]
\centering
\caption{The constrained polynomial on the stiff-inclusion operator ($\rho = """ + f"{rho:g}" + r"""$,
$N = """ + str(N) + r"""$, true $\kap = """ + f"{kappa_true:.0f}" + r"""$), with the $K = """ + str(K) + r"""$ smallest
eigenvalues and the estimate $\hat\kap$ taken from a coarse discretization of the same problem
with the stated number of unknowns, $\hat\kap$ including a factor """ + f"{safety:g}" + r""" (Section~\ref{sec:inclusion_setup}). ``err'' is the largest relative error among the supplied
eigenvalues. Each row is built at its own $\hat\kap$, so $d = """ + f"{MU_DEGREE:g}" + r"""\,d(""" + f"{EPS:g}" + r""")$ and the base's
certificate $t_0$ differ slightly between rows; read each $t$ against the $t_0$ beside it. $G$ is
the solution-error ratio against the base at the same degree and the same $\hat\kap$. The last row
uses the true spectrum, which a practitioner does not have.}
\label{tab:inclusion}
{\small\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lrrrrrrrrrrr}
\toprule
 & & & & & & & & & \multicolumn{3}{c}{$G$} \\
\cmidrule(lr){10-12}
coarse mesh & unknowns & err\,\% & $\hat\kap$ & $d$ & $K_{\rm eff}$ & $t$ & $t_0$ & $\gamma$
  & uniform & point & white \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table*}"""
    write_tex("inclusion", body, "tab_inclusion")


# ==========================================================================
# Rebuttal-only
# ==========================================================================
def cscan(n=16, K=32, cs=(0.0, 0.25, 0.5, 1.0), etas=(0.0, 0.01)):
    """Why c = 1/2: the merge scale at mu = 1.5 on 2D Poisson."""
    print(f"\nREBUTTAL ONLY  merge scale at mu = {MU_DEGREE}, 2D Poisson K={K}")
    lam, beta = poisson_modes(2, n)
    kappa = 1.0 / lam[0]
    for eta in etas:
        lh = lam[:K] * (1.0 + eta * np.random.default_rng(1).uniform(-1, 1, K))
        for c in cs:
            show(f"eta={eta} c={c}", run(kappa, lh, lam, beta, c=c))


# ==========================================================================
PAPER = {"ladder": tab_ladder, "spectra": tab_spectra,
         "eigerror": tab_eigerror, "inclusion": tab_inclusion}
REBUTTAL = {"cscan": cscan}
ALL = {**PAPER, **REBUTTAL}


def main():
    names = [a.lower() for a in sys.argv[1:]] or list(PAPER)
    bad = [n for n in names if n not in ALL]
    if bad:
        sys.exit(f"unknown: {', '.join(bad)}\npaper: {', '.join(PAPER)}\n"
                 f"rebuttal: {', '.join(REBUTTAL)}")
    t0 = time.time()
    path = os.path.join(TABLES_DIR, "summary.json")
    if os.path.exists(path):
        with open(path) as f:
            SUMMARY.update(json.load(f))
    for n in names:
        ALL[n]()
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(SUMMARY, f, indent=1, default=float)
    print(f"\nTotal wall time: {time.time() - t0:.0f}s   [summary -> {path}]")


if __name__ == "__main__":
    main()