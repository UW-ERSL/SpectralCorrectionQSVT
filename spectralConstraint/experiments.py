"""
experiments.py -- generates every table and figure of Section 5.

Run:  python3 experiments.py            (writes tables/*.tex and figs/*.pdf)

All quantities are computed from the polynomials and the exact eigendecomposition of the
finite-difference operator.  For a Hermitian A with spectrum {lam_k} and eigenvectors
{v_k}, the post-selected QSVT output is exactly x_QSVT = p(A) b, so F, e_C, e_x, tau and
P_succ are available in closed form; statevector simulation reproduces them and is not
needed to produce the tables.

Conventions
-----------
A is the block-encoded operator, spectrum in [a,1], a = 1/kappa.
r(x)      = x p(x) - 1                      residual
eps(d)    = minimax value of the base at degree d
t         = LP optimum                       LOWER bound on sup|r|, never quoted
tbar      = Ehlich-Zeller certificate        UPPER bound on sup|r|, always quoted
ISO-GUARANTEE: the degree at which tbar(p_SC) <= eps(d_ref) of the reference base.
"""
import os, json
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from base import ChebIterPolynomial as CI
from constrainedMinMax import constrained_minimax, certify, apply_correction

os.makedirs("tables", exist_ok=True); os.makedirs("figs", exist_ok=True)
XS = np.linspace(-1.0, 1.0, 60001)
EPS_REF = 0.2
MULTS = (1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50,
         1.60, 1.70, 1.80, 2.00)


# ----------------------------------------------------------------- spectra
def spectrum(kind, n=16):
    """Return (lam normalised into (0,1], V or None, label)."""
    if kind == "1D":
        h = 1.0/(n+1); k = np.arange(1, n+1)
        raw = 2.0 - 2.0*np.cos(k*np.pi*h)
        S = np.sqrt(2.0*h)*np.sin(np.outer(k, k)*np.pi*h)      # eigenvectors
        return raw/raw[-1], S, f"1D $n={n}$"
    if kind == "2D":
        h = 1.0/(n+1); k = np.arange(1, n+1)
        d1 = 2.0 - 2.0*np.cos(k*np.pi*h)
        raw = (d1[:, None] + d1[None, :]).ravel()
        o = np.argsort(raw)
        S = np.sqrt(2.0*h)*np.sin(np.outer(k, k)*np.pi*h)
        return raw[o]/raw[o][-1], (S, o), f"2D $n={n}$"
    if kind == "Weyl3":
        kap = 1.0/spectrum("1D", n)[0][0]
        k = np.arange(1, n+1, dtype=float)**(2.0/3.0)
        lam = (k - k[0])/(k[-1] - k[0])*(1.0 - 1.0/kap) + 1.0/kap
        return lam, None, "Weyl 3"
    raise ValueError(kind)


def loads(kind, dim, n, S):
    """beta = V^T b, normalised, in the same order as the spectrum."""
    if dim == 1:
        b = {"uniform": np.ones(n),
             "point": np.eye(n)[n//2],
             "random": np.random.default_rng(0).standard_normal(n)}[kind]
        beta = S.T @ b
    else:
        Sm, o = S
        B = {"uniform": np.ones((n, n)),
             "point": np.zeros((n, n)),
             "random": np.random.default_rng(0).standard_normal((n, n))}[kind]
        if kind == "point": B[n//2, n//2] = 1.0
        beta = (Sm.T @ B @ Sm).ravel()[o]
    return beta/np.linalg.norm(beta)


# ----------------------------------------------------------------- metrics
def metrics(p, lam, beta):
    """Exact QSVT metrics.  x_QSVT = p(A)b, x = A^{-1}b, both in the eigenbasis."""
    x = beta/lam
    xq = p(lam)*beta
    tau = float(np.max(np.abs(p(XS))))
    P = float(np.sum(xq**2)/tau**2)
    F = float((np.dot(x, xq)/(np.linalg.norm(x)*np.linalg.norm(xq)))**2)
    eC = float(abs(np.dot(beta, xq) - np.dot(beta, x))/abs(np.dot(beta, x)))
    ex = float(np.linalg.norm(xq - x)/np.linalg.norm(x))
    return dict(tau=tau, P=P, F=F, eC=eC, ex=ex)


def minnorm(p0, lam_hat, n0):
    j = np.arange(n0)
    B = np.cos(np.outer(np.arccos(np.clip(lam_hat, -1+1e-14, 1-1e-14)), 2*j+1))
    dc = np.linalg.pinv(lam_hat[:, None]*B, rcond=1e-12) @ (1.0 - lam_hat*p0(lam_hat))
    return apply_correction(p0, dc), float(np.linalg.norm(dc))


def project_equalities(p, lam_e, n0):
    """The LP satisfies its equality rows only to the solver's feasibility tolerance
    (~1e-8 in HiGHS), which caps the achievable accuracy when K is large.  We therefore
    take the LP solution for its shape and remove the remaining residual at the supplied
    points by a minimum-norm step, which is the smallest change that restores exact
    interpolation.  The certificate is recomputed afterwards, so nothing is assumed."""
    j = np.arange(n0)
    B = np.cos(np.outer(np.arccos(np.clip(lam_e, -1+1e-14, 1-1e-14)), 2*j+1))
    dc = np.linalg.pinv(lam_e[:, None]*B, rcond=1e-12) @ (1.0 - lam_e*p(lam_e))
    return apply_correction(p, dc)


def lp_solution(kappa, lam_hat, d):
    """Constrained-minimax polynomial at degree d, with the equalities projected exactly.
    Returns (p, K_eff, t_lp) or None."""
    a = 1.0/kappa
    r = constrained_minimax(CI, kappa, lam_hat, d, variant="full", c_grid=(0.0,))
    if r is None:
        return None
    # NB constrained_minimax clips the supplied values to a*(1+1e-9); projecting onto the
    # clipped points would leave a residual of order |(xp)'| * a * 1e-9 ~ 1e-9 at lambda_1.
    # Project onto the values as given.
    lam_e = dedup(np.clip(np.asarray(lam_hat, float), a, 1.0))
    p = project_equalities(r["p"], lam_e, len(r["p"].coef[1::2]))
    return p, r["K_eff"], float(r["t_lp"])


def dedup(lam, tol=1e-12):
    lam = np.sort(np.asarray(lam, float))
    keep = [lam[0]]
    for v in lam[1:]:
        if v - keep[-1] > tol*max(1.0, abs(keep[-1])): keep.append(v)
    return np.array(keep)


def iso_degree(kappa, lam_hat, d_ref, eps_ref, method="minmax", mults=MULTS):
    """Smallest degree on the grid at which the certificate meets eps_ref."""
    a = 1.0/kappa
    for m in mults:
        d = int(d_ref*m) | 1
        p0 = CI.poly(d, a); n0 = len(p0.coef[1::2])
        if method == "minmax":
            r = lp_solution(kappa, lam_hat, d)
            if r is None: continue
            p, keff, extra = r[0], r[1], dict(t_lp=r[2])
        else:
            lam_e = dedup(lam_hat)
            p, nrm = minnorm(p0, lam_e, n0)
            keff, extra = len(lam_e), dict(dc=nrm)
        tbar = certify(p, [(a, 1.0)], d+1)
        if tbar <= eps_ref:
            return dict(d=d, mult=d/d_ref, p=p, p0=p0, K_eff=keff, tbar=float(tbar),
                        eps_d=float(CI.error_for_degree(d, a)), **extra)
    return None


def fmt(v, nd=2):
    if v == 0 or (isinstance(v, float) and abs(v) < 1e-14):
        return r"$O(\epsmach)$"
    e = int(np.floor(np.log10(abs(v))))
    return f"${v/10**e:.{nd}f}\\times10^{{{e}}}$" if abs(e) > 2 else f"${v:.{nd+2}g}$"


import render   # LaTeX rendering lives in render.py so tables can be rebuilt
                 # from summary.json without recomputing anything


def write(name, body):
    """Superseded by render.py; kept so the computation functions run unchanged."""
    pass


# =============================================================== Table 1
def table_degree(n=16):
    """5.1.2  Degree-accuracy: at fixed d the constrained value cannot beat eps(d);
    raising d brings the certificate down to the reference tolerance."""
    lam, S, _ = spectrum("1D", n); kappa = 1.0/lam[0]; a = lam[0]
    lam_hat = lam.copy()
    d_ref = CI.mindegree(EPS_REF, a)
    rows = []
    for m in (1.00, 1.15, 1.30, 1.45, 1.60):
        d = int(d_ref*m) | 1
        p, keff, t_lp = lp_solution(kappa, lam_hat, d)
        a_ = a
        tau_ref = float(np.max(np.abs(CI.poly(d_ref, a)(XS))))
        tau = float(np.max(np.abs(p(XS))))
        rows.append(dict(d=d, mult=d/d_ref, eps=float(CI.error_for_degree(d, a_)),
                         t=t_lp, tbar=float(certify(p, [(a_, 1.0)], d+1)),
                         RN=float(np.max(np.abs(lam*p(lam)-1.0))),
                         gamma=tau/tau_ref, Q=(d*tau)/(d_ref*tau_ref)))
    b = [r"\begin{tabular}{rrrrrr}", r"\toprule",
         r"$d$ & $d/d_{\rm ref}$ & $\varepsilon(d)$ & $t$ & $\bar t$ & $R_N$ \\",
         r"\midrule"]
    for r in rows:
        b.append(f"${r['d']}$ & ${r['mult']:.2f}$ & {fmt(r['eps'])} & {fmt(r['t'])} & "
                 f"{fmt(r['tbar'])} & {fmt(r['RN'])} \\\\")
    b += [r"\bottomrule", r"\end{tabular}"]
    write("degree_accuracy", "\n".join(b))
    return dict(kappa=kappa, d_ref=d_ref, rows=rows)


# =============================================================== Table 2
def table_iso():
    """5.1.3  Degree price at fixed guarantee, across spectra."""
    cases = [("1D",    spectrum("1D", 16),    16),
             ("Weyl3", spectrum("Weyl3", 16), 16),
             ("2D",    spectrum("2D", 16),    32)]
    b = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"spectrum & $K$ & $K_{\rm eff}$ & $d_{\rm ref}$ & $d$ & $d/d_{\rm ref}$ "
         r"& $\bar t$ & $\gamma$ \\", r"\midrule"]
    out = []
    for tag, (lam, S, lbl), K in cases:
        kappa = 1.0/lam[0]; a = lam[0]
        d_ref = CI.mindegree(EPS_REF, a)
        eps_ref = float(CI.error_for_degree(d_ref, a))
        res = iso_degree(kappa, lam[:K], d_ref, eps_ref)
        tau0 = float(np.max(np.abs(CI.poly(d_ref, a)(XS))))
        gam = float(np.max(np.abs(res["p"](XS))))/tau0
        out.append(dict(tag=tag, kappa=kappa, K=K, eps_ref=eps_ref, gamma=gam, **{
            k: v for k, v in res.items() if k != "p" and k != "p0"}))
        b.append(f"{lbl} & ${K}$ & ${res['K_eff']}$ & ${d_ref}$ & ${res['d']}$ & "
                 f"${res['mult']:.2f}$ & {fmt(res['tbar'])} & ${gam:.2f}$ \\\\")
    b += [r"\bottomrule", r"\end{tabular}"]
    write("isoguarantee", "\n".join(b))
    return out


# =============================================================== Table 3
def table_1d(n=16):
    """5.2.1  1D, K = N, full metrics against the reference base and a tight base."""
    lam, S, _ = spectrum("1D", n); kappa = 1.0/lam[0]; a = lam[0]
    d_ref = CI.mindegree(EPS_REF, a); eps_ref = float(CI.error_for_degree(d_ref, a))
    res = iso_degree(kappa, lam, d_ref, eps_ref)
    p_ref = CI.poly(d_ref, a)
    b = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Method & $d$ & $F$ & $e_C$ & $e_x$ & $P_{\rm succ}$ & $\mathcal{Q}$ \\"]
    out = {}
    for load in ("uniform", "point"):
        beta = loads(load, 1, n, S)
        # base degree that MATCHES the achieved solution error of p_SC
        ex_sc = metrics(res["p"], lam, beta)["ex"]
        d_tight = CI.mindegree(max(ex_sc, 1e-10), a)
        p_tight = CI.poly(d_tight, a)
        b += [r"\midrule",
              r"\multicolumn{7}{l}{\textit{" +
              ("Uniform load" if load == "uniform" else "Point load at midpoint") +
              r"}} \\", r"\midrule"]
        for tag, p, d in ((r"$p_{SC}$, iso-guarantee", res["p"], res["d"]),
                          (r"$p_0$, same guarantee", p_ref, d_ref),
                          (r"$p_0$, same $e_x$", p_tight, d_tight)):
            m = metrics(p, lam, beta); Q = d/np.sqrt(m["P"])
            out[f"{load}|{tag}"] = dict(d=d, Q=Q, **m)
            b.append(f"{tag} & ${d}$ & ${m['F']:.6f}$ & {fmt(m['eC'])} & {fmt(m['ex'])} "
                     f"& ${m['P']:.4f}$ & ${Q:.1f}$ \\\\")
    b += [r"\bottomrule", r"\end{tabular}"]
    write("qsvt_1d", "\n".join(b))
    return dict(kappa=kappa, d_ref=d_ref, d=res["d"], mult=res["mult"],
                tbar=res["tbar"], eps_ref=eps_ref, rows=out)


# =============================================================== Table 4
def table_perturb(n=16, ntrial=20):
    """5.2.2  Robustness: perturbed eigenvalues at the iso-guarantee degree."""
    lam, S, _ = spectrum("1D", n); kappa = 1.0/lam[0]; a = lam[0]
    d_ref = CI.mindegree(EPS_REF, a); eps_ref = float(CI.error_for_degree(d_ref, a))
    d = iso_degree(kappa, lam, d_ref, eps_ref)["d"]
    beta = loads("uniform", 1, n, S)
    rng = np.random.default_rng(0)
    b = [r"\begin{tabular}{rrrrrr}", r"\toprule",
         r"$\eta$ & $\bar t$ & $1-F$ & $e_C$ & $e_x$ & $P_{\rm succ}$ \\", r"\midrule"]
    out = []
    for eta in (0.0, 0.01, 0.05, 0.10, 0.20):
        acc = []
        for _ in range(1 if eta == 0 else ntrial):
            lh = lam*(1.0 + eta*rng.uniform(-1, 1, n)) if eta else lam
            r = lp_solution(kappa, lh, d)
            if r is None: continue
            p = r[0]
            acc.append((certify(p, [(a, 1.0)], d+1), metrics(p, lam, beta)))
        tb = np.mean([x[0] for x in acc])
        M = {k: np.mean([x[1][k] for x in acc]) for k in ("F", "eC", "ex", "P")}
        out.append(dict(eta=eta, tbar=float(tb), **{k: float(v) for k, v in M.items()}))
        b.append(f"${eta}$ & {fmt(tb)} & {fmt(1-M['F'])} & {fmt(M['eC'])} & "
                 f"{fmt(M['ex'])} & ${M['P']:.4f}$ \\\\")
    b += [r"\bottomrule", r"\end{tabular}"]
    write("perturbation", "\n".join(b))
    return dict(d=d, ntrial=ntrial, rows=out)


# =============================================================== Table 5
def table_2d(n=16, Ks=(1, 4, 8, 16, 32)):
    """5.3.1  2D, accuracy versus K, each row at its own iso-guarantee degree."""
    lam, S, _ = spectrum("2D", n); kappa = 1.0/lam[0]; a = lam[0]
    d_ref = CI.mindegree(EPS_REF, a); eps_ref = float(CI.error_for_degree(d_ref, a))
    beta = loads("uniform", 2, n, S)
    p_ref = CI.poly(d_ref, a); m0 = metrics(p_ref, lam, beta)
    b = [r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
         r"$K$ & $K_{\rm eff}$ & $d$ & $d/d_{\rm ref}$ & $\bar t$ & $F$ & $e_C$ "
         r"& $e_x$ & $P_{\rm succ}$ \\", r"\midrule",
         f"0 (base) & --- & ${d_ref}$ & $1.00$ & {fmt(eps_ref)} & ${m0['F']:.6f}$ & "
         f"{fmt(m0['eC'])} & {fmt(m0['ex'])} & ${m0['P']:.4f}$ \\\\"]
    out = [dict(K=0, **m0)]
    for K in Ks:
        res = iso_degree(kappa, lam[:K], d_ref, eps_ref)
        m = metrics(res["p"], lam, beta)
        out.append(dict(K=K, K_eff=res["K_eff"], d=res["d"], mult=res["mult"],
                        tbar=res["tbar"], gain=m0["ex"]/m["ex"], **m))
        b.append(f"${K}$ & ${res['K_eff']}$ & ${res['d']}$ & ${res['mult']:.2f}$ & "
                 f"{fmt(res['tbar'])} & ${m['F']:.6f}$ & {fmt(m['eC'])} & "
                 f"{fmt(m['ex'])} & ${m['P']:.4f}$ \\\\")
    b += [r"\bottomrule", r"\end{tabular}"]
    write("qsvt_2d", "\n".join(b))
    return dict(kappa=kappa, d_ref=d_ref, eps_ref=eps_ref, base=m0, rows=out)


# =============================================================== Table 6
def table_multiplier(n=16, K=32):
    """5.3.2  Choosing the degree: what the certificate and the accuracy do as the
    degree rises, and where the iso-guarantee point falls.  Both constructions."""
    lam, S, _ = spectrum("2D", n); kappa = 1.0/lam[0]; a = lam[0]
    d_ref = CI.mindegree(EPS_REF, a); eps_ref = float(CI.error_for_degree(d_ref, a))
    beta = {k: loads(k, 2, n, S) for k in ("uniform", "point", "random")}
    m0 = {k: metrics(CI.poly(d_ref, a), lam, beta[k])["ex"] for k in beta}
    tau0 = float(np.max(np.abs(CI.poly(d_ref, a)(XS))))
    b = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"construction & $d/d_{\rm ref}$ & $d$ & $\bar t$ & $\gamma$ "
         r"& uniform & point & random \\", r"\midrule"]
    out = []
    for meth, tag in (("minmax", r"$p_{SC}$ (LP)"), ("minnorm", r"$p_{SC}$ (min-norm)")):
        for m in (1.15, 1.25, 1.35, 1.50):
            d = int(d_ref*m) | 1
            p0 = CI.poly(d, a); n0 = len(p0.coef[1::2])
            if meth == "minmax":
                p = lp_solution(kappa, lam[:K], d)[0]
            else:
                p, _ = minnorm(p0, dedup(lam[:K]), n0)
            tb = certify(p, [(a, 1.0)], d+1)
            g = {k: m0[k]/metrics(p, lam, beta[k])["ex"] for k in beta}
            gam = float(np.max(np.abs(p(XS))))/tau0
            out.append(dict(method=meth, mult=m, d=d, tbar=float(tb), gamma=gam,
                            **{f"g_{k}": v for k, v in g.items()}))
            iso = r"$^{\dagger}$" if tb <= eps_ref else ""
            b.append(f"{tag if m == 1.15 else ''} & ${m:.2f}${iso} & ${d}$ & {fmt(tb)} & "
                     f"${gam:.2f}$ & ${g['uniform']:.1f}$ & ${g['point']:.1f}$ & "
                     f"${g['random']:.1f}$ \\\\")
        b.append(r"\midrule" if meth == "minmax" else "")
    b = [x for x in b if x != ""] + [r"\bottomrule", r"\end{tabular}"]
    write("multiplier", "\n".join(b))
    return dict(eps_ref=eps_ref, d_ref=d_ref, rows=out)


# =============================================================== Figure
def figure_residual(n=16, K=32):
    """Residual at the spectrum.  The continuous residual oscillates O(d) times and is
    unreadable when plotted; what governs the QSVT error is its value AT the eigenvalues,
    which is what this shows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lam, S, _ = spectrum("2D", n); kappa = 1.0/lam[0]; a = lam[0]
    d_ref = CI.mindegree(EPS_REF, a); eps_ref = float(CI.error_for_degree(d_ref, a))
    res = iso_degree(kappa, lam[:K], d_ref, eps_ref)
    d = res["d"]; p0d = CI.poly(d, a)
    pmn, _ = minnorm(p0d, dedup(lam[:K]), len(p0d.coef[1::2]))
    p_ref = CI.poly(d_ref, a)
    R = lambda p: np.abs(lam*p(lam) - 1.0) + 1e-17

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.axhline(eps_ref, ls="--", lw=0.7, color="k")
    ax.text(0.62, eps_ref*1.35, r"$\varepsilon(d_{\rm ref})$", fontsize=6)
    ax.semilogy(lam, R(p_ref), ".", ms=2.5, color="0.55", label=rf"base, $d={d_ref}$")
    ax.semilogy(lam, R(res["p"]), ".", ms=2.5, color="C0", label=rf"LP, $d={d}$")
    ax.semilogy(lam, R(pmn), ".", ms=2.5, color="C3", label=rf"min-norm, $d={d}$")
    ax.axvspan(a, lam[K-1], color="C0", alpha=0.07, lw=0)
    ax.text(lam[K-1]*1.1, 3e-15, rf"$\lambda_1\ldots\lambda_{{{K}}}$ constrained",
            fontsize=6)
    ax.set_xlim(0, 1.02); ax.set_ylim(1e-16, 3)
    ax.set_xlabel(r"$\lambda_k$")
    ax.set_ylabel(r"$|\lambda_k\,p(\lambda_k)-1|$")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.2)
    fig.savefig("figs/residualCompare.pdf"); fig.savefig("figs/residualCompare.png", dpi=220)
    print("figure written: figs/residualCompare.pdf")


if __name__ == "__main__":
    summary = {}
    summary["degree"]     = table_degree()
    summary["iso"]        = table_iso()
    summary["qsvt_1d"]    = table_1d()
    summary["perturb"]    = table_perturb()
    summary["qsvt_2d"]    = table_2d()
    summary["multiplier"] = table_multiplier()
    json.dump(summary, open("summary.json", "w"), indent=1, default=str)
    render.all_tables(json.load(open("summary.json")))
    figure_residual()
    print("\nall tables in tables/, figure in figs/, raw numbers in summary.json")
