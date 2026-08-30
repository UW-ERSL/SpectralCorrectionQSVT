"""
costAccuracy.py -- cost-accuracy trade-off curves, 1D Poisson.

For each base family and each base tolerance eps:
    x  = achieved accuracy, max_k |lam_k p(lam_k) - 1| over the spectrum
    y  = total query cost   Q = d / sqrt(P_succ)
The base polynomial and its spectrally corrected counterpart (K = N) are
plotted on the same axes, so the vertical gap at a common x is the projected
saving G.  Bottom-left is accurate and cheap.

    python costAccuracy.py            # writes figs/costAccuracy.pdf
"""
import io, os, contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev

from PolynomialApproximators import (ChebIterPolynomial as CI, MangPolynomial as MA,
                                     SunderhaufPolynomial as SU,
                                     spectral_correction_adaptive)
from PoissonFunctions import build_1d_poisson, eigs_1d_poisson

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.size"] = 9

FIGDIR = "figs"
FIGFMT = os.environ.get("FIGFMT", "pdf").lower()
C_REL, C_L2, C_ABS = "#1f77b4", "#d62728", "#9467bd"
L_REL = r"$L_\infty^{\mathrm{rel}}$"
L_L2 = r"$L_2^{\mathrm{rel}}$"
L_ABS = r"$L_\infty^{\mathrm{abs}}$"
SURFACE, GRIDCOL, SPINECOL, INK, MUTED = "#f7f7f7", "white", "#cccccc", "#222222", "#444444"

M = 4
EPS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 3e-3, 1e-3]
BASES = [(L_REL, CI, C_REL, "-", "o")]


def setup():
    A, b = build_1d_poisson(M)
    eig = eigs_1d_poisson(M)
    s = 1.01 * eig.max()
    eig, A = eig / s, A / s
    w, V = np.linalg.eigh(A)
    return eig, w, V.T @ b, 1.0 / float(eig[0])


def cost(p, d, w, beta, xs):
    """Q = d / sqrt(P_succ) with P_succ = ||p(A)b||^2 / tau^2."""
    tau = float(np.max(np.abs(p(xs))))
    nrm = float(np.linalg.norm(p(w) * beta))
    return d / np.sqrt(nrm ** 2 / tau ** 2), tau


def sweep():
    eig, w, beta, kappa = setup()
    a = 1.0 / kappa
    xs = np.linspace(-1.0, 1.0, 60001)
    out = {}
    for label, cls, *_ in BASES:
        rows = []
        for e in EPS:
            d = cls.mindegree(e, a)
            p0 = cls.poly(d, a)
            Q0, tau0 = cost(p0, d, w, beta, xs)
            acc0 = float(np.max(np.abs(eig * p0(eig) - 1.0)))
            with contextlib.redirect_stdout(io.StringIO()):
                dc = spectral_correction_adaptive(p0, eig)
            c = p0.coef.copy()
            c[1::2] += dc
            pS = Chebyshev(c)
            QS, tauS = cost(pS, d, w, beta, xs)
            accS = float(np.max(np.abs(eig * pS(eig) - 1.0)))
            rows.append((e, d, acc0, Q0, accS, QS, tauS / tau0))
        out[label] = np.array(rows)
    return out, kappa


def plot(data, kappa):
    from matplotlib.ticker import LogLocator, FuncFormatter

    label, cls, col, ls, mk = BASES[0]
    r = data[label]

    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="major", color=GRIDCOL, lw=1.1, zorder=0)
    ax.grid(True, which="minor", color=GRIDCOL, lw=0.5, alpha=0.6, zorder=0)
    for sp in ax.spines.values():
        sp.set_color(SPINECOL)
    ax.tick_params(labelsize=8, color=SPINECOL)

    ax.plot(r[:, 2], r[:, 3], ls, color=col, lw=1.8, marker=mk, ms=5,
            mfc="white", mec=col, mew=1.2, zorder=3, label=r"$p_0$")
    ax.plot(r[:, 4], r[:, 5], ls="none", marker=mk, ms=6.5, color=col,
            mec="white", mew=1.0, zorder=4, label=r"$p_{SC}$, $K=N$")
    for row in r:
        ax.annotate("", xy=(row[4], row[5]), xytext=(row[2], row[3]),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8,
                                    shrinkA=5, shrinkB=5, alpha=0.35), zorder=2)

    lo = r[:, 5].min(); hi = r[:, 3].max()
    ax.set_xlim(1.6, 2e-16); ax.set_ylim(lo / 1.5, hi * 1.6)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))

    # the headline: tight base vs corrected loose base
    tight = r[-1]; loose_sc = r[0]
    ax.annotate(rf"$\varepsilon=10^{{-3}}$", xy=(tight[2], tight[3]),
                xytext=(-6, 8), textcoords="offset points", fontsize=8,
                color=MUTED, ha="right")
    ax.annotate(rf"$\varepsilon=0.5$", xy=(loose_sc[4], loose_sc[5]),
                xytext=(0, -14), textcoords="offset points", fontsize=8,
                color=MUTED, ha="center")
    ax.annotate("", xy=(loose_sc[4], loose_sc[5] * 1.02),
                xytext=(loose_sc[4], tight[3]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0), zorder=5)
    ax.annotate(rf"${tight[3] / loose_sc[5]:.2f}\times$",
                xy=(loose_sc[4], (tight[3] * loose_sc[5]) ** 0.5),
                xytext=(8, 0), textcoords="offset points", fontsize=9,
                color=INK, va="center")

    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.45, 0.62), fontsize=9,
                    framealpha=0.95, edgecolor=SPINECOL)
    leg.get_frame().set_linewidth(0.8)

    ax.set_xlabel(r"achieved accuracy   $R_K = \max_k |\lambda_k\,p(\lambda_k)-1|$",
                  color=INK, labelpad=2)
    ax.set_ylabel(r"total query cost   $\mathcal{Q} = d/\sqrt{P_{\rm succ}}$",
                  color=INK)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, f"costAccuracy.{FIGFMT}")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    data, kappa = sweep()
    for k, r in data.items():
        print(f"\n{k}  (kappa={kappa:.2f})")
        print(f"{'eps':>7} {'d':>6} {'acc_base':>11} {'Q_base':>9} "
              f"{'acc_SC':>11} {'Q_SC':>9} {'infl':>6}")
        for row in r:
            print(f"{row[0]:>7g} {int(row[1]):>6} {row[2]:>11.2e} {row[3]:>9.1f} "
                  f"{row[4]:>11.2e} {row[5]:>9.1f} {row[6]:>6.2f}")
    plot(data, kappa)