"""
makeFigures.py -- every figure in the paper, from one file.

    python makeFigures.py            # all figures
    python makeFigures.py 1          # Figure 1 only
    python makeFigures.py 2 3        # Figures 2 and 3

Outputs (all into figs/, created on demand; the .tex has
\\graphicspath{{figs/}{./}}):

  Figure 1   polynomialComparison.png        approximants vs 1/x
             polynomialErrorsRelative.png    |x p(x) - 1|,  L-inf RELATIVE
             polynomialErrorsAbsolute.png    |p(x) - 1/x|,  L-inf ABSOLUTE
  Figure 2   spectralPolynomials.png         pure spectral polynomial vs degree
  Figure 3   spectralChebIterError{1,2,3,1D}.png
                                             p_0 vs p_SC

  Target 7   Poisson2D_classical.png         2D Poisson solution surfaces
             Poisson2D_base.png
             Poisson2D_corrected.png

Of these the paper currently uses spectralPolynomials (Fig. 1) and
spectralChebIterError1D (Fig. 2).  The rest are kept for the rebuttal and for
inspection; target 7's surfaces were cut from the paper but the target still
runs.

Target 7 runs the QSVT circuits and therefore needs pyqsp and qiskit-aer;
targets 1-3 need only numpy and matplotlib.  pyqsp 0.2.0 will not build against
modern setuptools (AttributeError: install_layout) -- use a venv with
setuptools<60 and pip install --no-build-isolation.

--------------------------------------------------------------------------
NOTES ON WHAT THESE PLOT, AND WHY
--------------------------------------------------------------------------
Figure 1(b,c).  The same three polynomials are drawn under both uniform
criteria.  Three degree-20-ish polynomials on a log axis cross each other
dozens of times and the raw curves are unreadable; the message is not the
individual oscillations but the ENVELOPE of their peaks -- flat in the
criterion a construction optimises, sloped by one power of x in the other.
The peaks are therefore the primary mark, with the raw error faint behind.
Since x p(x) - 1 = x (p(x) - 1/x), the two panels differ by exactly one power
of x, so the measured envelope slopes differ by exactly 1.

Figure 2.  With n = ceil(n_factor * N) odd Chebyshev terms the polynomial is
c_0 T_1 + ... + c_{n-1} T_{2n-1}, so d = 2n - 1.  At n_factor = 1 and N = 3
that is d = 5, not the d = 6 the submitted figure was labelled [R1 minor 3].
tau is reported per curve so the caption's claim -- tau approaches kappa as
the degree grows -- is visible rather than asserted.

Figure 3.  Built from ChebIterPolynomial, the closed-form minimiser of the
L-inf relative criterion and the single base polynomial p_0 of the paper.  The
1D panel uses the standardised normalisation 1.01*lambda_max, i.e. kappa = 9.57
and lambda = 0.1045, 0.3782, 0.7164, 0.9901.  The plotting grid is refined
geometrically
around each targeted eigenvalue: on a uniform grid the corrected polynomial's
plunge to machine precision falls between samples and the figure's main
feature is silently lost.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Vector output by default: line art stays sharp at any zoom and in print, and
# is usually smaller than the raster equivalent.  Set FIGFMT=png to override.
# fonttype 42 embeds TrueType rather than Type 3, which some journals reject.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
FIGFMT = os.environ.get("FIGFMT", "pdf").lower()
from numpy.polynomial import Chebyshev

from PolynomialApproximators import (ChebIterPolynomial as CI,
                                     MangPolynomial as MA,
                                     SunderhaufPolynomial as SU,
                                     SpectralPolynomial,
                                     spectral_correction_adaptive)

# Figure 7 only; imported lazily so targets 1-3 run without pyqsp/qiskit.

FIGDIR = "figs"

# Categorical palette, one hue per accuracy criterion, used in every figure.
# Validated (light surface): lightness band, chroma floor, CVD separation
# (worst adjacent dE 21.1 protan), normal-vision floor (dE 23.0), contrast --
# all pass.  Line style carries identity independently of hue, for print and
# for colour-vision deficiency.
C_REL, C_L2, C_ABS = "#1f77b4", "#d62728", "#9467bd"
L_REL = r"$L_\infty^{\mathrm{rel}}$"
L_L2 = r"$L_2^{\mathrm{rel}}$"
L_ABS = r"$L_\infty^{\mathrm{abs}}$"

SURFACE = "#f7f7f7"
GRIDCOL = "white"
SPINECOL = "#cccccc"
INK = "#222222"
MUTED = "#444444"


def _surface(ax, logy=False):
    """The recessive light surface shared by every panel in the paper."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRIDCOL, lw=1.1, zorder=0)
    if logy:
        ax.grid(True, which="minor", color=GRIDCOL, lw=0.5, alpha=0.6, zorder=0)
        ax.set_yscale("log")
    for sp in ax.spines.values():
        sp.set_color(SPINECOL)
    ax.tick_params(labelsize=9, color=SPINECOL)
    ax.set_xlim(0.0, 1.02)


def _legend(ax, loc):
    ax.legend(fontsize=8, loc=loc, framealpha=0.95,
              edgecolor=SPINECOL, labelcolor=INK, handlelength=1.6,
              borderpad=0.3, labelspacing=0.3, handletextpad=0.5)


def _save(fig, name):
    """Write to FIGDIR, honouring FIGFMT.  The .tex omits the extension, so
    pdflatex picks up whichever of .pdf / .png is present."""
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, os.path.splitext(name)[0] + "." + FIGFMT)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {path}")


# ==========================================================================
# FIGURE 1 -- the accuracy criteria
# ==========================================================================
FIG1_KAPPA, FIG1_EPS = 10.0, 0.2
FIG1_A = 1.0 / FIG1_KAPPA

# Short labels: the construction is named in the text, and long legend entries
# force the panel wider than the column it is placed in.
SERIES = [
    (L_REL + r", $d=%d$", CI, C_REL, "-",   1.8),
    (L_L2 + r", $d=%d$",  MA, C_L2,  "--",  1.6),
    (L_ABS + r", $d=%d$", SU, C_ABS, "-.",  1.6),
]


def _peaks(x, y):
    """Local maxima of y -- the envelope is the message; the nulls are not."""
    loc = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    return x[1:-1][loc], y[1:-1][loc]


def _fig1_approx(fname="polynomialComparison.png"):
    x = np.linspace(FIG1_A, 1.0, 40001)
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    _surface(ax)
    ax.plot(x, 1.0 / x, color=MUTED, linestyle=":", lw=1.4, zorder=2,
            label=r"$1/x$")
    for label, cls, colour, ls, lw in SERIES:
        d = cls.mindegree(FIG1_EPS, FIG1_A)
        ax.plot(x, cls.poly(d, FIG1_A)(x), color=colour, linestyle=ls, lw=lw,
                label=label % d, zorder=3, solid_capstyle="round")
    ax.set_xlabel(r"$x$", fontsize=11)
    ax.set_ylabel(r"$p(x)$", fontsize=11)
    _legend(ax, "upper right")
    _save(fig, fname)


def _fig1_error(kind, fname):
    """kind: 'relative' -> |xp-1|;  'absolute' -> |p-1/x|."""
    x = np.linspace(FIG1_A, 1.0, 200001)
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    _surface(ax, logy=True)
    # Common limits across panels, so the one-power-of-x relationship reads.
    ax.set_ylim(1.5e-2, 2.5)

    for label, cls, colour, ls, lw in SERIES:
        d = cls.mindegree(FIG1_EPS, FIG1_A)
        p = cls.poly(d, FIG1_A)
        y = (np.abs(x * p(x) - 1.0) if kind == "relative"
             else np.abs(p(x) - 1.0 / x))
        ax.plot(x, y, color=colour, lw=0.6, alpha=0.13, zorder=2,
                rasterized=True)   # 2e5 points; keep it out of the vector path
        px, py = _peaks(x, y)
        ax.plot(px, py, color=colour, linestyle=ls, lw=lw, marker="o", ms=3.4,
                mfc="white", mec=colour, mew=1.1, label=label % d, zorder=4,
                solid_capstyle="round")

    ax.axhline(FIG1_EPS, color=MUTED, linestyle=":", lw=1.2, zorder=3)
    ax.annotate(rf"$\varepsilon = {FIG1_EPS}$", xy=(0.012, FIG1_EPS * 1.5),
                xycoords=("axes fraction", "data"), va="bottom", ha="left",
                fontsize=9, color=MUTED,
                bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"))
    ax.set_xlabel(r"$x$", fontsize=11)
    ax.set_ylabel(r"$|x\,p(x) - 1|$" if kind == "relative"
                  else r"$|p(x) - 1/x|$", fontsize=11)
    # Keep the legend off the envelopes: the relative panel is empty at the
    # top, the absolute panel at the bottom left.
    _legend(ax, "upper right" if kind == "relative" else "lower left")
    _save(fig, fname)


def _fig1_envelopes():
    """Report the peak-envelope slopes quoted in the caption."""
    x = np.linspace(FIG1_A, 1.0, 400001)
    print("    peak-envelope slope (error ~ x^m):")
    for label, cls, *_ in SERIES:
        d = cls.mindegree(FIG1_EPS, FIG1_A)
        p = cls.poly(d, FIG1_A)
        for kind in ("relative", "absolute"):
            y = (np.abs(x * p(x) - 1.0) if kind == "relative"
                 else np.abs(p(x) - 1.0 / x))
            xs, ys = _peaks(x, y)
            if len(ys) > 2:
                m = np.polyfit(np.log(xs), np.log(ys), 1)[0]
                name = label.split(',')[0].strip()
                print(f"      {name:<28} {kind:>8}: "
                      f"x^{m:+.2f}   spread {ys.max()/ys.min():.1f}x")


def figure1():
    print("  Figure 1 -- accuracy criteria")
    _fig1_approx()
    _fig1_error("relative", "polynomialErrorsRelative.png")
    _fig1_error("absolute", "polynomialErrorsAbsolute.png")
    _fig1_envelopes()


# ==========================================================================
# FIGURE 2 -- the pure spectral polynomial
# ==========================================================================
FIG2_KAPPA = 10.0
FIG2_LAM = np.array([1.0 / FIG2_KAPPA, 0.5, 1.0]) / 1.01   # paper normalisation
FIG2_NFACTORS = (1, 1.5, 3, 6)                              # -> d = 5, 9, 17, 35
# Degree is ORDERED, so a single-hue sequential ramp, not categorical colours.
FIG2_RAMP = ["#9ecae1", "#4292c6", "#2171b5", "#08306b"]


def figure2():
    print("  Figure 2 -- pure spectral polynomial")
    lam = FIG2_LAM
    a = lam.min()
    x = np.linspace(a, 1.0, 4001)
    xf = np.linspace(-1.0, 1.0, 200001)

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    _surface(ax)
    ax.plot(x, 1.0 / x, color=MUTED, linestyle=":", lw=1.5, zorder=5,
            label=r"$1/x$")

    print(f"    lambda = {np.round(lam, 4)}   kappa = {lam.max()/lam.min():.1f}")
    for nf, colour in zip(FIG2_NFACTORS, FIG2_RAMP):
        sp = SpectralPolynomial(lam, n_factor=nf)
        p, d = sp.poly(), sp.mindegree()
        tau = float(np.max(np.abs(p(xf))))
        resid = float(np.max(np.abs(lam * p(lam) - 1.0)))
        ax.plot(x, p(x), color=colour, lw=1.8, zorder=3,
                label=rf"$d={d}$  ($\tau={tau:.1f}$)", solid_capstyle="round")
        print(f"      n_factor={nf:<4} d={d:<3} tau={tau:7.2f}  "
              f"max|lam p(lam) - 1| = {resid:.1e}")

    ax.plot(lam, 1.0 / lam, "o", ms=7, mfc="white", mec=INK, mew=1.3,
            zorder=6, label=r"$\lambda_k$")
    ax.set_ylim(-6.0, 21.0)          # clipped; the d=5 curve reaches -43.9
    ax.set_xlabel(r"$x$", fontsize=11)
    ax.set_ylabel(r"$p_S(x)$", fontsize=11)
    _legend(ax, "upper right")
    _save(fig, "spectralPolynomials.png")


# ==========================================================================
# FIGURES 3-6 -- spectral correction, base vs corrected
# ==========================================================================
# The paper uses the single base polynomial p_0 = ChebIter.  The other two
# families remain importable for the rebuttal's degree-scaling comparison,
# but no figure is built from them.
BASES = [("ChebIter", CI, L_REL, C_REL)]

CASES = [("1", 10.0, 0.2, [0.10, 0.50, 1.00]),
         ("2", 10.0, 0.2, [0.10, 0.15, 1.00]),
         ("3", 10.0, 0.2, [0.10, 0.10, 1.00]),
         ("1D", 9.57, 0.1, [0.1045, 0.3782, 0.7164, 0.9901])]

FLOOR = 1e-17   # clip machine-zero residuals onto the axis floor


def _corrected(p0, lam):
    dc, info = spectral_correction_adaptive(p0, np.asarray(lam, float),
                                            return_info=True)
    coef = p0.coef.copy()
    coef[1::2] += dc
    return Chebyshev(coef), info


def _error_panel(stem, cls, label, colour, tag, kappa, eps, lam):
    a = 1.0 / kappa
    d = cls.mindegree(eps, a)
    p0 = cls.poly(d, a)
    pSC, info = _corrected(p0, lam)

    lam_arr = np.unique(np.asarray(lam, float))
    x = np.linspace(a, 1.0, 40001)
    fine = np.concatenate([lk + np.geomspace(1e-12, 5e-3, 400) * sgn
                           for lk in lam_arr for sgn in (-1.0, 1.0)])
    x = np.unique(np.concatenate([x, lam_arr,
                                  fine[(fine >= a) & (fine <= 1.0)]]))

    e0 = np.abs(x * p0(x) - 1.0)
    e1 = np.maximum(np.abs(x * pSC(x) - 1.0), FLOOR)

    # Placed at 0.29\linewidth of a figure* (~2.0 in), so generate near that
    # size and set the fonts for it: scaling a 4.4 in panel down to 2.0 in put
    # the legend under 4 pt on the page.
    fig, ax = plt.subplots(figsize=(2.45, 2.15))
    _surface(ax, logy=True)
    ax.set_ylim(1e-18, 10.0)
    ax.plot(x, e0, color=colour, linestyle="--", lw=1.2, zorder=3,
            label=f"base, $d={d}$")
    ax.plot(x, e1, color=colour, linestyle="-", lw=1.6, zorder=4,
            label="corrected")
    ax.plot(lam_arr, np.maximum(np.abs(lam_arr * pSC(lam_arr) - 1.0), FLOOR),
            "o", ms=4.5, mfc="white", mec=INK, mew=1.0, zorder=5,
            label=r"$\lambda_k$")
    ax.axhline(eps, color=MUTED, linestyle=":", lw=1.0, zorder=2)
    ax.set_xlabel(r"$x$", fontsize=11)
    ax.set_ylabel(r"$|x\,p(x) - 1|$", fontsize=11)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc="center left", framealpha=0.95,
              edgecolor=SPINECOL, labelcolor=INK, handlelength=1.4,
              borderpad=0.3, labelspacing=0.3, handletextpad=0.5)
    _save(fig, f"spectral{stem}Error{tag}.png")
    return d, info


def figures36():
    print("  Figure 3 -- spectral correction")
    for tag, kappa, eps, lam in CASES:
        print(f"    case {tag}: kappa={kappa}  eps={eps}  lambda={lam}")
        for stem, cls, label, colour in BASES:
            d, info = _error_panel(stem, cls, label, colour,
                                   tag, kappa, eps, lam)
            print(f"      spectral{stem}Error{tag}: d={d:<4} "
                  f"K_eff={info['K_eff']} resid={info['resid_kept']:.2e} "
                  f"tau infl={info['tau_inflation']:.3f} "
                  f"{'ok' if info['ok'] else 'NOT VERIFIED'}")


# ==========================================================================
# FIGURE 7 -- 2D Poisson solution surfaces
# ==========================================================================
FIG7_M = 4               # N = 16 per side, 256 unknowns
FIG7_EPS = 0.2
FIG7_K = 32
FIG7_BASE = "chebiter"   # p_0, the base polynomial of the paper


def _surface3d(X, Y, Z, title, fname, zlim):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d)
    fig = plt.figure(figsize=(4.2, 3.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none",
                    rstride=1, cstride=1, antialiased=True,
                    rasterized=True)   # a facet mesh is huge as vector
    ax.set_xlabel(r"$x$", fontsize=11)
    ax.set_ylabel(r"$y$", fontsize=11)
    ax.set_zlabel(r"$u$", fontsize=11)
    ax.set_zlim(*zlim)               # common across panels, so the three
    ax.tick_params(labelsize=8)      # surfaces are directly comparable
    ax.set_title(title, fontsize=10, color=INK)
    _save(fig, fname)


def figure7():
    """The 2D Poisson surfaces.  Needs pyqsp and qiskit-aer."""
    print("  Figure 7 -- 2D Poisson solution surfaces")
    from PoissonFunctions import build_2d_poisson, eigs_2d_poisson
    from QSVTSolvers import StandardQSVT, SpectrallyBootstrappedQSVT

    N = 2 ** FIG7_M
    A, b = build_2d_poisson(FIG7_M, function_type="uniform")
    lam = np.sort(eigs_2d_poisson(FIG7_M))
    scale = 1.01 * lam.max()          # paper normalisation
    A, lam = A / scale, lam / scale
    kappa = 1.0 / lam.min()

    u_cl = np.linalg.solve(A, b)
    u_cl /= np.linalg.norm(u_cl)

    base = StandardQSVT(A, b, kappa, target_error=FIG7_EPS,
                        polyMethod=FIG7_BASE)
    u_base, _, _ = base.solve()
    sc = SpectrallyBootstrappedQSVT(A, b, lam_K=lam[:FIG7_K], kappa=kappa,
                                    target_error=FIG7_EPS,
                                    polyMethod=FIG7_BASE)
    u_sc, _, _ = sc.solve()

    d = len(base.angles) - 1
    print(f"    N={N*N}  kappa={kappa:.1f}  eps={FIG7_EPS}  d={d}  K={FIG7_K}")
    for nm, u in (("classical", u_cl), ("base", u_base), ("corrected", u_sc)):
        print(f"      max({nm:<9}) = {np.max(u):.4f}")

    h = 1.0 / (N + 1)
    g = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(g, g)
    zmax = max(np.max(u) for u in (u_cl, u_base, u_sc))
    zlim = (0.0, 1.05 * zmax)

    for u, title, fname in (
            (u_cl,   rf"Classical;  max $= {np.max(u_cl):.4f}$",
             "Poisson2D_classical.png"),
            (u_base, rf"$p_0$;  max $= {np.max(u_base):.4f}$",
             "Poisson2D_base.png"),
            (u_sc,   rf"$p_{{SC}}$;  max $= {np.max(u_sc):.4f}$",
             "Poisson2D_corrected.png")):
        _surface3d(X, Y, u.reshape((N, N)), title, fname, zlim)


# ==========================================================================
TARGETS = {"1": figure1, "2": figure2, "3": figures36, "7": figure7}

if __name__ == "__main__":
    which = sys.argv[1:] or list(TARGETS)
    unknown = [w for w in which if w not in TARGETS]
    if unknown:
        sys.exit(f"unknown figure(s) {unknown}; choose from {list(TARGETS)}")
    for w in which:
        TARGETS[w]()