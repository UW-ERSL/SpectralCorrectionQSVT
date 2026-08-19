"""
makeCriterionFigure.py -- the two-panel criterion figure.

Each construction equioscillates in the criterion it optimises and in no other.
Plotting the same three polynomials under both uniform criteria makes this
visible: a curve is flat in exactly one panel, and touches the tolerance line
there.

  panel (b)  |x p(x) - 1|      L-inf RELATIVE   -- ChebIter flat, Sunderhauf ~ x
  panel (c)  |p(x) - 1/x|      L-inf ABSOLUTE   -- Sunderhauf flat, others ~ 1/x

Since x p(x) - 1 = x (p(x) - 1/x), the panels differ by one power of x, so the
measured envelope slopes differ by exactly 1.

Outputs figs/polynomialComparison.png, figs/polynomialErrorsRelative.png and
figs/polynomialErrorsAbsolute.png, placed with \\subcaptionbox as Figure 1.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PolynomialApproximators import (ChebIterPolynomial as CI,
                                     MangPolynomial as MA,
                                     SunderhaufPolynomial as SU)

FIGDIR = "figs"  # all output goes here; the .tex has \graphicspath{{figs/}{./}}

KAPPA, EPS = 10.0, 0.2
A = 1.0 / KAPPA

# Palette: the categorical hues already used in this paper's figures.
# Validated (light surface): lightness band, chroma floor, CVD separation
# (worst adjacent dE 21.1 protan), normal-vision floor (dE 23.0), contrast -- all pass.
# Line style carries identity independently of hue, for print and CVD.
SERIES = [
    (r"$L^\infty$ relative (ChebIter, $d=%d$)", CI, "#1f77b4", "-",   2.0),
    (r"$L^2$ relative (Mang, $d=%d$)",          MA, "#d62728", "--",  1.8),
    (r"$L^\infty$ absolute (Sünderhauf, $d=%d$)", SU, "#9467bd", "-.", 1.8),
]


def _axes(ax):
    """Recessive grid on a light surface, matching the paper's other figures."""
    ax.set_facecolor("#f7f7f7")
    ax.grid(True, which="major", color="white", lw=1.1, zorder=0)
    ax.grid(True, which="minor", color="white", lw=0.5, alpha=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
    ax.tick_params(labelsize=10, color="#cccccc")
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.02)
    # Common limits across panels so the two are directly comparable, and a floor
    # that clips the interpolation nulls -- the envelope of the PEAKS is the
    # message, and unclipped nulls squeeze it into the top decade.
    ax.set_ylim(1e-4, 4.0)


def panel(kind, fname):
    """kind: 'relative' -> |xp-1|;  'absolute' -> |p-1/x|."""
    x = np.linspace(A, 1.0, 40001)
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    _axes(ax)

    for label, cls, colour, ls, lw in SERIES:
        d = cls.mindegree(EPS, A)
        p = cls.poly(d, A)
        y = np.abs(x * p(x) - 1.0) if kind == "relative" else np.abs(p(x) - 1.0 / x)
        ax.plot(x, y, color=colour, linestyle=ls, lw=lw,
                label=label % d, zorder=3, solid_capstyle="round")

    ax.axhline(EPS, color="#444444", linestyle=":", lw=1.2, zorder=2)
    # Label the tolerance line in clear space above it, left of the first peak.
    ax.annotate(rf"$\varepsilon = {EPS}$", xy=(0.012, EPS * 1.45),
                xycoords=("axes fraction", "data"),
                va="bottom", ha="left", fontsize=10, color="#444444",
                bbox=dict(boxstyle="square,pad=0.15", fc="#f7f7f7", ec="none"))

    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$|x\,p(x) - 1|$" if kind == "relative"
                  else r"$|p(x) - 1/x|$", fontsize=13)
    ax.legend(fontsize=8.5, loc="lower center", ncol=1, framealpha=0.95,
              edgecolor="#cccccc", labelcolor="#222222")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.join(FIGDIR, fname)}")


def approx_panel(fname="polynomialComparison.png"):
    """Panel (a): the approximants themselves against 1/x."""
    x = np.linspace(A, 1.0, 40001)
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    ax.set_facecolor("#f7f7f7")
    ax.grid(True, which="major", color="white", lw=1.1, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
    ax.tick_params(labelsize=10, color="#cccccc")
    ax.set_xlim(0.0, 1.02)

    ax.plot(x, 1.0 / x, color="#444444", linestyle=":", lw=1.4, zorder=2,
            label=r"$1/x$")
    for label, cls, colour, ls, lw in SERIES:
        d = cls.mindegree(EPS, A)
        ax.plot(x, cls.poly(d, A)(x), color=colour, linestyle=ls, lw=lw,
                label=label % d, zorder=3, solid_capstyle="round")

    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$p(x)$", fontsize=13)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95,
              edgecolor="#cccccc", labelcolor="#222222")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.join(FIGDIR, fname)}")


def envelopes():
    """Report the peak-envelope slopes quoted in the caption."""
    x = np.linspace(A, 1.0, 400001)
    print("\n  peak-envelope slope  (|xp-1| ~ x^m):")
    for label, cls, *_ in SERIES:
        d = cls.mindegree(EPS, A)
        p = cls.poly(d, A)
        for kind in ("relative", "absolute"):
            y = np.abs(x * p(x) - 1.0) if kind == "relative" else np.abs(p(x) - 1.0 / x)
            loc = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
            xs, ys = x[1:-1][loc], y[1:-1][loc]
            if len(ys) > 2:
                m = np.polyfit(np.log(xs), np.log(ys), 1)[0]
                print(f"    {label.split('(')[0].strip():<22} {kind:>8}: "
                      f"x^{m:+.2f}   spread {ys.max()/ys.min():.1f}x")


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    approx_panel()
    panel("relative", "polynomialErrorsRelative.png")
    panel("absolute", "polynomialErrorsAbsolute.png")
    envelopes()