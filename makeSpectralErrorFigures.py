"""
makeSpectralErrorFigures.py -- the spectral-correction error panels.

Produces the twelve panels of Figures 3-6:

    figs/spectral{ChebIter,Mang,Sunderhauf}Error1.png   Example 1  lam = .10 .50 1.0
    figs/spectral{ChebIter,Mang,Sunderhauf}Error2.png   Example 2  lam = .10 .15 1.0
    figs/spectral{ChebIter,Mang,Sunderhauf}Error3.png   Example 3  lam = .10 .10 1.0
    figs/spectral{ChebIter,Mang,Sunderhauf}Error1D.png  1D Poisson N=4

Each panel plots |x p(x) - 1| for the base polynomial and for its spectral
correction, with the targeted eigenvalues marked.

Two things changed relative to the submitted version:

  (1) The L-inf relative panels are built from ChebIterPolynomial, not from our
      Remez implementation.  At these degrees (d = 23 here, d = 29 for the 1D
      case) the two agree to ~1e-8, so the curves are unchanged; the closed form
      is used because Remez is not converged at the higher degrees used
      elsewhere in the paper, and the paper should not mix the two.

  (2) The 1D Poisson case uses the standardised normalisation 1.01*lambda_max,
      i.e. kappa = 9.57 and lam = .1045 .3782 .7164 .9901.  The submitted
      version used lambda_max = 1 (kappa = 9.47).

Bases are labelled by the criterion they optimise, per Table 1 of the paper.

Usage:  python makeSpectralErrorFigures.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev

from PolynomialApproximators import (ChebIterPolynomial as CI,
                                     MangPolynomial as MA,
                                     SunderhaufPolynomial as SU,
                                     spectral_correction_adaptive)

# (file stem, class, criterion label, colour) -- colours match Figure 1.
BASES = [
    ("ChebIter",   CI, r"$L^\infty_{\mathrm{rel}}$", "#1f77b4"),
    ("Mang",       MA, r"$L^{2}$",                   "#d62728"),
    ("Sunderhauf", SU, r"$L^\infty_{\mathrm{abs}}$", "#9467bd"),
]

CASES = [
    ("1",  10.0,  0.2, [0.10, 0.50, 1.00]),
    ("2",  10.0,  0.2, [0.10, 0.15, 1.00]),
    ("3",  10.0,  0.2, [0.10, 0.10, 1.00]),
    ("1D",  9.57, 0.1, [0.1045, 0.3782, 0.7164, 0.9901]),
]

FLOOR = 1e-17   # clip machine-zero residuals so they sit on the axis floor
FIGDIR = "figs" # all output goes here; the .tex has \graphicspath{{figs/}{./}}


def corrected(p0, lam):
    dc, info = spectral_correction_adaptive(p0, np.asarray(lam, float),
                                            return_info=True)
    coef = p0.coef.copy()
    coef[1::2] += dc
    return Chebyshev(coef), info


def panel(stem, cls, label, colour, tag, kappa, eps, lam):
    a = 1.0 / kappa
    d = cls.mindegree(eps, a)
    p0 = cls.poly(d, a)
    pSC, info = corrected(p0, lam)

    # Uniform grid, refined around each targeted eigenvalue so the corrected
    # curve's plunge to machine precision is actually resolved on the plot.
    lam_arr = np.unique(np.asarray(lam, float))
    x = np.linspace(a, 1.0, 40001)
    fine = np.concatenate([lk + np.geomspace(1e-12, 5e-3, 400) * sgn
                           for lk in lam_arr for sgn in (-1.0, 1.0)])
    x = np.unique(np.concatenate([x, lam_arr, fine[(fine >= a) & (fine <= 1.0)]]))
    e0 = np.abs(x * p0(x) - 1.0)
    e1 = np.maximum(np.abs(x * pSC(x) - 1.0), FLOOR)

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.set_facecolor("#f7f7f7")
    ax.grid(True, which="major", color="white", lw=1.1, zorder=0)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.tick_params(labelsize=10, color="#cccccc")
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(1e-18, 10.0)

    ax.plot(x, e0, color=colour, linestyle="--", lw=1.6, zorder=3,
            label=f"{label} base, $d={d}$")
    ax.plot(x, e1, color=colour, linestyle="-", lw=2.0, zorder=4,
            label=f"SC-{label}, $d={d}$")

    lam_u = lam_arr
    ax.plot(lam_u, np.maximum(np.abs(lam_u * pSC(lam_u) - 1.0), FLOOR),
            "o", ms=6, mfc="white", mec="#222222", mew=1.2, zorder=5,
            label=r"targeted $\lambda_k$")

    ax.axhline(eps, color="#444444", linestyle=":", lw=1.1, zorder=2)
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$|x\,p(x) - 1|$", fontsize=13)
    ax.legend(fontsize=8, loc="center left", framealpha=0.95,
              edgecolor="#cccccc", labelcolor="#222222")
    fig.tight_layout()
    fname = os.path.join(FIGDIR, f"spectral{stem}Error{tag}.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fname, d, info


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    for tag, kappa, eps, lam in CASES:
        print(f"\ncase {tag}:  kappa={kappa}  eps={eps}  "
              f"lambda={lam}")
        for stem, cls, label, colour in BASES:
            fname, d, info = panel(stem, cls, label, colour,
                                   tag, kappa, eps, lam)
            print(f"  {fname:<34} d={d:<4} K_eff={info['K_eff']} "
                  f"resid={info['resid_kept']:.2e} "
                  f"tau infl={info['tau_inflation']:.3f} "
                  f"{'ok' if info['ok'] else 'NOT VERIFIED'}")