# Spectral Bootstrapping for QSVT

Eigenvalue-bootstrapped polynomial approximation for Quantum Singular Value Transformation (QSVT) linear solvers.

Given a base polynomial $p_0(x) \approx 1/x$ and $K$ known eigenvalues of the system matrix, the spectral correction enforces $\lambda_k p_S(\lambda_k) = 1$ at those eigenvalues **without increasing the polynomial degree**, yielding significant circuit depth reductions.

## Overview

| Module | Description |
|---|---|
| `SpectralPolynomial.py` | Base polynomial constructions (Remez, Mang, Sünderhauf) and the min-norm spectral correction |
| `SpectralQSVT.py` | QSVT solver with spectral bootstrapping, QSP phase computation, and Qiskit circuit simulation |
| `PoissonFunctions.py` | 1D and 2D Poisson finite difference discretizations with closed-form eigenvalues |
| `tutorial.ipynb` | Step-by-step tutorial covering polynomial construction, spectral correction, and QSVT validation |
| `paperExperiments.ipynb` | Experiments reproducing the results in the accompanying paper |

## Installation

### Prerequisites

- Python 3.9+

### Setup

```bash
git clone https://github.com/UW-ERSL/SpectralBootstrapping.git
cd SpectralBootstrapping
pip install -r requirements.txt
```

## Quick Start

### 1. Spectral polynomial (no quantum circuit)

```python
import numpy as np
from SpectralPolynomial import MangPolynomial, spectral_correction
from numpy.polynomial.chebyshev import Chebyshev

# Problem setup
kappa = 10.0
a = 1.0 / kappa
eps = 0.5
eigenvalues = np.array([0.1, 0.5, 1.0])  # known eigenvalues in (0, 1]

# Base polynomial
degree = MangPolynomial.mindegree(eps, a)
p0 = MangPolynomial.poly(degree, a)

# Spectral correction
c_corr = spectral_correction(p0, eigenvalues)
coef = p0.coef.copy()
coef[1::2] += c_corr
p_S = Chebyshev(coef)

# Verify: residuals at eigenvalues should be ~1e-15
for lam in eigenvalues:
    print(f"lambda={lam:.2f}:  |lambda*p_S(lambda) - 1| = {abs(lam*p_S(lam) - 1):.2e}")
```

### 2. Full QSVT solve (1D Poisson)

```python
from SpectralQSVT import QSVT, SpectralQSVT
from PoissonFunctions import build_1d_poisson, eigs_1d_poisson
import numpy as np

# Build problem
m = 4  # N = 2^m = 16 interior nodes
A, b = build_1d_poisson(m, function_type="uniform")
eigs = eigs_1d_poisson(m)
A /= (1.01 * np.max(eigs))
eigs /= (1.01 * np.max(eigs))
kappa = 1.0 / np.min(eigs)

# Base QSVT
solver_base = QSVT(A, b, kappa=kappa, target_error=0.5, polyMethod='Mang')
u_base, prob_base, _ = solver_base.solve()

# Spectral QSVT (correct all eigenvalues)
K = len(eigs)
solver_spec = SpectralQSVT(A, b, lam_K=eigs[:K], kappa=kappa,
                            target_error=0.5, polyMethod='Mang')
u_spec, prob_spec, _ = solver_spec.solve()

# Compare fidelity
u_cl = np.linalg.solve(A, b)
u_cl /= np.linalg.norm(u_cl)
print(f"Base fidelity:     {abs(np.vdot(u_base, u_cl))**2:.6f}")
print(f"Spectral fidelity: {abs(np.vdot(u_spec, u_cl))**2:.6f}")
```

## Method

The spectral correction solves a $K \times K$ Gram system:

$$\mathbf{G}\,\boldsymbol{\alpha} = \mathbf{r}, \qquad G_{ij} = \lambda_i \lambda_j \sum_{\ell=0}^{n_0-1} T_{2\ell+1}(\lambda_i)\,T_{2\ell+1}(\lambda_j)$$

where $r_k = 1 - \lambda_k p_0(\lambda_k)$ are the residuals at the target eigenvalues. The correction $p_{\mathrm{corr}}$ is added to $p_0$ in the odd Chebyshev basis, preserving the degree.

Key properties:
- **Same degree** as the base polynomial (no circuit depth increase)
- **Machine-precision** residuals at corrected eigenvalues ($\sim 10^{-15}$)
- **Robust** to eigenvalue perturbations (tested up to 10% relative error)
- **Base-agnostic**: works with Remez, Mang, or Sünderhauf

## Available Base Polynomials

| Polynomial | Method | Reference |
|---|---|---|
| `RemezPolynomial` | Minimax (relative criterion) | Trefethen, *Approximation Theory and Approximation Practice* (2019) |
| `MangPolynomial` | Least-squares Chebyshev | Mang et al. (2024) |
| `SunderhaufPolynomial` | Analytical closed-form | Sünderhauf et al., *Quantum* 8:1226 (2024) |

## Citation

If you use this code, please cite:

```bibtex
@article{spectralbootstrapping2026,
  title   = {Eigenvalue-Bootstrapped Polynomial Approximation for {QSVT}},
  author  = {Krishnan Suresh},
  journal = {[Journal]},
  year    = {2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
