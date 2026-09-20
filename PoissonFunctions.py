import numpy as np


def build_1d_poisson(m: int, function_type="uniform"):
    """Build the 1D Poisson system for a given m (N = 2^m grid points)."""
    N = 2**m
    h = 1.0 / (N + 1)
    A = (np.diag(np.full(N, 2.0))
         + np.diag(np.full(N - 1, -1.0), k=1)
         + np.diag(np.full(N - 1, -1.0), k=-1))
    A /= h**2  # Scale by grid spacing squared
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
        mid = N1 // 2 * N1 + N1 // 2   # centre of N1 x N1 grid
        b[mid] = 1.0
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

def eigs_1d_poisson(m):
    N1  = 2**m
    h   = 1.0 / (N1 + 1)
    k   = np.arange(1, N1 + 1)
    lam = 4.0 / h**2 * np.sin(k * np.pi / (2*(N1+1)))**2
    lam = np.sort(lam)
    return lam

def eigs_2d_poisson(m):
    N1   = 2**m
    h    = 1.0 / (N1 + 1)
    k    = np.arange(1, N1 + 1)
    lam1 = 4.0 / h**2 * np.sin(k * np.pi / (2*(N1+1)))**2
    # tensor product: all pairwise sums
    lam2d = (lam1[:, None] + lam1[None, :]).ravel()
    return lam2d

if __name__ == "__main__":
    m = 4
    A, b = build_1d_poisson(m, function_type="uniform")
    print("1D Poisson eigenvalues:", eigs_1d_poisson(m))


def eigs_3d_poisson(n):
    """Eigenvalues of the 7-point FD Laplacian on an n x n x n interior grid.

    lam_{ijk} = 4/h^2 [sin^2(i pi h/2) + sin^2(j pi h/2) + sin^2(k pi h/2)],
    h = 1/(n+1), i,j,k = 1..n.  Returned sorted ascending, length n^3.

    Unlike a synthetic Weyl spectrum lam_k ~ k^{2/3}, this one is heavily
    DEGENERATE: lam_{ijk} is symmetric under permutations of (i,j,k), so
    multiplicities of 3 and 6 are the rule.  That distinction matters for the
    correction, because duplicate eigenvalues merge losslessly while distinct
    but unresolvable ones do not.
    """
    h = 1.0 / (n + 1)
    k = np.arange(1, n + 1)
    l1 = 4.0 / h**2 * np.sin(k * np.pi / (2 * (n + 1)))**2
    l3 = (l1[:, None, None] + l1[None, :, None] + l1[None, None, :]).ravel()
    return np.sort(l3)


def modes_3d_poisson(n):
    """Spectrum AND modal load amplitudes for the 3D operator, without ever
    forming an n^3 x n^3 matrix.

    The eigenvectors are tensor products of the 1D sine modes, so the modal
    amplitudes of a separable load factorise: a uniform load and a centred point
    load are each an outer product of the corresponding 1D coefficient vectors.
    A random load is generated directly in the eigenbasis, which is equivalent
    since the modal transform is orthogonal.

    Returns (w_sorted, {load: beta_sorted}), betas normalised to unit norm.
    """
    h = 1.0 / (n + 1)
    k = np.arange(1, n + 1)
    l1 = 4.0 / h**2 * np.sin(k * np.pi / (2 * (n + 1)))**2
    # orthonormal 1D sine transform, S[p, i] = sqrt(2h) sin(i pi (p+1) h)
    p = np.arange(1, n + 1)
    S = np.sqrt(2.0 * h) * np.sin(np.outer(p, k) * np.pi * h)

    u1 = S.T @ np.ones(n)                      # uniform load, 1D coefficients
    e1 = np.zeros(n); e1[n // 2] = 1.0
    s1 = S.T @ e1                              # centred delta, 1D coefficients

    w = (l1[:, None, None] + l1[None, :, None] + l1[None, None, :]).ravel()
    B = {'uniform': np.einsum('i,j,k->ijk', u1, u1, u1).ravel(),
         'point':   np.einsum('i,j,k->ijk', s1, s1, s1).ravel()}
    r = np.random.default_rng(0).standard_normal(n ** 3)
    B['random'] = r
    order = np.argsort(w)
    w = w[order]
    B = {L: v[order] / np.linalg.norm(v) for L, v in B.items()}
    return w, B