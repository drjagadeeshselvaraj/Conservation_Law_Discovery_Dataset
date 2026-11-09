"""lawsample: sample-size calculator for null-space conservation-law discovery.

Model: derivative features have covariance sigma^2 on k conserved directions and
sigma^2 (1 + lam_j) on the other directions. With A1 = sum(1 + 1/lam_j) and
A2 = sum((1 + 1/lam_j)^2) over the non-conserved directions:
  * the bottom-k sample eigenspace has positive alignment iff n > A2,
  * its squared alignment is (1 - A2/n) / (1 - A1/n) for n > A2.
For equal lam, A2 = (p - k)(1 + 1/lam)^2 and no estimator can do better (Theorem 2 of the article,
average case, fixed k).  Fixed-k asymptotics; use with care when k is large.
"""
import numpy as np

def constants(lam):
    lam = np.asarray(lam, float)
    if np.any(lam <= 0): raise ValueError("all lam_j must be positive")
    t = 1 + 1 / lam
    return float(t.sum()), float((t ** 2).sum())

def alignment(n, lam):
    """Predicted squared alignment at sample size n (0 at or below the threshold)."""
    A1, A2 = constants(lam)
    return 0.0 if n <= A2 else (1 - A2 / n) / (1 - A1 / n)

def threshold(lam):
    return constants(lam)[1]

def equal_threshold(p, k, lam):
    return (p - k) * (1 + 1 / lam) ** 2

def samples_for_alignment(target, lam):
    """Smallest n with predicted alignment >= target (0 < target < 1)."""
    A1, A2 = constants(lam)
    # (1 - A2/n) = target (1 - A1/n)  ->  n = (A2 - target*A1) / (1 - target)
    return (A2 - target * A1) / (1 - target)

def from_matrices(S, N, k=1):
    """Estimate lam_j from a feature second-moment matrix S and a feature-noise covariance N
    (generalised eigenproblem S c = mu N c). Returns (lam, mu_null). Eigenvalues mu_j - 1 are
    read as lam_j for the non-conserved directions; the k smallest are the conserved ones."""
    import scipy.linalg as sl
    mu = sl.eigh(S, N, eigvals_only=True)
    lam = mu[k:] - 1
    return lam[lam > 0], mu[:k]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Sample size for null-space law discovery")
    ap.add_argument("--p", type=int, required=True, help="library size")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--lam", type=float, required=True, help="signal/noise ratio per non-conserved direction")
    ap.add_argument("--target", type=float, default=0.9)
    a = ap.parse_args()
    lam = np.full(a.p - a.k, a.lam)
    print(f"threshold n* = {threshold(lam):.0f}; n for alignment {a.target}: {samples_for_alignment(a.target, lam):.0f}")
