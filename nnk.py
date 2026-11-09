import numpy as np, scipy.linalg as sl
from scipy.optimize import minimize

def nn_invariant(X, Xd, H=24, restarts=2, rg=None, iters=150, wd=0.0):
    """Small tanh MLP Q(x) minimising mean(Qdot^2)/Var(Q) (Liu-Tegmark-style Lie-derivative loss)."""
    rg = rg or np.random.default_rng(0)
    m, d = X.shape
    def unpack(th):
        W1 = th[:H*d].reshape(H, d); b1 = th[H*d:H*d+H]; w2 = th[H*d+H:H*d+2*H]
        return W1, b1, w2
    def lossgrad(th):
        W1, b1, w2 = unpack(th)
        a = X @ W1.T + b1; h = np.tanh(a); s = 1 - h*h
        B = Xd @ W1.T; G = s * B
        Q = h @ w2; Qd = G @ w2
        Qm = Q.mean(); den = ((Q-Qm)**2).mean() + 1e-6; num = (Qd**2).mean()
        L = num/den + wd*(np.sum(W1**2)+np.sum(w2**2))/len(th)
        gQd = 2*Qd/(m*den); gQ = -num/den**2 * 2*(Q-Qm)/m
        gw2 = G.T @ gQd + h.T @ gQ
        dG = np.outer(gQd, w2); ds = dG * B; dB = dG * s
        dh = -2*h*ds + np.outer(gQ, w2)
        da = dh * s
        gW1 = da.T @ X + dB.T @ Xd; gb1 = da.sum(0)
        gW1 = gW1 + 2*wd*W1/len(th); gw2 = gw2 + 2*wd*w2/len(th)
        return L, np.concatenate([gW1.ravel(), gb1, gw2])
    best = None
    for _ in range(restarts):
        th0 = np.concatenate([rg.standard_normal(H*d)*0.7, rg.standard_normal(H)*0.3, rg.standard_normal(H)*0.5])
        r = minimize(lossgrad, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=iters))
        if best is None or r.fun < best.fun: best = r
    W1, b1, w2 = unpack(best.x)
    return lambda Z: np.tanh(Z @ W1.T + b1) @ w2

def kernel_invariant(X, Xd, reg=1e-3, ell_mult=1.0):
    """Kernel invariant learner: Q = sum_i alpha_i k(x, x_i) minimising mean(Qdot^2)/Var(Q) (RBF kernel)."""
    m = len(X)
    d2 = ((X[:, None, :]-X[None, :, :])**2).sum(-1)
    ell2 = np.median(d2[d2 > 0]) * ell_mult**2
    K = np.exp(-d2/(2*ell2))
    # dk(x_j, x_i)/dt along xdot_j = -k * (x_j - x_i).xdot_j / ell2
    Dm = -K * ((X*Xd).sum(1)[:, None] - Xd @ X.T) / ell2
    Kc = K - K.mean(0) - K.mean(1)[:, None] + K.mean()
    A = Dm.T @ Dm / m
    B = Kc.T @ Kc / m + reg * np.trace(Kc.T@Kc/m)/m * np.eye(m)
    w, V = sl.eigh(A + 1e-12*np.trace(A)/m*np.eye(m), B)
    al = V[:, 0]
    def Q(Z):
        d2z = ((Z[:, None, :]-X[None, :, :])**2).sum(-1)
        return np.exp(-d2z/(2*ell2)) @ al
    return Q

def project_to_library(Qvals, Phi, ridge=1e-6):
    """Library coefficients c with Phi c ~ Q (both centred), ridge-regularised, standardised columns."""
    Pc = Phi - Phi.mean(0); sc = Pc.std(0) + 1e-12
    Ps = Pc / sc; q = Qvals - Qvals.mean()
    c = np.linalg.solve(Ps.T@Ps + ridge*len(q)*np.eye(Ps.shape[1]), Ps.T@q)
    return c / sc


# ---------------- tuned versions (hyperparameters chosen on held-out rows, no ground truth) ----------------
def stlsq_pi(Fm, tau, ridge, iters=8):
    """Implicit sparse null vector (SINDy-PI style): for each library term j as left-hand side, sparse regression
    on the other terms (STLSQ, ridge); returns list of (j, v) with F v ~ 0."""
    P_ = Fm.shape[1]; out = []
    for j in range(P_):
        idx = [q for q in range(P_) if q != j]; A = Fm[:, idx]; b = -Fm[:, j]
        G = A.T @ A + ridge * len(A) * np.eye(P_-1)
        c = np.linalg.solve(G, A.T @ b)
        for _ in range(iters):
            sm = np.abs(c) < tau; c[sm] = 0; big = ~sm
            if big.sum() == 0: break
            Gb = A[:, big].T @ A[:, big] + ridge * len(A) * np.eye(big.sum())
            c[big] = np.linalg.solve(Gb, A[:, big].T @ b)
        v = np.zeros(P_); v[idx] = c; v[j] = 1.0
        out.append(v / np.linalg.norm(v))
    return out

def sindy_pi_tuned(Fm, rg, taus=(0.02, 0.05, 0.1, 0.2, 0.4), ridges=(1e-6, 1e-4, 1e-2)):
    """Select (tau, ridge, pivot) by held-out residual with a sparsity tie-break (one-standard-error style):
    fit on half the rows, score ||F_val v||^2 / ||v||^2 on the other half, choose the sparsest candidate within 10%
    of the best validation residual, then refit on all rows with the chosen (tau, ridge, pivot)."""
    m = len(Fm); perm = rg.permutation(m); a, b = perm[:m//2], perm[m//2:]
    cands = []
    for tau in taus:
        for rd in ridges:
            for j, v in enumerate(stlsq_pi(Fm[a], tau, rd)):
                r = float(np.mean((Fm[b] @ v) ** 2)); nnz = int((np.abs(v) > 1e-8).sum())
                cands.append((r, nnz, tau, rd, j))
    best = min(c[0] for c in cands)
    ok = [c for c in cands if c[0] <= 1.1 * best]
    r, nnz, tau, rd, j = min(ok, key=lambda c: (c[1], c[0]))
    return stlsq_pi(Fm, tau, rd)[j], (tau, rd, j, nnz)

def heldout_split(m, rg):
    perm = rg.permutation(m); return perm[:m//2], perm[m//2:]

def nn_loss(Q, Xd_, X_):
    pass

def nn_invariant_tuned(X, Xd, rg, widths=(12, 32), wds=(0.0, 1e-2)):
    """Choose width and weight decay by the held-out loss mean(Qdot^2)/Var(Q); refit on all rows."""
    a, b = heldout_split(len(X), rg); best = None
    for H in widths:
        for wd in wds:
            Q = nn_invariant(X[a], Xd[a], H=H, restarts=2, rg=rg, wd=wd)
            # held-out Lie derivative via finite differences of the fitted function along the flow
            eps = 1e-4
            qd = (Q(X[b] + eps * Xd[b]) - Q(X[b] - eps * Xd[b])) / (2 * eps)
            L = float(np.mean(qd ** 2) / (np.var(Q(X[b])) + 1e-6))
            if best is None or L < best[0]: best = (L, H, wd)
    _, H, wd = best
    return nn_invariant(X, Xd, H=H, restarts=3, rg=rg, wd=wd), (H, wd)

def kernel_invariant_tuned(X, Xd, rg, ells=(0.5, 1.0, 2.0), regs=(1e-4, 1e-3, 1e-2)):
    a, b = heldout_split(len(X), rg); best = None
    for ell in ells:
        for rgz in regs:
            Q = kernel_invariant(X[a], Xd[a], reg=rgz, ell_mult=ell)
            eps = 1e-4
            qd = (Q(X[b] + eps * Xd[b]) - Q(X[b] - eps * Xd[b])) / (2 * eps)
            L = float(np.mean(qd ** 2) / (np.var(Q(X[b])) + 1e-6))
            if best is None or L < best[0]: best = (L, ell, rgz)
    _, ell, rgz = best
    return kernel_invariant(X, Xd, reg=rgz, ell_mult=ell), (ell, rgz)
