"""Numerical checks of the theory (Section 5 and Appendix A of the paper).

    python theory_check.py synthetic                 # every statement on random graphs, CPU
    python theory_check.py real --dataset amazon     # oracle diagnostics on a real graph

`synthetic` draws random two-class graphs with random symmetric scores and
eta in [0, 1.6] and checks every identity and bound. Each check is printed
with the id of the claim it belongs to (theory-phase1/theory.md, "Claims")
and the paper statement. Keep the three in sync: when a statement changes,
change theory.tex, proof.tex, the check here and the claims table together.

`real` uses ALL ground-truth labels and is analysis only.
"""
import argparse
import json
import math
import os

import numpy as np

# ----------------------------------------------------------------------------
# dense reference implementation of the operator (independent of the model code)
# ----------------------------------------------------------------------------


def random_graph(rng, n=60):
    """Two-class graph on n nodes, labels in {-1,+1} (+1 = anomaly), no isolated
    structure assumed. Returns (A~ with self-loops, y)."""
    pi = rng.uniform(0.15, 0.45)
    y = np.where(rng.rand(n) < pi, 1.0, -1.0)
    if (y > 0).sum() < 2:
        y[:2] = 1.0
    if (y < 0).sum() < 2:
        y[-2:] = -1.0
    p_in, p_out = rng.uniform(0.05, 0.3), rng.uniform(0.05, 0.3)
    same = np.equal.outer(y, y)
    upper = np.triu(rng.rand(n, n) < np.where(same, p_in, p_out), 1)
    A = (upper | upper.T).astype(float)
    return A + np.eye(n), y


def random_scores(rng, At, y, informative=None):
    """Symmetric scores in [0,1] on the edges of A~, zero on the diagonal."""
    n = At.shape[0]
    t = true_heterophily(At, y)
    if informative is None:
        informative = rng.rand() < 0.5
    raw = rng.rand(n, n)
    if informative:
        raw = np.clip(0.6 * t + 0.2 + 0.35 * rng.randn(n, n), 0, 1)
    q = np.triu(raw, 1)
    q = (q + q.T) * (At > 0)
    np.fill_diagonal(q, 0.0)
    return q


def true_heterophily(At, y):
    t = (np.not_equal.outer(y, y)).astype(float) * (At > 0)
    np.fill_diagonal(t, 0.0)
    return t


def operator(At, q, eta):
    """W = A~ (.) (1 - 2 eta q), D = diag(|W| 1), N = D^-1/2 W D^-1/2, L = I - N."""
    W = At * (1.0 - 2.0 * eta * q)
    D = np.abs(W).sum(1)
    N = W / np.sqrt(np.outer(D, D))
    return W, D, N, np.eye(At.shape[0]) - N


def kappa(k, P):
    return (P + 1) / 2.0 * math.comb(P, k)


def beta_channel(L, k, P):
    """W_k(L) = kappa_k (L/2)^k (I - L/2)^(P-k), the paper's normalization."""
    n = L.shape[0]
    half = L / 2.0
    return kappa(k, P) * np.linalg.matrix_power(half, k) @ np.linalg.matrix_power(np.eye(n) - half, P - k)


def frustration(W, y):
    s_star = np.outer(y, y)
    wrong = (np.sign(W) != s_star) & (W != 0)
    return (np.abs(W) * wrong).sum() / np.abs(W).sum()


def gauss_tail(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


# ----------------------------------------------------------------------------
# bookkeeping
# ----------------------------------------------------------------------------

class Checks:
    def __init__(self):
        self.rows = {}
        self.order = []

    def record(self, key, claim, statement, what, ok, value=None):
        if key not in self.rows:
            self.rows[key] = {'claim': claim, 'statement': statement, 'what': what, 'ok': True,
                              'worst': None, 'instances': 0}
            self.order.append(key)
        row = self.rows[key]
        row['ok'] = row['ok'] and bool(ok)
        row['instances'] += 1
        if value is not None:
            row['worst'] = value if row['worst'] is None else max(row['worst'], value)

    def report(self):
        passed = 0
        for i, key in enumerate(self.order, 1):
            r = self.rows[key]
            passed += r['ok']
            worst = '' if r['worst'] is None else '  [worst {:.3g}]'.format(r['worst'])
            print('{:>2}. {} {:<4} {:<26} {}{}'.format(i, 'PASS' if r['ok'] else 'FAIL', r['claim'], r['statement'],
                                                    r['what'], worst))
        print('\n{} of {} checks pass'.format(passed, len(self.order)))
        return passed, len(self.order)


# ----------------------------------------------------------------------------
# the checks
# ----------------------------------------------------------------------------

def check_instance(rng, ck, P=2, tol=1e-9):
    At, y = random_graph(rng)
    n = At.shape[0]
    Y = np.diag(y)
    t = true_heterophily(At, y)
    q = random_scores(rng, At, y)
    eta = rng.uniform(0.0, 1.6)
    deg = At.sum(1)
    h = t.sum(1) / deg
    W, D, N, L = operator(At, q, eta)
    ytil = np.sqrt(D) * y

    # ---- Proposition 1 ------------------------------------------------------
    ck.record('p1a', 'C1', 'Prop 1', 'D >= I for every eta in [0, 1.6]', D.min() >= 1 - tol)
    ev = np.linalg.eigvalsh(L)
    ck.record('p1b', 'C1', 'Prop 1', 'spec(L_eta) in [0, 2]', ev.min() >= -1e-9 and ev.max() <= 2 + 1e-9,
              max(-ev.min(), ev.max() - 2))

    # ---- Theorem 1 ----------------------------------------------------------
    eo = rng.uniform(0.5, 1.0)
    Wo, Do, No, Lo = operator(At, t, eo)
    Wc, Dc, Nc, Lc = operator(At, t, 1.0 - eo)
    ck.record('t1a', 'C3', 'Thm 1(i)', 'L_eta(t*) = Y L_{1-eta}(t*) Y, right side non-negative',
              np.abs(Lo - Y @ Lc @ Y).max() < tol and Wc.min() >= -tol, np.abs(Lo - Y @ Lc @ Y).max())
    spec_gap = np.abs(np.linalg.eigvalsh(Lo) - np.linalg.eigvalsh(Lc)).max()
    chan_gap = max(np.abs(beta_channel(Lo, k, P) - Y @ beta_channel(Lc, k, P) @ Y).max() for k in range(P + 1))
    ck.record('t1b', 'C3', 'Thm 1(ii)', 'same spectrum; W_k(L_eta) = Y W_k(L_{1-eta}) Y',
              spec_gap < 1e-8 and chan_gap < 1e-8, max(spec_gap, chan_gap))
    eo2 = rng.uniform(0.5 + 1e-3, 1.0)
    _, Do2, _, Lo2 = operator(At, t, eo2)
    ck.record('t1c', 'C3', 'Thm 1(iii)', 'L_eta(t*) D^1/2 y = 0', np.abs(Lo2 @ (np.sqrt(Do2) * y)).max() < 1e-9,
              np.abs(Lo2 @ (np.sqrt(Do2) * y)).max())

    # ---- Theorem 2 ----------------------------------------------------------
    rho = ytil @ L @ ytil / (ytil @ ytil)
    Fr = frustration(W, y)
    ck.record('t2', 'C2', 'Thm 2', 'label frequency = 2 x frustration', abs(rho - 2 * Fr) < 1e-10, abs(rho - 2 * Fr))

    # ---- Corollary 1 --------------------------------------------------------
    het = t > 0
    Hn, M = het.sum(), (At > 0).sum()
    S_het, S_all = q[het].sum(), q[At > 0].sum()
    W0, D0, N0, L0 = operator(At, q, 0.0)
    y0 = np.sqrt(D0) * y
    ck.record('c1a', 'C5', 'Cor 1', 'rho_0 = 2H/M (twice the edge heterophily)',
              abs(y0 @ L0 @ y0 / (y0 @ y0) - 2 * Hn / M) < 1e-10)
    e1 = rng.uniform(0.0, 0.5)
    closed = (Hn - 2 * e1 * S_het) / (M - 2 * e1 * S_all)
    ck.record('c1b', 'C5', 'Cor 1', 'closed form of Fr_eta on [0, 1/2]',
              abs(frustration(operator(At, q, e1)[0], y) - closed) < 1e-10)
    lift = S_het / max(Hn, 1) - S_all / M
    step = 1e-5
    e2 = rng.uniform(0.0, 0.5 - step)
    slope = (frustration(operator(At, q, e2 + step)[0], y) - frustration(operator(At, q, e2)[0], y)) / step
    ck.record('c1c', 'C5', 'Cor 1', 'Fr_eta decreases on [0, 1/2] iff the lift is positive',
              abs(lift) < 1e-6 or (slope < 0) == (lift > 0))
    e3 = rng.uniform(0.5, 1.6)
    hom = (At > 0) & ~het
    num = np.clip(1 - 2 * e3 * q[het], 0, None).sum() + np.clip(2 * e3 * q[hom] - 1, 0, None).sum()
    ck.record('c1d', 'C5', 'Cor 1 (text)', 'formula of Fr_eta for eta > 1/2',
              abs(frustration(operator(At, q, e3)[0], y) - num / np.abs(1 - 2 * e3 * q[At > 0]).sum()) < 1e-10)

    # ---- Corollary 2 --------------------------------------------------------
    def energy(Lm, vec, k):
        return np.linalg.norm(beta_channel(Lm, k, P) @ vec) ** 2 / (kappa(k, P) ** 2 * (vec @ vec))
    E0, EP = energy(L, ytil, 0), energy(L, ytil, P)
    ck.record('c2a', 'C4', 'Cor 2', '(1-Fr)^2P <= E_0 <= 1-Fr', (1 - Fr) ** (2 * P) - 1e-10 <= E0 <= 1 - Fr + 1e-10)
    ck.record('c2b', 'C4', 'Cor 2', 'Fr^2P <= E_P <= Fr', Fr ** (2 * P) - 1e-10 <= EP <= Fr + 1e-10)
    yo = np.sqrt(Do2) * y
    ck.record('c2c', 'C4', 'Cor 2 (text)', 'oracle signs: E_0 = 1 and E_k = 0 for k >= 1',
              abs(energy(Lo2, yo, 0) - 1) < 1e-9 and all(energy(Lo2, yo, k) < 1e-9 for k in range(1, P + 1)))

    # ---- Theorem 3 ----------------------------------------------------------
    def chi_of(G):
        return y * (G @ y) / np.linalg.norm(G, axis=1)
    G = W / D[:, None]
    margin = (W * np.outer(y, y)).sum(1) / np.abs(W).sum(1)
    d_eff = np.abs(W).sum(1) ** 2 / (W ** 2).sum(1)
    ck.record('t3a', 'C6', 'Thm 3(a)', 'one step: chi_u = m_u sqrt(d_eff,u)',
              np.abs(chi_of(G) - margin * np.sqrt(d_eff)).max() < 1e-9)
    chi_uns = chi_of(At / deg[:, None])
    Wp = At * (1 - t)
    chi_pr = chi_of(Wp / Wp.sum(1)[:, None])
    Ws = At * (1 - 2 * t)
    chi_sg = chi_of(Ws / np.abs(Ws).sum(1)[:, None])
    ok = (np.abs(chi_uns - (1 - 2 * h) * np.sqrt(deg)).max() < 1e-9
          and np.abs(chi_pr - np.sqrt((1 - h) * deg)).max() < 1e-9
          and np.abs(chi_sg - np.sqrt(deg)).max() < 1e-9)
    ck.record('t3b', 'C6', 'Thm 3(b)', 'oracle table: unsigned, pruning, signing', ok)
    G0 = np.linalg.matrix_power((np.eye(n) + N0) / 2.0, P)
    Ns = operator(At, t, 1.0)[2]
    Gs = np.linalg.matrix_power((np.eye(n) + Ns) / 2.0, P)
    lhs, rhs = chi_of(Gs), np.abs(G0).sum(1) / np.linalg.norm(G0, axis=1)
    both = np.array([len(set(y[G0[u] > 1e-14])) == 2 for u in range(n)])
    strict = lhs - np.abs(chi_of(G0)) > 1e-10
    ck.record('t3c', 'C7', 'Thm 3(c)', 'oracle signing dominates the unsigned low-pass filter, strictly iff mixed support',
              np.abs(lhs - rhs).max() < 1e-9 and (lhs >= np.abs(chi_of(G0)) - 1e-10).all() and (strict == both).all())
    def chi_eta(e):
        We = At * (1 - 2 * e * q)
        return y * ((We / np.abs(We).sum(1)[:, None]) @ y) * np.sqrt(np.abs(We).sum(1) ** 2 / (We ** 2).sum(1))
    fd = (chi_eta(1e-6) - chi_eta(-1e-6)) / 2e-6
    hetn, homn = t.sum(1), (At - t).sum(1)
    q_het = np.where(hetn > 0, (q * t).sum(1) / np.maximum(hetn, 1), 0.0)
    q_hom = (q * (At - t)).sum(1) / homn
    pred = 4 * np.sqrt(deg) * h * (1 - h) * (q_het - q_hom)
    ck.record('t3d', 'C9', 'Thm 3(d)', 'd chi_u / d eta at 0, against finite differences',
              np.abs(fd - pred).max() < 1e-4, np.abs(fd - pred).max())

    # ---- Theorem 4 ----------------------------------------------------------
    u = rng.randint(n)
    nb = np.where(At[u] > 0)[0]
    slope0 = rng.uniform(0.4, 1.0)
    p = np.clip(slope0 * q[u, nb], 0, 1)
    p[nb == u] = 0.0
    c = 1 - 2 * p
    mu_norm, sigma = rng.uniform(0.2, 1.5), rng.uniform(0.5, 2.0)
    tau = sigma ** 2 + mu_norm ** 2 * (1 - c ** 2)
    def J(w):
        return mu_norm * (w @ c) / math.sqrt((w ** 2) @ tau)
    J_star = mu_norm * math.sqrt((c ** 2 / tau).sum())
    J_plus = mu_norm * math.sqrt((np.clip(c, 0, None) ** 2 / tau).sum())
    rand_w = [rng.randn(len(nb)) for _ in range(20)]
    ck.record('t4a', 'C8', 'Thm 4(i)', 'J <= J*, attained by weights c/tau',
              all(J(w) <= J_star + 1e-10 for w in rand_w) and abs(J(c / tau) - J_star) < 1e-10)
    pos_w = [np.abs(w) for w in rand_w]
    ck.record('t4b', 'C8', 'Thm 4(ii)', 'non-negative weights: J <= J*_+, attained by (c)_+/tau',
              all(J(w) <= J_plus + 1e-10 for w in pos_w) and abs(J(np.clip(c, 0, None) / tau) - J_plus) < 1e-10)
    gap = mu_norm ** 2 * (c[p > 0.5] ** 2 / tau[p > 0.5]).sum()
    ck.record('t4c', 'C8', 'Thm 4(iii)', 'gap J*^2 - J*_+^2', abs(J_star ** 2 - J_plus ** 2 - gap) < 1e-12,
              abs(J_star ** 2 - J_plus ** 2 - gap))
    # Corollary 3: in the low-signal regime the family 1 - 2 eta_0 q contains the optimum
    tau_low = sigma ** 2 + (1e-3 * sigma) ** 2 * (1 - c ** 2)
    ratio = ((c @ c) / math.sqrt((c ** 2) @ tau_low)) / math.sqrt((c ** 2 / tau_low).sum())
    ck.record('c3', 'C8', 'Cor 3', 'linear reliability: weights 1 - 2 eta_0 q are optimal at low signal',
              ratio > 1 - 1e-6, 1 - ratio)

    # ---- Corollary 4 --------------------------------------------------------
    touches = (np.add.outer(y > 0, y > 0) > 0) * (At > 0)
    We1 = At * np.where(touches, -1.0, 1.0)
    np.fill_diagonal(We1, 1.0)
    m_e1 = (We1 * np.outer(y, y)).sum(1) / np.abs(We1).sum(1)
    an = y > 0
    ok = (np.abs(m_e1[~an] - 1).max() < 1e-12 and np.abs(m_e1[an] - (2 * h[an] - 1 + 2 / deg[an])).max() < 1e-12
          and ((m_e1[an] > 1 - 2 * h[an] + 1e-12) == (h[an] > 0.5 - 0.5 / deg[an] + 1e-12)).all())
    ck.record('c4', 'C13', 'Cor 4', 'ideal E1 operator: margins, and it helps an anomaly iff h_u > 1/2 - 1/(2 d_u)', ok)

    # ---- Theorems 5-7, Lemma 2, Corollary 5 ----------------------------------
    eps = rng.choice([0.02, 0.1, 0.3])
    noise = np.triu(eps * rng.randn(n, n), 1)
    q2 = np.clip(q + (noise + noise.T) * (At > 0), 0, 1)
    np.fill_diagonal(q2, 0.0)
    eta2 = max(eta + rng.uniform(-0.1, 0.1), 0.0)
    W2, D2, N2, L2 = operator(At, q2, eta2)
    delta = (np.abs(W - W2).sum(1) / D).max()
    gamma = (D / deg).min()
    eps_loc = ((np.abs(q - q2) * At).sum(1) / deg).max()
    qbar = ((q2 * At).sum(1) / deg).max()
    dL = np.linalg.norm(L - L2, 2)
    ck.record('t5a', 'C10', 'Thm 5', '||L - L\'||_2 <= 3 delta + delta^2', dL <= 3 * delta + delta ** 2 + 1e-10,
              dL / (3 * delta + delta ** 2))
    ck.record('t5b', 'C10', 'Thm 5', 'delta <= (2/gamma)(eta eps_loc + |eta - eta\'| qbar\')',
              delta <= 2 / gamma * (eta * eps_loc + abs(eta - eta2) * qbar) + 1e-10,
              delta / (2 / gamma * (eta * eps_loc + abs(eta - eta2) * qbar)))
    d_max = deg.max() - 1
    wsdm = 2 * d_max * (1 + d_max) * (eta * np.abs(q - q2).max() + abs(eta - eta2))
    ck.record('t5c', 'C10', 'Thm 5 (text)', 'tighter than the WSDM bound 2 d_max (1 + d_max)(...) [worst = smallest factor, negated]',
              3 * delta + delta ** 2 <= wsdm + 1e-10, -(wsdm / (3 * delta + delta ** 2)))

    W3, D3, N3, L3 = operator(At, q2, eta)            # same eta, for Theorems 6 and 7
    delta_s = (np.abs(W - W3).sum(1) / D).max()
    B = (((q - q2) ** 2 * At).sum(1) / deg).sum()
    lhs6, rhs6 = np.linalg.norm(L - L3, 'fro'), 2 * eta / gamma * (1 + math.sqrt(2) * (2 + delta_s)) * math.sqrt(B)
    ck.record('t6', 'C11', 'Thm 6', 'Frobenius distance bounded by the local Brier score', lhs6 <= rhs6 + 1e-10,
              lhs6 / rhs6 if rhs6 > 0 else 0.0)
    H0 = rng.randn(n, 8)
    gate = rng.dirichlet(np.ones(P + 1))
    Z1 = np.concatenate([math.sqrt(gate[k]) * beta_channel(L, k, P) @ H0 for k in range(P + 1)], 1)
    Z2 = np.concatenate([math.sqrt(gate[k]) * beta_channel(L3, k, P) @ H0 for k in range(P + 1)], 1)
    lam = max(P / 2.0 * kappa(k, P) for k in range(P + 1))
    rhs7 = lam * (3 * delta_s + delta_s ** 2) * np.linalg.norm(H0, 'fro')
    ck.record('t7', 'C12', 'Thm 7', 'representation robustness', np.linalg.norm(Z1 - Z2, 'fro') <= rhs7 + 1e-10,
              np.linalg.norm(Z1 - Z2, 'fro') / rhs7 if rhs7 > 0 else 0.0)
    grid = np.linspace(0, 2, 4001)
    worst = 0.0
    for k in range(P + 1):
        f = kappa(k, P) * (grid / 2) ** k * (1 - grid / 2) ** (P - k)
        lip = np.abs(np.gradient(f, grid)).max()
        worst = max(worst, np.linalg.norm(beta_channel(L, k, P) - beta_channel(L3, k, P), 'fro')
                    / max(lip * np.linalg.norm(L - L3, 'fro'), 1e-300))
    ck.record('l2', 'C12', 'Lemma 2', 'Frobenius Lipschitz continuity of the kernels', worst <= 1 + 1e-6, worst)

    e5 = rng.uniform(0.55, 1.0)
    small = np.triu(rng.choice([0.01, 0.05, 0.2]) * np.abs(rng.randn(n, n)), 1)
    q_hat = np.clip(t + (1 - 2 * t) * (small + small.T) * (At > 0), 0, 1)
    np.fill_diagonal(q_hat, 0.0)
    _, Ds, _, Ls = operator(At, t, e5)
    _, _, _, Lh = operator(At, q_hat, e5)
    ys = np.sqrt(Ds) * y
    resp = beta_channel(Lh, 0, P) @ ys
    frac = Ds[np.sign(resp) != np.sign(ys)].sum() / Ds.sum()
    d5 = 2 * e5 * ((np.abs(q_hat - t) * At).sum(1) / deg).max() / (Ds / deg).min()
    bound = (P / 2.0 * (3 * d5 + d5 ** 2)) ** 2
    ck.record('c5', 'C12', 'Cor 5', 'share of misaligned nodes below the bound', frac <= bound + 1e-12,
              frac / bound if bound > 0 else 0.0)

    # ---- Theorem 8, Corollary 6: the Brier score controls separability --------
    S = np.outer(y, y) * At
    worst_ii, ok_i, ok_ii, ok_c6a = 0.0, True, True, True
    for q_c, eta_c in ((q, eta), (q_hat, e5)):            # arbitrary scores, and scores close to the oracle
        Wc8 = At * (1.0 - 2.0 * eta_c * q_c)
        chi8 = chi_of(Wc8 / np.abs(Wc8).sum(1)[:, None])
        cos = (Wc8 * S).sum(1) / (np.linalg.norm(Wc8, axis=1) * np.sqrt(deg))
        ok_i = ok_i and np.abs(chi8 - np.sqrt(deg) * cos).max() < 1e-9
        b8 = ((eta_c * q_c - t) ** 2 * At).sum(1) / deg
        good = b8 < 0.25
        ok_ii = (ok_ii and (chi8 ** 2 >= deg * (1 - 4 * b8) - 1e-9).all()
                 and (chi8[good] >= np.sqrt(deg[good] * (1 - 4 * b8[good])) - 1e-9).all())
        worst_ii = max(worst_ii, ((deg - chi8 ** 2) / np.maximum(4 * deg * b8, 1e-12)).max())
        beat = b8 <= h * (1 - h)
        ok_c6a = ok_c6a and (np.abs(chi8[beat]) >= np.abs(1 - 2 * h[beat]) * np.sqrt(deg[beat]) - 1e-9).all()
        if q_c is q:
            bound8 = 2 * (deg * b8).sum() / np.abs(Wc8).sum()
            rho8 = 2 * frustration(Wc8, y)
    ck.record('t8a', 'C15', 'Thm 8(i)', 'chi_u = sqrt(d_u) cos(theta_u)', ok_i)
    ck.record('t8b', 'C15', 'Thm 8(ii)', 'chi_u^2 >= d_u (1 - 4 b_u); correct side when b_u < 1/4', ok_ii, worst_ii)
    ck.record('t8c', 'C15', 'Thm 8(iii)', 'rho_eta <= 2 sum_u d_u b_u / sum_u D_uu, with equality at eta = 0',
              rho8 <= bound8 + 1e-10 and abs(2 * frustration(At.copy(), y) - 2 * ((0.0 * q - t) ** 2 * At).sum() / M) < 1e-12,
              rho8 / bound8)
    ck.record('c6a', 'C16', 'Cor 6(i)', '|chi_u| at least its unsigned value when b_u <= h_u (1 - h_u)', ok_c6a)
    flips = np.triu(rng.rand(n, n) < rng.uniform(0.0, 0.5), 1)
    Wh = At * np.where(flips | flips.T, -S, S)
    np.fill_diagonal(Wh, 1.0)
    eps_u = ((Wh != S) & (At > 0)).sum(1) / deg
    ok = (np.abs(chi_of(Wh / np.abs(Wh).sum(1)[:, None]) - (1 - 2 * eps_u) * np.sqrt(deg)).max() < 1e-9
          and abs(2 * frustration(Wh, y) - 2 * (eps_u * deg).sum() / M) < 1e-10)
    ck.record('c6b', 'C16', 'Cor 6(ii)', 'hard signs: chi_u = (1 - 2 eps_u) sqrt(d_u), label frequency = 2 eps', ok)
    total = lambda e: ((e * q - t) ** 2 * At).sum()
    eta_ls = (q * t).sum() / (q ** 2).sum()
    sin2 = 1 - (q * t).sum() ** 2 / ((q ** 2).sum() * Hn)
    ck.record('c6c', 'C16', 'Cor 6(iii)', 'least-squares slope minimises the total local Brier score, at H sin^2',
              total(eta_ls) <= min(total(eta_ls + 1e-3), total(eta_ls - 1e-3)) and abs(total(eta_ls) - Hn * sin2) < 1e-8
              and abs(total(0.0) - Hn) < 1e-9)

    # ---- channels sum to a multiple of the identity (theory.md, N9) ----------
    total = sum(beta_channel(L, k, P) for k in range(P + 1))
    ck.record('n9', 'N9', 'identity (not in paper)', 'sum_k W_k(L_eta) = (P+1)/2 I for every eta and q',
              np.abs(total - (P + 1) / 2.0 * np.eye(n)).max() < 1e-9)


def check_monte_carlo(rng, ck, samples=20000):
    """Theorem 3(a) and Theorem 4(iv) by simulation on one graph."""
    At, y = random_graph(rng)
    n = At.shape[0]
    q = random_scores(rng, At, y)
    eta = rng.uniform(0.2, 1.2)
    W, D, N, L = operator(At, q, eta)
    G = W / D[:, None]
    chi = y * (G @ y) / np.linalg.norm(G, axis=1)
    mu = np.array([0.6, -0.3, 0.2])
    sigma = 1.0
    nodes = rng.choice(n, 6, replace=False)
    ok, worst = True, 0.0
    for u in nodes:
        # <z_u, mu>/|mu| ~ N([Gy]_u |mu|, sigma^2 |G_u|^2): simulate the feature noise directly
        xi = sigma * rng.randn(samples, n, len(mu))
        z = (G[u] @ y) * mu + np.einsum('v,svd->sd', G[u], xi)
        err = ((z @ mu) * y[u] <= 0).mean()
        pred = gauss_tail(chi[u] * np.linalg.norm(mu) / sigma)
        se = math.sqrt(max(pred * (1 - pred), 1e-6) / samples)
        worst = max(worst, abs(err - pred) / se)
        ok = ok and abs(err - pred) < 4 * se + 1e-3
    ck.record('t3mc', 'C6', 'Thm 3(a)', 'error rate of the readout = Phi-bar(chi |mu|/sigma), Monte Carlo [worst in s.e.]',
              ok, worst)

    u = int(nodes[0])
    nb = np.where(At[u] > 0)[0]
    p = np.clip(0.8 * q[u, nb], 0, 1)
    p[nb == u] = 0.0
    c = 1 - 2 * p
    mu_norm = 1.0
    tau = sigma ** 2 + mu_norm ** 2 * (1 - c ** 2)
    w = c / tau
    signs = np.where(rng.rand(samples * 5, len(nb)) < p, -1.0, 1.0)
    proj = mu_norm * (signs @ w) + sigma * (rng.randn(samples * 5, len(nb)) @ w)     # y_u <z_u, mu-hat>
    J_emp = proj.mean() / proj.std()
    J_th = mu_norm * (w @ c) / math.sqrt((w ** 2) @ tau)
    ck.record('t4mc', 'C8', 'Thm 4', 'deflection formula, Monte Carlo', abs(J_emp - J_th) < 0.03 * max(J_th, 1),
              abs(J_emp - J_th))
    ck.record('t4d', 'C8', 'Thm 4(iv)', 'error <= 1/(1 + J^2) (Cantelli), Monte Carlo',
              (proj <= 0).mean() <= 1 / (1 + J_th ** 2) + 1e-3, (proj <= 0).mean() * (1 + J_th ** 2))


def check_prop2(rng, ck, edges=3_000_000):
    """Proposition 2 by simulation: edges of five descriptor types with their own
    class-configuration probabilities; a node of class c is labeled with
    probability r_c, independently."""
    types = 5
    probs = rng.dirichlet(np.ones(3), size=types)          # P_nn, P_an (= q*), P_aa per type
    z = rng.randint(types, size=edges)
    cfg = (rng.rand(edges)[:, None] > probs.cumsum(1)[z]).sum(1)      # 0 = nn, 1 = an, 2 = aa
    first_is_a = np.where(cfg == 1, rng.rand(edges) < 0.5, cfg == 2)
    second_is_a = np.where(cfg == 1, ~first_is_a, cfg == 2)
    for name, r1, r0 in (('uniform', 0.3, 0.3), ('class-dependent', 0.4, 0.1)):
        lab1 = rng.rand(edges) < np.where(first_is_a, r1, r0)
        lab2 = rng.rand(edges) < np.where(second_is_a, r1, r0)
        one = lab1 ^ lab2
        target = np.where(lab1, first_is_a, second_is_a)
        worst = 0.0
        for k in range(types):
            m = one & (z == k)
            emp = target[m].mean()
            p_nn, q_star, p_aa = probs[k]
            odds = (r1 * (2 * (1 - r1) * p_aa + (1 - r0) * q_star)) / (r0 * (2 * (1 - r0) * p_nn + (1 - r1) * q_star))
            pred = odds / (1 + odds)
            if name == 'uniform':
                assert abs(pred - (0.5 * q_star + p_aa)) < 1e-12     # the two parts of the proposition agree
            worst = max(worst, abs(emp - pred))
        ck.record('p2' + name, 'C13', 'Prop 2', 'E1 target under {} labeling, simulation'.format(name),
                  worst < 3e-3, worst)


def run_synthetic(graphs=40, seed=0):
    rng = np.random.RandomState(seed)
    ck = Checks()
    for i in range(graphs):
        check_instance(rng, ck, P=2 if i % 2 == 0 else 3)
    check_monte_carlo(rng, ck)
    check_prop2(rng, ck)
    passed, total = ck.report()
    print('Proposition 3 is a property of the cross-fitting procedure and has no numerical check.')
    return ck, passed == total


# ----------------------------------------------------------------------------
# real graphs
# ----------------------------------------------------------------------------

def run_real(dataset, draws=5, out=None, device=None, symmetrize=True, edge_epochs=150):
    """Oracle facts of one dataset (experiment B7, C1, C2 previews), and what a
    calibrator learns under GADBench's label budget for three targets. Uses
    ALL labels; analysis only."""
    import torch
    import gadbench
    import diagnostics as dg
    import main_gadbench as mg

    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    graph = gadbench.load_graph(dataset, symmetrize=symmetrize)
    labels = graph.ndata['label']
    known = gadbench.known_label_mask(graph)
    src, dst = graph.edges()
    n = graph.num_nodes()
    sl = src == dst
    valid = dg.valid_edges(src, dst, known)
    t = dg.true_heterophily(src, dst, labels)
    an, no = known & (labels == 1), known & (labels == 0)
    h, deg = dg.local_heterophily(src, t, n, valid)

    def chi(w):
        return dg.node_factors(src, w, t, n, valid)[2]
    zero = torch.zeros_like(t)
    factors = {'unsigned': chi(dg.edge_weight(zero, 0.0, self_loop=sl)),
               'oracle_prune': chi(dg.edge_weight(t, None, 'prune', 0.5, sl)),
               'oracle_sign': chi(dg.edge_weight(t, 1.0, self_loop=sl))}
    e_ok = valid & ~sl
    a_s, a_d = labels[src] == 1, labels[dst] == 1
    facts = {
        'dataset': dataset, 'nodes_known': int(known.sum()), 'directed_edges_valid': int(valid.sum()),
        'anomaly_rate': float(an.sum()) / float(known.sum()),
        'rho_0': 2.0 * dg.frustration(dg.edge_weight(zero, 0.0, self_loop=sl), t, valid),
        'edge_heterophily_no_self_loops': float(t[e_ok].mean()),
        'h_anomalies': float(h[an].mean()), 'h_normals': float(h[no].mean()),
        'anomalies_h_above_half': float((h[an] > 0.5).float().mean()),
        'anomalies_below_cor4_threshold': int((h[an] < 0.5 - 0.5 / deg[an]).sum()),
        'class_mean_separation_unsigned': 2.0 * abs(1.0 - float(h[an].mean()) - float(h[no].mean())),
        'chi_anomalies': {k: float(v[an].mean()) for k, v in factors.items()},
        'chi_normals': {k: float(v[no].mean()) for k, v in factors.items()},
        'aa_share_of_anomaly_incident_edges': float((e_ok & a_s & a_d).sum()) / float((e_ok & (a_s | a_d)).sum()),
    }
    print(json.dumps(facts, indent=1))

    rows = []
    targets = (('E2 only', dict(edge_target='e2')),
               ('E1+E2 (paper rule)', dict(edge_target='e1e2')),
               ('E1+E2 with true labels', dict(edge_target='e1e2', oracle_lu=True)))
    for d in range(draws):
        masks = gadbench.random_split(labels, known, np.random.RandomState(d))
        labeled = masks[0]
        unl_pair = ~labeled[src] & ~labeled[dst]
        for name, kw in targets:
            from edge_calibration import cross_fitted_edge_calibration
            gadbench.set_seed(d)
            q, diag = cross_fitted_edge_calibration(
                graph, graph.ndata['feature'], labels, labeled, epochs=edge_epochs, seed=d, device=device,
                pos_weight=False, train_on_self_loops=False, lambda_p=0.0, **kw)
            s = dg.summarize_scores(graph, q, labels, known, eval_edges=unl_pair)
            s.update({'draw': d, 'target': name, 'num_E2': diag['num_E_LL'], 'num_E1': diag['num_E_LU'],
                      'folds_trained': diag['num_trained_folds']})
            rows.append(s)
            print('draw {} {:<24} E2 {} E1 {} | edge AUROC {} Brier {:.3f} | NN/AN/AA {:.2f}/{:.2f}/{:.2f} | '
                  'rho at eta=1 {:.3f}, min {:.3f} at {:.2f} | slope {:.2f}'.format(
                      d, name, diag['num_E_LL'], diag['num_E_LU'],
                      '{:.3f}'.format(s['edge']['auroc']) if 'auroc' in s['edge'] else 'n/a', s['edge']['brier'],
                      s['edge']['mean_score_nn'], s['edge']['mean_score_an'], s['edge']['mean_score_aa'],
                      s['label_freq']['by_eta']['1.0'], s['label_freq']['min_value'], s['label_freq']['min_eta'],
                      s['edge']['slope']), flush=True)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, 'w') as f:
            json.dump({'facts': facts, 'calibrators': rows}, f, indent=1)
        print('written to', out)
    return facts, rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)
    s = sub.add_parser('synthetic')
    s.add_argument('--graphs', type=int, default=40)
    s.add_argument('--seed', type=int, default=0)
    r = sub.add_parser('real')
    r.add_argument('--dataset', type=str, required=True)
    r.add_argument('--draws', type=int, default=5)
    r.add_argument('--edge_epochs', type=int, default=150)
    r.add_argument('--out', type=str, default=None)
    r.add_argument('--device', type=str, default=None)
    r.add_argument('--no_symmetrize', action='store_true')
    args = parser.parse_args()
    if args.mode == 'synthetic':
        _, all_ok = run_synthetic(args.graphs, args.seed)
        raise SystemExit(0 if all_ok else 1)
    run_real(args.dataset, args.draws, args.out, args.device, not args.no_symmetrize, args.edge_epochs)
