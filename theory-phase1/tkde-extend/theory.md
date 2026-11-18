# ESO-GAD / ESO-GAD+ — Theory Plan

Target: WSDM (current submission) → TKDE (ESO-GAD+ extension).

**Framing constraints to preserve throughout.** Edge semantics are encoded into the
*operator* $\bm{L}_\eta$, not into the filter bank — any phrasing that says "into the
filter" undermines the core contribution. The frozen calibrator is a *deliberate design
choice* that buys Lipschitz-stable guarantees, not a limitation; the conclusion currently
calls it "the main drawback," which hands a reviewer the quote. Avoid describing ESO-GAD as
a "straightforward extension" of BWGNN.

---

## Carried forward from the current manuscript

### Proposition 1 (spectral validity) — self-loop assumption must be in the statement

Setting $\eta = 1$ with $\hat{q} \approx \tfrac{1}{2}$ degenerates $\bm{D}^{(\eta)}$ unless
the self-loop convention is assumed. This belongs in the theorem statement, not the
implementation notes.

```latex
\begin{proposition}[Spectral validity]
Let $\tilde{\bm{A}} = \bm{A} + \bm{I}$ and let $\hat{\bm{Q}} = [\hat q_{uv}]$ satisfy
$\hat q_{uv} = \hat q_{vu} \in [0,1]$ and $\hat q_{uu} \equiv 0$. Then for every
$\hat{\bm{q}} \in [0,1]^{|\mathcal{E}|}$ and every $\eta \in [0,1]$,
\[
  \bm{D}^{(\eta)} \succeq \bm{I} \succ 0,
  \qquad
  \lambda_i\!\left(\bm{L}_\eta\right) \in [0,2] \quad \text{for all } i .
\]
\end{proposition}
```

Note the current PDF writes $\lambda_i \subseteq [0,2]$; it should be $\in$, or be stated
over the spectrum as a set.

### Theorem 2 — the $\sqrt{g_k}$ parameterization is load-bearing

Keep $\bm{Z} = \big\Vert_{k=0}^{P} \sqrt{g_k}\,\bm{H}_k$. It gives
$\|\bm{Z}\|_F^2 = \sum_k g_k \|\bm{H}_k\|_F^2$, a convex combination of band energies, so
the gate reallocates energy without rescaling the representation and **cancels from the
stability bound entirely**. Linear weighting would yield $\sum_k g_k^2 \|\bm{H}_k\|_F^2$,
making the Lipschitz constant scale with $\|\bm{g}\|_2$ and letting the gate enter a bound
that should depend only on the operator and the graph.

Consequence to state explicitly: the KL term's role is in **optimization dynamics**
(preventing band starvation under a moving operator), not in the stability result. T2 below
gives it a second, statistical justification.

### Complexity claim — correct the Eq. (19) argument

The coefficients $\theta^{(k)}_j$ are fixed constants independent of $\eta$. What must be
recomputed at each $\eta$ update are the propagations $\bm{L}_\eta^{j}\bm{H}^{(0)}$. The
current §5.3.5 sentence states the opposite and is a technical error in your own complexity
claim. Correct framing: the operator rebuild recurs every epoch (unlike BWGNN's one-time
construction) but remains dominated by the $O(P\,m\,d_h)$ propagation term by a factor of
$P\,d_h$.

---

## T1 — Neyman orthogonality of the downstream objective

**Priority: load-bearing for TKDE.**

Treat $q$ as a **nuisance parameter** and the filter/gate parameters
$(\psi, \omega, \eta, \bm{a})$ as the target. You already use cross-fitting, which is the
machinery of double/debiased ML; the paper does not currently cash in on it.

**Statement to prove.** Under an orthogonality condition on $\eta$, the downstream excess
risk depends on calibration error at rate
$$
  O\!\left(\|\hat q - q^*\|^2\right)
  \quad\text{rather than}\quad
  O\!\left(\|\hat q - q^*\|\right).
$$

**Proof technique.** Second-order Taylor expansion of the downstream loss in $q$ about
$q^*$; cross-fitting supplies the independence that annihilates the first-order term.

**Prerequisite (blocking).** Cross-fitting folds must be **node-disjoint**, not
edge-disjoint. If a labeled node appears in two folds, out-of-fold probabilities are not
label-independent and the entire frozen/deterministic/Lipschitz story — Prop. 1, Thm. 1,
Thm. 2 — loses its footing. Add one sentence to §4.1 and one line to the appendix stating
the fold construction.

**What it buys.** Converts "we deliberately admit this approximation error because the
signed operator is designed to absorb the noisy estimate" from an assertion into a theorem,
and pre-empts the obvious "why not train the calibrator jointly?" objection by showing joint
training is what *breaks* orthogonality.

---

## T2 — PAC-Bayes bound on the spectral gate

**Priority: high, low cost — the objective already has the right shape.**

Eq. (23) is $\mathcal{L}_{\mathrm{BCE}} + \alpha D_{\mathrm{KL}}(\bm{g} \,\|\, \bm{g}_0)$,
which is literally a PAC-Bayes objective: $\bm{g}_0$ uniform is a prior over frequency
bands, $\alpha$ is the temperature, and the learned $\bm{g}$ is the posterior.

**Statement to prove.** A generalization bound on the node-level classification risk in
which the existing KL term *is* the bound's KL term, with $\alpha$ identified as the
inverse temperature.

**What it buys.** Promotes the KL regularizer from an engineering stabilizer to a
principled component with a stated role, and gives §5.3.3 (the Uniform / Learnable /
Learnable+KL ablation) a theoretical referent instead of leaving it purely empirical.

---

## T3 — Spectral separation with an explicit breakdown point

**Priority: load-bearing. This is the operator-level analogue of BWGNN's right-shift.**

**Setting.** A contextual SBM (or similar generative model) in which $q^*$ is available in
closed form.

**Statement to prove.** For a calibration satisfying $\|\hat q - q^*\|_\infty \le
\varepsilon$ and $\eta > \tfrac{1}{2}$, the Rayleigh-quotient gap between the
anomaly-indicator signal and the normal-class signal is strictly larger under $\bm{L}_\eta$
than under the unsigned $\bm{L}$, up to $O(\varepsilon)$.

**Required companion result — the breakdown point.** State the critical $\varepsilon^\dagger$
above which signing is *harmful* rather than helpful, as a function of $\eta$ and the
graph's heterophily rate.

**What it buys.** Two things. First, a positive result explaining *why* signing helps, which
the paper currently only demonstrates empirically. Second — and more valuable — the
strongest possible defense of adaptive $\eta$: show in E6 that the learned $\eta$ backs off
*before* $\varepsilon^\dagger$ is reached. That turns Fig. 3's inverted-U into predicted
behavior rather than a tuning curve.

Note the sign-flip threshold already in §4.2: $W^{(\eta)}_{uv} < 0$ requires
$\hat q_{uv} > 1/(2\eta)$, so no edge flips unless $\eta > \tfrac{1}{2}$. The manuscript
currently says "for $\eta > 0$, an edge is exactly pruned when $\hat q_{uv} = 1/(2\eta)$" —
that level set is only reachable for $\eta \ge \tfrac{1}{2}$. Fix alongside T3, since T3
depends on the same threshold.

---

## T4 — Spectral trajectory in $\eta$

**Priority: first candidate for cuts.**

Characterize how the band peaks of $\mathcal{W}_k(\bm{L}_\eta)$ move as $\eta$ sweeps
$[0,1]$: a Weyl-type monotonicity result on $\lambda_{\max}(\bm{L}_\eta)$, plus where each
Beta channel's response center lands.

**What it buys.** Turns Fig. 3 from a measured curve into a predicted one. Genuinely nice,
but not load-bearing — cut before T1 or T3 if the timeline compresses.

---

## T5 — Identifiability under perfect camouflage

**Priority: medium — defends the descriptor design.**

**Statement to prove.** Under perfect camouflage (anomalies matching the feature
distribution of their neighbors), no feature-only calibrator can identify $q^*$ better than
chance. Then show which structural descriptors in Eq. (5) — the degree asymmetry
$|\log\frac{1+d_u}{1+d_v}|$ and the overlap terms $J_{uv}$,
$|\mathcal{N}_u \cap \mathcal{N}_v| / \sqrt{d_u d_v}$ — break the degeneracy.

**What it buys.** Makes $\bm{z}^s_{uv}$ a principled necessity rather than an ad hoc feature
block. Pairs directly with the descriptor-block ablation (E5).

---

## Dependency order

```
node-disjoint folds  ──┬──> E1 (stratified calibrator validation) ──┬──> T1
                       │                                            │
                       └──> E5 (cross-fitting ablation) ────────────┘

T3 (breakdown point) ──────> E6 (contextual-SBM sweep) ──> adaptive-η defense

T2 ──────> reinterprets §5.3.3 (gating ablation)
T5 ──────> reinterprets E5 descriptor-block ablation
```

T1 and T3 are the two results that carry the TKDE version. T1 justifies the two-stage
architecture; T3 justifies the signing itself. If time forces a cut, lose T4 first, then T5.
