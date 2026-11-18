# ESO-GAD / ESO-GAD+ — Experiment Plan

Two tracks. **Track A** is re-analysis of runs already in hand and should land before the
WSDM deadline. **Track B** is new work for the TKDE ESO-GAD+ version.

Protocol baseline throughout: GADBench semi-supervised split (100 labeled nodes, 20
anomalies), 10 random splits, AUROC + AUPRC (average precision).

---

## The vulnerability these experiments neutralize

A reviewer will ask whether the gain comes from edge semantics or from smuggling the 100
training labels into the operator. Eq. (9) assigns singly-labeled edges the target $y_u$ of
the labeled endpoint — so on $\mathcal{E}_1$, which dominates $\mathcal{E}_2$ under 100
labels, the calibrator is trained to predict *"is my labeled endpoint anomalous."* It is a
node-level anomaly classifier in edge-level costume, and $\bm{L}_\eta$ then broadcasts that
node score into the operator. The alternative hypothesis is that ESO-GAD is label
propagation with extra steps.

E1 and E2 exist to kill that hypothesis. They are the highest-value experiments in the
entire plan relative to their cost.

---

# Track A — before the WSDM deadline

## E1 — Stratified calibrator validation

**What.** Held-out AUC and AUPRC of $\hat q$ reported **separately** on three edge strata:

| Stratum | Definition | Why it matters |
|---|---|---|
| $\mathcal{E}_2$ | both endpoints labeled | clean targets; upper bound on calibrator quality |
| $\mathcal{E}_1$ | exactly one endpoint labeled | imputed targets; where the label-propagation worry lives |
| $\mathcal{E}_0$ | neither endpoint labeled | **where the operator does most of its work** |

**Why.** $\mathcal{E}_0$ is where the imputed targets create distribution shift and where
nothing in the current paper demonstrates generalization. If $\hat q$ generalizes to
$\mathcal{E}_0$, the label-propagation reading dies.

**Also fix in the same pass.** Eq. (7) defines $\mathcal{E}_1$ with "or," making
$\mathcal{E}_2 \subset \mathcal{E}_1$ — so "$\mathcal{E}_2 \cup \mathcal{E}_1$" is redundant
and Eq. (9)'s case split is ill-posed for doubly-labeled edges. Redefine $\mathcal{E}_1$ as
*exactly one* endpoint labeled. Strata above assume the corrected definition.

**Cost.** Re-analysis only.

## E2 — Degenerate-calibrator control

**What.** Replace the edge-descriptor MLP with
$$
  \hat q_{uv} \;=\; |\hat s_u - \hat s_v|
$$
where $\hat s$ comes from a node-only anomaly classifier trained on the same 100 labels.
Everything downstream unchanged.

**Why.** If full ESO-GAD beats this, the *edge* descriptors $\bm{z}_{uv}$ are earning their
place and the method is not a re-skinned node classifier.

**Pairs with.** The matched-estimator ablation (signed-but-uncalibrated $\hat q$), which
separates calibration effects from signing and gating effects. That ablation is
non-negotiable — the "sign, don't prune" claim does not survive review without it.

**Cost.** One additional training run per dataset per split.

## E3 — Significance testing and Recall@K

**What.** Paired Wilcoxon signed-rank test over the 10 GADBench splits, ESO-GAD vs. each
baseline. Add **Recall@K** as a third metric alongside AUROC/AUPRC.

**Why.** Both are standard under this protocol and their absence is noticed. It also lets
§5.2 drop "particularly significant," which currently appears with no test behind it.

**Cost.** Pure re-analysis of logged predictions.

## E4 — Table hygiene (blocking, not optional)

Not an experiment, but gates everything above. Current defects:

- **Duplicated baseline rows.** Table 2: PCGNN/GraphSAGE on Reddit both `58.59±6.35`;
  DSGAD/GCN on Tolokers both `66.35±2.75`. Table 3: PCGNN/GraphSAGE on Reddit both
  `4.32±0.77`; DSGAD/GCN on Tolokers both `33.60±2.19`; PCGNN/GraphSAGE on Weibo share
  `±9.57` and `±5.78`. Two independent methods matching on *both* mean and standard
  deviation across 10 splits reads as copy-paste. Re-run or re-transcribe.
- **Std mismatch.** Amazon AUROC `94.12±1.37` (Tables 2, 5) vs. `94.12±1.27` (Table 4).
- **Claimed gains contradict the tables.** §5.2 says the largest BWGNN gains are on
  T-Finance, YelpChi, Tolokers, Questions. Actual AUROC deltas: Questions +8.36,
  **Reddit +8.40**, YelpChi +5.35, Tolokers +4.76, T-Finance +2.14. Reddit (second largest)
  is omitted; T-Finance (smallest) is included.
- **Model name.** ChiGNN (Tables 2, 3) vs. ChiGAD (§5.1.2, Table 6) — global decision needed.
- **Dataset count.** §5.1.1 lists six, tables have seven; **Questions** is uncited
  (Platonov et al. [19], same as Tolokers).

---

# Track B — TKDE (ESO-GAD+)

## E5 — M1: filter-bank agnosticism

**Highest priority in the entire TKDE plan.**

**What.** Run $\bm{L}_\eta$ under **BernNet, JacobiConv, ChebNetII, GPR-GNN, and plain
2-layer GCN**, in addition to the Beta-wavelet bank. Report as a $\Delta$-over-own-backbone
table: each backbone's unsigned baseline next to its signed version.

**Claim to establish.** The operator carries the gain; the filter is interchangeable.

**Why it is load-bearing.** The current paper demonstrates one operator under one filter
bank with one calibrator. This experiment converts ESO-GAD from "a method" into "a module,"
which is what determines whether other groups build on it.

## E6 — Full calibrator suite

| Ablation | Question answered |
|---|---|
| Reliability diagrams + ECE | Does the Brier term actually calibrate? (currently asserted, never shown) |
| **Oracle $q^*$ upper bound** | How much headroom remains above the learned calibrator? |
| Cross-fitting: $K{=}1$ vs. $K{=}5$ | Does cross-fitting matter empirically? |
| **Node- vs. edge-disjoint folds** | Blocking prerequisite for T1 |
| Brier-loss ablation (BCE only) | Isolates the $\mathcal{L}_{MSE}$ term in Eq. (11) |
| Descriptor blocks: $\bm{z}^x$ / $\bm{z}^s$ / both | Validates Eq. (4)–(5); pairs with T5 |

The oracle bound is the most informative single row here — it tells reviewers (and you)
whether the remaining gap is a calibration problem or an operator problem.

## E7 — Synthetic sweep for the T3 breakdown point

**What.** Contextual SBM with known $q^*$. Sweep (i) heterophily rate and (ii) injected
calibration noise $\varepsilon$. Locate $\varepsilon^\dagger$ empirically and overlay the
**learned adaptive $\eta$** at each noise level.

**Target finding.** Learned $\eta$ decreases as $\varepsilon$ approaches $\varepsilon^\dagger$
— i.e. the model retreats from signing exactly when theory says signing stops helping.

**Extends.** §5.3.2 already perturbs $\hat q$ with $\Delta_\epsilon \sim U(-\epsilon,\epsilon)$
on real graphs. The synthetic version adds ground-truth $q^*$, which the real-data version
structurally cannot have.

## E8 — Benchmark coverage

- **Full GADBench**, both the 100-label semi-supervised protocol and the 40% fully
  supervised one.
- Add **DGraph-Fin** for scale.
- Report AUROC, AUPRC, Recall@K, plus a **Nemenyi critical-difference diagram** across all
  datasets.
- Own the Elliptic underperformance proactively in the abstract and results rather than
  leaving it for a reviewer to find.

## E9 — Robustness

| Perturbation | Setup |
|---|---|
| Structural attack | random and adversarial edge insertion, increasing budget |
| Feature noise | Gaussian / dropout on $\bm{X}$ |
| Label noise | flip a fraction of the 100 training labels |

The label-noise axis is the most pointed, since the calibrator's targets derive from those
same 100 labels — it tests the whole two-stage pipeline at its weakest joint.

## E10 — Efficiency

**What.** Time and peak memory vs. $|\mathcal{E}|$, with and without the proposed caching.

**The caching idea.** $\bm{W}^{(\eta)} = \tilde{\bm{A}} \odot (\bm{1}\bm{1}^\top -
2\eta\hat{\bm{Q}})$ is **affine in $\eta$**, so a large part of each
$\bm{L}_\eta^{j}\bm{H}^{(0)}$ propagation decomposes into $\eta$-independent pieces that can
be precomputed once and recombined per step. If this works, the Table 6 gap largely closes
and a current weakness becomes a contribution.

**Table layout (already designed).** Datasets as rows ordered by ascending edge count;
methods as columns. Recommended rows: Tolokers, YelpChi, DGraph-Fin, T-Finance. See
`todo_runtime.md` and `runtime_table.tex`.

**Fairness requirement.** Charge preprocessing to *all* baselines — GHRN's graph
purification especially, which is otherwise invisible in end-to-end timing.

## E11 — Interpretability: precision@flip

**What.** Of the edges that flip sign ($\hat q_{uv} > 1/(2\eta)$), what fraction are
anomaly-adjacent? Report per dataset against the base rate.

**Why.** Nearly free from runs you will already have, and it converts the signing mechanism
from a black box into something a reviewer can see working.

---

## Sequencing

**Immediately (days, no new runs):** E4 table hygiene → E1 → E3. Each is re-analysis and
each removes a reviewer objection outright.

**Before the deadline (one run per dataset):** E2, plus the matched-estimator ablation.

**TKDE, in dependency order:**

```
node-disjoint folds ──> E6 ──> T1
E7 ──> T3's breakdown point ──> adaptive-η defense
E5 (M1) ──> the operator-first claim
E8, E9, E10, E11 ──> supporting evidence
```

**If time compresses:** E5 and E6 are the pair that cannot be cut — E5 proves the operator
is the contribution, E6 proves the calibrator is sound. E9 and E11 go first.
