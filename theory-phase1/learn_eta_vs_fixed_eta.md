# Learned η against fixed η, and ESO-GAD against feature-only models

Nov 18, 2026 · companion to `theory.md` and `experiment.md` · every number here is from the Nov 18 workspace on the earlier machine; the run folders are listed at the end. **The workspace moved on Nov 19:** `src/` is now `wsdm-2026/BWGNN_codebase/`. The scripts were rebuilt there (`main_gadbench.py`, `dp0_eta_check.py`, `flat_eta_check.py`; `experiment.md`, "Rebuilt on Nov 19"), the old `runs/` folders are not on the new machine, and nothing here has been re-run yet. Experiments now run on the local GPU; Vast.ai only for runs that need more than 24 GB

## The question

ESO-GAD weights each edge by $W_{uv}=1-2\eta\,\hat q_{uv}$, where $\hat q_{uv}$ is the frozen heterophily score of Phase 1 and $\eta$ is learned with the node loss. Two things came out of the first checks:

1. **The learned $\eta$ does not move.** It ends where it starts, so "learned" currently means "set to its start value, 0.5".
2. **Performance is flat in $\eta$.** On Amazon and YelpChi with 100 labels no fixed $\eta$ beats $\eta=0$ (the unsigned backbone) significantly, and a model that uses the features alone does as well or better.

This file explains both from the theory, lists what can be changed, and lists the experiments needed to show that ESO-GAD adds something over models that do not learn on the graph.

**Reference models.** "Feature-only" means MLP, random forest and XGBoost on the node features. RF-Graph and XGB-Graph are different: they are trees on features that were averaged over neighbors beforehand, so they use the graph but learn nothing on it. Both kinds are reference levels below.

## Summary

| Finding | Cause | Fix | Experiment that shows it |
| --- | --- | --- | --- |
| Learned $\eta$ stays at its start | The node loss does not identify $\eta$ (Section 2.1) | Compute $\eta$ in Phase 1 and fix it (F1) | X4, X9 |
| No gain near $\eta=0$ | Anomalies have non-positive node-wise lift under the paper's edge target (Section 2.2) | Soft-imputed edge target (F2) | X3, X5 |
| No gain at large $\eta$ | Heterophilic edges get weights near zero: the operator prunes, it does not sign (Section 2.3) | Recalibrated soft-sign weights (F3) | X5, X8 |
| Graph model at or below the feature-only model | At 100 labels the graph adds little on Amazon; unsigned aggregation hurts on YelpChi (Section 2.4) | Explicit feature channel (F4); signed aggregates inside XGB-Graph (F5) | X1, X2, X6, X7 |

Recommendation: **fix $\eta$ from Phase 1 and stop presenting it as learned**, adopt the soft target if X5 confirms it, and make X7 (signed aggregation inside XGB-Graph) the main argument against the tree baselines. All of this is open until the experiments run; the evidence so far is from two datasets and 3 to 10 random draws.

## 1. What was measured

### 1.1 The learned η follows its start

Final $\eta$ from three starting values, same seed. 40% labels: one run each. 100 labels: mean over 10 random draws.

| Setting | Amazon, start 0.2 / 0.5 / 0.8 | YelpChi, start 0.2 / 0.5 / 0.8 |
| --- | --- | --- |
| 40% labels, 100 epochs | 0.21 / 0.52 / 0.82 | 0.36 / 0.60 / 0.83 |
| 40% labels, 500 epochs | 0.20 / 0.54 / 0.84 | 0.60 / 0.61 / 0.78 |
| 100 labels, 500 epochs | 0.21 / 0.52 / 0.79 | 0.20 / 0.51 / 0.82 |

With 100 labels the three starts never end within 0.05 of each other, in 0 of 10 draws on either dataset. The training loss at epoch 100 is 0.006 on Amazon and 0.0001 on YelpChi. Only YelpChi at 40% labels shows $\eta$ being pulled to a value (0.60, next to a reliability slope of 0.62).

### 1.2 Performance is flat in η

Test AUPRC with $\eta$ fixed, 100 labels, mean over 10 random draws (standard deviation over draws 0.02 to 0.06):

| $\eta$ | 0 | 0.125 | 0.25 | 0.375 | 0.5 | 0.625 | 0.75 | 0.875 | 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Amazon | 0.806 | 0.806 | 0.787 | 0.812 | 0.809 | 0.806 | 0.817 | 0.818 | 0.798 |
| YelpChi | 0.238 | 0.240 | 0.240 | 0.239 | 0.239 | 0.232 | 0.232 | 0.226 | 0.228 |

The best value beats $\eta=0$ by 0.012 on Amazon (paired Wilcoxon $p=0.43$) and by 0.002 on YelpChi. The $\eta=0$ rows equal GADBench's BWGNN numbers (81.7 and 23.7), so the pipeline is sound. The WSDM table reports 83.7 and 28.9 for ESO-GAD.

### 1.3 What the operator does to each class

The node factor $\chi_u$ of Theorem 3(a) is positive when one propagation step puts node $u$ on the side of its own class, and its size is the signal-to-noise gain. Mean over anomalies, 3 draws, 100 labels:

| | Amazon | YelpChi |
| --- | --- | --- |
| $\chi_u$ at $\eta$ = 0 / 0.25 / 0.5 / 0.75 / 1, calibrated scores | −9.9 / −9.9 / −7.1 / +0.9 / +2.0 | −7.4 / −7.5 / −7.4 / −6.0 / −4.3 |
| $\chi_u$ with oracle signs / oracle pruning | +12.8 / +4.0 | +12.0 / +5.0 |
| Mean weight at $\eta=1$ on heterophilic / homophilic edges | −0.11 / +0.70 | +0.39 / +0.65 |
| Test AUPRC: $\eta=0$ / oracle signs / oracle pruning | 0.823 / 0.990 / 0.862 | 0.236 / 0.990 / 0.464 |

With the true signs the same model reaches 0.99 on both graphs, so the signal is in the graph. (The oracle operator also reveals the class through the neighborhood's net sign, so 0.99 is a ceiling, not a fair score.)

### 1.4 Reference levels

| | Amazon | YelpChi | Source |
| --- | --- | --- | --- |
| Feature-only network (cross-fitted MLP on the features, 100 labels) | 0.819 | 0.285 | this workspace, 3 draws |
| Unsigned graph model ($\eta=0$) | 0.822 | 0.236 | same 3 draws |
| Best signed variant so far (soft target, $\eta=0.75$) | 0.821 | 0.296 | same 3 draws |

AUPRC from GADBench's semi-supervised table against the WSDM row of ESO-GAD. The GADBench numbers come from an automated extraction of its paper and must be checked against the PDF before they are quoted.

| | Amazon | YelpChi | T-Finance | Tolokers | Reddit | Weibo | Questions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RF (features only) | 81.9 | 27.4 | 74.9 | 33.1 | 4.3 | 61.6 | 11.3 |
| RF-Graph | 70.7 | 23.6 | 81.1 | 35.8 | 4.5 | 73.8 | 10.1 |
| XGB-Graph | 84.4 | 24.8 | 78.3 | 34.1 | 4.1 | 75.9 | 7.7 |
| BWGNN | 81.7 | 23.7 | 60.9 | 35.3 | 4.2 | 80.6 | 6.5 |
| ESO-GAD (WSDM) | 83.7 | 28.9 | 73.6 | 40.4 | 5.5 | 86.2 | 9.8 |

On Amazon and YelpChi the features alone already match the graph models. The graph clearly helps on Weibo, and trees with neighbor averages lead on T-Finance. Amazon and YelpChi are therefore the weakest places to look for a gain of the operator.

## 2. Underlying theory

Notation is the paper's. $\tilde d_u$ is the degree with the self-loop, $\tilde h_u$ the share of heterophilic edges of $u$, and $\bar q^{\mathrm{het}}_u$, $\bar q^{\mathrm{hom}}_u$ the mean scores on the heterophilic and homophilic edges of $u$ (the self-loop is homophilic with score 0).

### 2.1 Why the node loss cannot identify η

The Beta channels are $\mathcal W_k(L)=\kappa_k (L/2)^k(I-L/2)^{P-k}$ with $\kappa_k=\frac{P+1}{2}\binom Pk$. By the binomial theorem,

$$\sum_{k=0}^{P}\mathcal W_k(L_\eta)=\frac{P+1}{2}\,I\qquad\text{for every }\eta\text{ and every score vector}.$$

The classifier sees the concatenation of the channel outputs, so one linear layer can add them up and recover the encoder output $H^{(0)}$. Hence, for every $\eta$, the model contains the feature-only network (encoder, then head, no graph). Two consequences:

- With 100 labels that sub-network can fit the training nodes, so the training loss goes to zero whatever the operator is. At that point the gradient of the loss in $\eta$ is zero, and $\eta$ stops. This is what Section 1.1 shows.
- $\eta$ is a parameter of the operator, but the only signal it receives is a loss that the operator is not needed to minimise. It is identified only when the labels are too many to fit without the graph (YelpChi at 40% labels).

This is candidate result N9 of `theory.md`. It is a remark with a three-line proof and it changes how $\eta$ should be presented.

### 2.2 Why nothing happens near η = 0

While no weight is negative (that is, for $\eta\le 1/(2\max_v\hat q_{uv})$), the margin of node $u$ is

$$m_u(\eta)=\frac{(\tilde d_u-2H_u)-2\eta\,(S^{\mathrm{hom}}_u-S^{\mathrm{het}}_u)}{\tilde d_u-2\eta\,(S^{\mathrm{hom}}_u+S^{\mathrm{het}}_u)},$$

with $H_u$ the number of heterophilic edges and $S^{\mathrm{het}}_u$, $S^{\mathrm{hom}}_u$ the score sums. Its derivative is

$$m_u'(\eta)=\frac{4\,H_u(\tilde d_u-H_u)\,\bigl(\bar q^{\mathrm{het}}_u-\bar q^{\mathrm{hom}}_u\bigr)}{\bigl(\tilde d_u-2\eta\,(S^{\mathrm{hom}}_u+S^{\mathrm{het}}_u)\bigr)^2}.$$

The terms in $\eta$ cancel in the numerator, so the sign is the same on the whole range. Call $\bar q^{\mathrm{het}}_u-\bar q^{\mathrm{hom}}_u$ the **node-wise lift**. Then:

> **Node-wise lift (candidate N8).** In the reweighting regime the margin of a node increases with $\eta$ if and only if its node-wise lift is positive, and it changes at a rate proportional to $\tilde h_u(1-\tilde h_u)$.

At $\eta=0$ this is Theorem 3(d). It explains the flat region in three steps:

- **The degree normalization cancels a uniform change.** Only the contrast between a node's own edges matters, not the level of its scores.
- **Camouflage makes neighborhoods nearly pure.** Anomalies have $\tilde h_u\approx0.8$ to $0.85$ and normal nodes $\tilde h_u\approx0.04$ to $0.13$, so the factor $\tilde h_u(1-\tilde h_u)$ is small for both (0.11 and 0.03 on Amazon).
- **The paper's edge target gives anomalies the wrong contrast.** By Proposition 2 the target on singly-labeled edges estimates whether an edge touches an anomaly, so an anomaly's edges to other anomalies score higher than its edges to normal nodes. Measured lift of anomalies: −0.18 on Amazon, −0.07 on YelpChi; only 9% and 18% of anomalies have positive lift. Normal nodes have positive lift (+0.23, +0.09), but they are already far on the correct side.

So the paper's own theory predicts no gain for the nodes that need it.

### 2.3 Why large η prunes instead of signing

An edge turns negative only when $\hat q_{uv}>1/(2\eta)$, and its weight is $1-2\eta\hat q_{uv}$. With the scores as they are, the heterophilic edges average 0.56 on Amazon and 0.30 on YelpChi, so at $\eta=1$ their mean weight is −0.11 and +0.39: close to zero, or still positive. Theorem 3(b) says what that is worth: pruning reaches $\sqrt{(1-\tilde h_u)\tilde d_u}$, which for a camouflaged anomaly is little more than its own features, against $\sqrt{\tilde d_u}$ for signing. And the unsigned bank already has a channel of that quality: the high-pass channel gives every node a factor of about $2\tilde h_u\le2$ (Section 5.2 of the paper). On Amazon the calibrated operator moves anomalies from −9.9 to about +2, the same level.

Theorem 4 says the same thing in general. The part of the deflection that only signed weights can reach is

$$(\mathcal J^\star)^2-(\mathcal J^\star_+)^2=\lVert\mu\rVert^2\sum_{v:\,p(\hat q_{uv})>1/2}\frac{c_v^2}{\varsigma_v^2},$$

a sum over the edges whose calibrated probability of heterophily exceeds one half. When the calibrator is unsure on every edge that sum is empty, and a flat response in $\eta$ is the optimal behaviour, not a defect of training.

### 2.4 Why the feature-only model is hard to beat here

- **Amazon.** The feature-only network equals the graph model (0.819 against 0.822), and in GADBench a random forest on the features equals BWGNN. At 100 labels the graph adds nothing there for any operator.
- **YelpChi.** The unsigned graph model is below the feature-only network (0.236 against 0.285). Theorem 3(b) gives the reason: one unsigned low-pass step puts 94% of the anomalies on the wrong side, and with 100 labels the classifier does not learn to ignore that channel, although Section 2.1 shows it could. Signing repairs the damage (0.296 with the soft target) without clearly passing the feature-only level.
- **Circularity.** Under camouflage an anomaly's edges are almost all heterophilic and a normal node's almost all homophilic, so knowing the signs of a node's edges is nearly the same as knowing its class. The calibrator has to infer that from the same 100 labels the node classifier has. It can add information only where the *pair* tells more than the two nodes separately (feature dissimilarity to neighbors, overlap), or where more labels make the edge problem easier than the node problem.

## 3. Potential fixes

| ID | Change | Addresses | Evidence so far | Cost |
| --- | --- | --- | --- | --- |
| F1 | **Compute $\eta$ in Phase 1 and fix it.** Either the minimiser of the cross-fitted label frequency on labeled edges (Theorem 2), or the reliability slope (Corollary 3). Since Nov 19 the paper has the statement behind the second choice for any calibrator: the least-squares slope $\sum\hat qt^\star/\sum\hat q^2$ minimises the total local Brier score that bounds the loss of node separability (Theorem 8, Corollary 6) | Section 2.1 | The minimiser is stable across draws: 0.63 (Amazon, sd 0.03), 0.67 (YelpChi, sd 0.05), computed there with all labels; the labeled-edge estimate is not tested yet. It needs labeled–labeled edges, and split 0 has only 4 on Questions, 6 on Reddit and 50 on YelpChi, so a fallback value is needed there | none at run time |
| F2 | **Soft-imputed edge target.** On a singly-labeled edge, impute the unlabeled endpoint with a cross-fitted node probability $\hat\pi_v$: $t_{uv}=y_u(1-\hat\pi_v)+(1-y_u)\hat\pi_v$. The paper's rule is $\hat\pi_v\equiv0$ | Section 2.2 | Lift of anomalies on Amazon −0.18 → +0.26 (73% positive), $\chi_u$ at $\eta=1$ from −6.2 to +0.7. No change on YelpChi (−0.07 → −0.06), where $\hat\pi$ is weak. YelpChi AUPRC 0.296 against 0.261 with the paper's target, 3 draws | one small cross-fitted node model |
| F3 | **Recalibrated soft-sign weights.** Replace $1-2\eta\hat q$ by $1-2\hat p(\hat q)$ with $\hat p$ a monotone map fitted on held-out labeled edges (Theorem 4). Sure edges get weights near ±1, unsure edges near 0 | Section 2.3 | none yet. It cannot repair a wrong ranking inside a neighborhood, so it comes after F2 | small |
| F4 | **Explicit feature channel.** Give the encoder output its own gated channel next to the Beta channels | Section 2.4 (YelpChi) | none yet. The channel is already in the span of the bank, so this changes optimisation, not capacity; it should make the model start from the feature-only solution | small |
| F5 | **Signed aggregation inside XGB-Graph.** Replace its neighbor mean by the signed mean with weights $1-2\eta\hat q$ | Section 2.4, the tree baselines | none yet. Theorem 3 is a statement about one aggregation step on features, which is exactly what XGB-Graph computes, so this is the most direct test of the theorem and does not depend on training a network | CPU only |
| F6 | Wider range for $\eta$ or a two-parameter map | Section 2.3 | not promising alone: on YelpChi the edges between anomalies would flip before the heterophilic ones | none |

What F1 does and does not do: it makes $\eta$ a well-defined, start-independent number, which removes the DP0 problem. It does not create a gain; that needs scores with positive lift (F2) and a dataset where the graph has something to add.

**Learned or fixed?** Fixed. The learned value is its start value, so the paper should not say that training finds the reliability slope. The choices are (a) F1, or (b) a single default for all datasets with the fixed-$\eta$ curve shown. The learned variant can stay as an ablation with its start value stated.

## 4. Experiments

All under the GADBench protocol with the official splits (`src/main_gadbench.py`), seven datasets, unless stated. One sweep is one configuration on seven datasets and 10 splits, about 0.25 GPU-hours locally and about 0.15 on a rented 3090. Following the running rule, anything over 5 minutes gets one local smoke run and then goes into a Vast.ai batch.

| ID | Experiment | What it shows | Needs | Cost |
| --- | --- | --- | --- | --- |
| X1 | **Reference levels in the same code and splits**: MLP with the same encoder and head, XGBoost, XGB-Graph, RF-Graph | The lines every other result is read against. GADBench's published numbers are a first stand-in | the MLP is `--variant none` in `main_gadbench.py` (built Nov 19); `xgboost` 2.0.3 is in `gcnenv`; the tree models are in `wsdm-2026/GADBench-master/` | CPU, minutes |
| X2 | **Fixed-$\eta$ grid**, $\eta$ from 0 to 1 in steps of 0.125 (D1 of `experiment.md`), with the MLP line of X1 drawn in | Whether the operator helps at all, per dataset, and against which reference | a fixed-$\eta$ option in the driver | 9 sweeps |
| X3 | **Node-wise lift by class**, per dataset and target, from the saved scores | Predicts, before training, where X2 can show a gain. The figure is the gain over $\eta=0$ against the mean lift of anomalies, one point per dataset | saved scores | minutes |
| X4 | **Computed $\eta$ against the grid**: the $\eta$ of F1 against the best fixed $\eta$ of X2 | Whether F1 picks a good value without tuning | X2, saved scores | none extra |
| X5 | **Edge target**: paper, soft (F2), and node-posterior product, each at $\eta$ = 0.5, 0.75, 1 and at the computed $\eta$; lift and $\chi_u$ reported next to AUPRC | Whether F2 moves the lift and the result beyond Amazon and YelpChi | soft target in `edge_calibration.py` | 8 sweeps |
| X6 | **Label budget**: 20, 50, 100, 200, 500, 1000 labels, for MLP, $\eta=0$ and the signed model (E1 of `experiment.md`) | Where the graph and the signs start to pay. At 40% labels $\eta$ did matter on YelpChi, so 100 labels should be one end of a curve | label subsampling in the driver | about 18 sweeps on 4 datasets, about 3 GPU-hours |
| X7 | **Signed XGB-Graph** (F5): XGB-Graph with unsigned, pruned and signed neighbor aggregates of the same scores | The operator's value independent of the network, and a direct comparison with the strongest baseline on its own ground | X1 | CPU |
| X8 | **Headroom**: oracle signs and oracle pruning at $\eta=1$, and oracle scores with a share $\epsilon$ flipped | How far the calibrator is from what the graph allows, and how good the scores must be for signing to beat pruning | oracle-score option | 4 sweeps |
| X9 | **Learned $\eta$ from starts 0.2, 0.5, 0.8** | Documents Section 1.1 on all seven datasets, for the ablation | none | 3 sweeps |

Order: X1, X3 and X9 first (cheap, and X3 says where to look), then X2 with X4, then X5 and X7, then X6 and X8. X1 and X7 are part of the main comparison that was moved to the second project phase; they are listed here because this question cannot be answered without them. Everything together is about 45 sweeps, roughly 11 GPU-hours locally or 7 on a 3090, plus the CPU runs.

### How to read the outcomes

| Outcome | What it means | What the paper can claim |
| --- | --- | --- |
| X2 shows a gain over $\eta=0$ and over the MLP line on several datasets | The operator helps as the paper says | The current claim, with the fixed-$\eta$ curve as evidence and $\eta$ computed (F1) |
| X2 flat with the paper's target, gain appears with F2 (X5), and X3 predicts on which datasets | The theory found the flaw in the target and the fix | A stronger paper: the theory diagnoses and repairs the method. Proposition 2 and the lift result move to the centre |
| Signed model beats $\eta=0$ but not the feature-only line | Signing protects against harmful aggregation and adds nothing beyond the features at 100 labels | "Signed propagation is safe where unsigned propagation hurts", plus the regime of X6 where it gains |
| X7 improves XGB-Graph | The operator is backbone-agnostic and helps the strongest baseline | Lead with it: the contribution is the operator, verified by Theorem 3 on the baseline's own aggregation |
| Nothing moves on any dataset, with either target | The gain of the WSDM table does not reproduce in this code | Find the difference to the WSDM code first (hyperparameter selection, graph symmetrization, class-weighted loss, overlap statistics). No claim until then |

## 5. Decisions for the author

1. **Present $\eta$ as computed or as a default, not as learned.** F1 or one fixed value; either way the sentences "$\eta$ estimates the reliability slope" (after Proposition 3) and "the learned values fall near the favorable region" (Section 6) have to change.
2. **Adopt the soft target (F2) if X5 confirms it.** It changes Section 4.1 and makes Proposition 2 a design argument.
3. **Bring X1 and X7 forward** from the second project phase. Without the feature-only line the fixed-$\eta$ results cannot be interpreted.
4. **Which code produced the WSDM table.** The $\eta=0$ rows here match GADBench's BWGNN, and the signed rows do not match the WSDM gain on Amazon and YelpChi.

## 6. Limits of the evidence

- Two datasets, random draws of the labeled set in place of the official splits, 100 epochs in place of GADBench's 200 with patience.
- The 10-draw runs used the class-weighted calibrator loss; the 3-draw diagnostics used the unweighted loss of the paper. Neither had the neighborhood-overlap statistics on YelpChi.
- The 3-draw numbers (Sections 1.3, 1.4 and the evidence column of Section 3) have a spread of 0.02 to 0.06 in AUPRC between draws. They are previews.
- The feature-only network of Section 1.4 is a small cross-fitted MLP, not the MLP baseline of GADBench.

## 7. Where the numbers come from

| Folder | Content |
| --- | --- |
| `runs/plan_check/` | 18 runs at 40% labels with three starts of $\eta$ and two learning rates |
| `runs/dp0/` | 100 labels, 10 draws per dataset: three starts for 500 epochs, nine fixed values (script `src/dp0_eta_check.py`) |
| `runs/flat_eta/amazon.json`, `yelp.json` | $\chi_u$ by class, Theorem 3(d) term, weights at $\eta=1$, oracle variants (script in the same folder) |
| `runs/flat_eta/lift_*.json` | node-wise lift and AUPRC for the paper, soft and product targets, and the feature-only network |
| `runs/gadbench/` | one-split smoke runs of the seven datasets under the GADBench protocol |
| `experiment.md` | "Verification of the track", "DP0 result", "Why the loss is flat in η", "What would remove the flat region" |
| `theory.md` | candidate results N8 (node-wise lift), N9 ($\eta$ not identified), N10 (soft imputation) |
