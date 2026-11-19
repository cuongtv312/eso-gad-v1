# Theory contribution and the tests the paper needs

Nov 19, 2026 · companion to `theory.md`, `experiment.md` and `learn_eta_vs_fixed_eta.md` · written before the next analysis of $\eta$ · theorem numbers are the paper's; experiment ids are those of `experiment.md` and `learn_eta_vs_fixed_eta.md`

This file answers two questions: what Section 5 contributes as it stands, and which experiments are needed for the paper to reach a borderline accept or an accept at ICML. It is an assessment, not a result: no experiment has been run on this machine yet.

## What Section 5 contributes

Section 5 answers four questions. They are listed from the strongest result to the weakest.

| Question | Results | How strong |
| --- | --- | --- |
| Why does signing help? | Theorem 1 (with correct signs a heterophilic graph is spectrally a homophilic one), Theorem 2 (the label frequency is twice the frustration), Corollaries 1 and 2 | Exact on any graph, no assumption. Theorem 1 is close to classical balance theory, so what is new is the statement for the $\eta$-family and the label-frequency identity |
| Why sign, and not prune or reweight? | Theorem 3 (node factor: signing beats pruning by $(1-\tilde h_u)^{-1/2}$), Theorem 4 (soft signs are the optimal weights), Corollary 3 | Exact, but for one propagation step and a class-centred feature model with independent noise |
| What does an imperfect calibrator cost? | Theorem 8 (the loss of the squared node factor is at most four times the local Brier score, so it is second order; breakdown points; the best single $\eta$ is the least-squares slope), Corollary 6, Theorems 5 to 7 (degree-free stability) | Theorem 8 needs no assumption and its threshold cannot be improved. Theorems 5 to 7 are worst-case bounds, and Corollary 5 is vacuous on real graphs |
| What does weak edge supervision learn? | Proposition 2 and Corollary 4 (the target on singly-labeled edges estimates "touches an anomaly", not heterophily), Proposition 3 (which scores are free of a node's own label) | Exact. Proposition 2 exposes a flaw in the method's own target |

Taken together this is a coherent account, and more theory than most papers on graph anomaly detection carry. Two weaknesses are easy for a reviewer to find:

- **Every node-level result is for one propagation step** (Theorems 3(a), (b), (d), 4 and 8), while the model applies a filter of order $P$. Only Theorem 3(c) covers more than one step, and only for oracle signs.
- **No statement of Section 5 has a measured number next to it.** The table in `experiment.md`, "What the paper now commits us to", lists what each statement still owes.

## The main risk is the effect, not the theory

The theory explains why the signed operator should gain over its unsigned backbone. The runs of Nov 18 on the earlier machine did not show that gain, and they have not been reproduced here:

- With $\eta$ fixed on a grid, no value beat $\eta=0$ on Amazon or YelpChi beyond the spread over draws.
- The learned $\eta$ ended where it started, in every draw.
- A model that uses the features alone matched the graph model on Amazon and was ahead of the unsigned one on YelpChi.
- In GADBench's published table, tree models with neighbor averages are ahead of the WSDM row on T-Finance and Questions.

A paper whose operator does not beat its own $\eta=0$ control is rejected whatever the theorems say. So the first tests below are the ones that decide whether there is an operator-level claim, and the theory-specific tests come after them.

## Tests needed for a borderline accept

All six are required. Each line names the experiment, what it decides and which reviewer point it answers.

| # | Test | Ids | What it decides |
| --- | --- | --- | --- |
| 1 | Fixed-$\eta$ grid on the seven datasets and 10 splits, with the feature-only line drawn in, and paired Wilcoxon tests | D1, X1, X2 | Whether the operator helps at all, per dataset, and against which reference. Reviewer 2 asked for the $\eta=0$ control by name |
| 2 | Reproduce the WSDM rows of BWGNN and ESO-GAD with the flags of the WSDM runs (class-weighted calibrator loss, `lambda_p`, directed Weibo, Tolokers and Questions) | E0.6 | Whether the reported gain depends on those settings. If it does, the paper has to say which one |
| 3 | The theorems against data, from saved scores: label frequency against $\eta$ (Theorem 2); camouflage facts and $\chi_u$ by class under the four operators (Theorem 3); $1-\chi_u^2/\tilde d_u$ against $4b^\eta_u$ and the share of anomalies with $b^\eta_u<\frac14$ (Theorem 8) | C1, B7, C2, C21 | Whether the identities show up as the regularities the paper claims. Reviewer 3's "why separability" |
| 4 | Direct evaluation of the calibrator on edges with both endpoints unlabeled: edge AUROC, AUPRC, Brier score, calibration error, and the breakdown by edge type | B1, B2, B3 | Whether the scores generalize beyond the labeled nodes. Asked by Reviewers 1, 2 and 3 |
| 5 | Sign against reweight against prune with the same scores, plus oracle signing, oracle pruning and shuffled scores | D3, D9 | Theorem 4 on real data, the headroom of Theorem 1, and the control that the gain needs the scores on the right edges |
| 6 | RF-Graph and XGB-Graph in the main table | A2 | Named by Reviewers 2 and 4. With them the headline is no longer "best on six of seven" |

Tests 3 and 4 need Phase 1 only and take minutes. Tests 1, 2, 5 and 6 are about 12 to 15 sweeps. At 0.25 GPU-hours per sweep that is about 4 GPU-hours; the figure was measured on the earlier machine and has to be re-measured here.

## Tests that would move the paper to an accept

For an accept the theory has to do work, not only explain. These four show that it does.

| # | Test | Ids | What it would show |
| --- | --- | --- | --- |
| 7 | $\eta_{\mathrm{LS}}$ estimated in Phase 1 from cross-fitted doubly-labeled edges, against the all-label value and against the best fixed $\eta$ of test 1 | X4, C21 (Corollary 6) | That the theory sets the one operator parameter without tuning. This replaces "$\eta$ is learned", which the Nov 18 runs contradict |
| 8 | The soft-imputed edge target against the paper's target: node-wise lift, $\chi_u$ and AUPRC | D4, X5 | That Proposition 2 finds the flaw of the target and the fix removes it. The paper then reads "the theory diagnoses and repairs the method" |
| 9 | Signed aggregation inside XGB-Graph: unsigned, pruned and signed neighbor means of the same scores | X7, D8 | Theorem 3 is a statement about exactly this aggregation, so this tests the theorem on the strongest baseline's own ground and shows the operator is not tied to BWGNN |
| 10 | Audits of Assumptions 1 and 2 on the encoder output and the scores, and a small synthetic phase diagram for the breakdown point of Corollary 6 | C9, C10, G1 | That the assumptions are visible and checked, and that the breakdown point is where the theory puts it |

These add roughly as many sweeps again as tests 1 to 6.

## One theory addition worth more than the others

Extend Theorem 8 from one step to the low-pass filter. Reviewers will ask why a one-step result explains a model of order $P$, and Theorem 3(c) already has the oracle half of the statement. A row-wise perturbation bound for $g(I-L_\eta(\hat q))$ against $g(I-L_1(t^\star))$ would give the calibrated half. This is candidate N2 of `theory.md` seen from the node side.

## What to drop or soften if the tests fail

| If | Then |
| --- | --- |
| Test 1 shows no gain over $\eta=0$ on any dataset and test 2 does not reproduce the WSDM rows | No operator-level claim until the difference to the WSDM runs is found |
| Test 1 is flat with the paper's target and test 8 shows a gain with the soft target | Lead with Proposition 2 and the lift: the method changes, and the theory is the reason |
| The signed model beats $\eta=0$ but not the feature-only line | Claim that signed propagation is safe where unsigned propagation hurts, and show the label budgets at which it gains (X6) |
| Test 9 improves XGB-Graph | Lead with it: the contribution is the operator |

The sentences of the paper that already wait for these outcomes are listed in `experiment.md`: "best on six of seven" (Section 6.2), "the KL loss stabilizes training" (Section 4.2), and the two sentences that say $\eta$ estimates the reliability slope.
