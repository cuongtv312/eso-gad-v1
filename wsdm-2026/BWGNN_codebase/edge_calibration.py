"""Cross-fitted edge-heterophily calibration for EC-SBWGNN.

Implements Section 2.1 and Algorithm 1 (steps 1-3) of the EC-SBWGNN note:
a symmetric edge descriptor z_uv, a small MLP q_psi(z_uv) approximating
Pr(y_u != y_v | z_uv), and K-fold cross-fitting over the labeled node set
so that no edge's calibrated probability is produced by a model that saw
either endpoint's label during training.

Design decision: q_psi is trained ONLY here and then frozen. Algorithm 1's
main loop (step 5) lists (theta, omega, eta, g) as the trainable parameters
and does not re-list psi, so L_edge / L_Brier in the paper's loss formula
are read as the objective used during this cross-fitted pretraining stage,
not as a term that keeps updating psi jointly with the node classifier.
This also removes any circular dependency between the edge predictor and
the node predictor by construction, without needing a stop-gradient hack.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:  # pragma: no cover
    StratifiedKFold = None
    roc_auc_score = None
    average_precision_score = None


def expected_calibration_error(probs, targets, n_bins=10):
    """Standard ECE for a binary probability estimate: bin predictions into
    `n_bins` equal-width bins over [0,1], and take the bin-size-weighted
    average gap between each bin's mean predicted probability (confidence)
    and its mean true target (empirical positive rate)."""
    probs = probs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = probs.shape[0]
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if not np.any(in_bin):
            continue
        conf = probs[in_bin].mean()
        acc = targets[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(conf - acc)
    return float(ece)


class EdgeHeterophilyPredictor(nn.Module):
    """q_psi: small MLP mapping a symmetric edge descriptor to a heterophily logit."""

    def __init__(self, feat_dim, struct_dim, hidden_dim=32):
        super().__init__()
        in_dim = 2 * feat_dim + 1 + struct_dim  # |xu-xv|, xu*xv, cos(xu,xv), r_uv
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim >= 2 else hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 2 else hidden_dim, 1),
        )

    def forward(self, z):
        return self.mlp(z).squeeze(-1)  # logits, shape [num_edges]


def compute_structural_stats(graph, max_edges_for_common_neighbors=5_000_000):
    """Label-free, symmetric structural edge statistics r_uv.

    r_uv = [log-degree sum, |log-degree difference|, Salton index, Jaccard index].
    The two neighborhood-overlap statistics require a sparse A @ A product; on
    very large graphs (e.g. T-Social) this can blow up in memory, so above
    `max_edges_for_common_neighbors` they are skipped and left at zero, keeping
    only the cheap degree-based statistics. This mirrors the note's own
    complexity discussion (Sec 2.3) rather than silently failing.
    """
    src, dst = graph.edges()
    n = graph.num_nodes()
    num_edges = graph.num_edges()

    deg = graph.in_degrees().float().clamp(min=1)
    log_deg = torch.log1p(deg)
    deg_sum = (log_deg[src] + log_deg[dst]).unsqueeze(-1)
    deg_diff = (log_deg[src] - log_deg[dst]).abs().unsqueeze(-1)

    if num_edges <= max_edges_for_common_neighbors:
        src_np, dst_np = src.numpy().copy(), dst.numpy().copy()
        A = sp.csr_matrix((np.ones(num_edges, dtype=np.float32), (src_np, dst_np)), shape=(n, n))
        A.data[:] = 1.0
        CN = A.dot(A).tocsr()
        cn_vals = np.asarray(CN[src_np, dst_np]).flatten()
        deg_np = deg.numpy()
        denom = np.sqrt(np.clip(deg_np[src_np] * deg_np[dst_np], 1e-12, None))
        salton = cn_vals / denom
        union = np.clip(deg_np[src_np] + deg_np[dst_np] - cn_vals, 1.0, None)
        jaccard = cn_vals / union
        salton_t = torch.tensor(salton, dtype=torch.float32).unsqueeze(-1)
        jaccard_t = torch.tensor(jaccard, dtype=torch.float32).unsqueeze(-1)
    else:
        salton_t = torch.zeros(num_edges, 1)
        jaccard_t = torch.zeros(num_edges, 1)

    return torch.cat([deg_sum, deg_diff, salton_t, jaccard_t], dim=-1)


def build_edge_descriptors(graph, feat, struct_stats, eps=1e-8):
    """z_uv = [|xu-xv|, xu (.) xv, cos(xu,xv), r_uv]. Symmetric under u<->v swap by construction."""
    src, dst = graph.edges()
    xu, xv = feat[src], feat[dst]
    diff = (xu - xv).abs()
    prod = xu * xv
    cos = F.cosine_similarity(xu, xv, dim=-1, eps=eps).unsqueeze(-1)
    return torch.cat([diff, prod, cos, struct_stats], dim=-1)


def symmetrize_edge_values(graph, values):
    """Average q_uv and q_vu (pseudocode step 3). z is already symmetric, so this
    is mostly a defensive no-op; skipped for multigraphs where a unique reverse
    edge id is ill-defined (e.g. homogenized multi-relational Yelp/Amazon graphs
    can carry parallel edges between the same pair of nodes)."""
    if graph.is_multigraph:
        return values
    src, dst = graph.edges()
    try:
        rev_eids = graph.edge_ids(dst, src)
    except Exception:
        return values
    return 0.5 * (values + values[rev_eids])


def stratified_fold_assignment(labels, train_mask, k, seed=2):
    """Assigns each train (labeled) node to one of k stratified folds; -1 elsewhere."""
    train_idx = train_mask.nonzero(as_tuple=True)[0].numpy()
    train_labels = labels[train_mask].numpy()
    fold_of = -np.ones(labels.shape[0], dtype=np.int64)

    if StratifiedKFold is None or k < 2 or len(np.unique(train_labels)) < 2:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(train_idx))
        for i, idx in enumerate(perm):
            fold_of[train_idx[idx]] = i % max(k, 1)
        return torch.tensor(fold_of, dtype=torch.long)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold_id, (_, val_idx) in enumerate(skf.split(train_idx, train_labels)):
        fold_of[train_idx[val_idx]] = fold_id
    return torch.tensor(fold_of, dtype=torch.long)


def _pos_weight(t):
    """(n_neg/n_pos), detached -- shared imbalance-weighting rule used for
    both the exact (E_LL) and pseudo (E_LU) BCE terms, each computed from
    its own edge set's own class balance."""
    n_pos = t.sum().clamp(min=1.0)
    n_neg = (t.numel() - t.sum()).clamp(min=1.0)
    return (n_neg / n_pos).detach()


def _train_fold_predictor(z, t, eligible, feat_dim, struct_dim, hidden_dim, epochs, lr, weight_decay,
                           z_lu=None, t_lu=None, eligible_lu=None, lambda_p=0.0):
    """Trains one fold's q_psi. With lambda_p=0 (default) this is exactly
    the original E_LL-only objective. With lambda_p>0 and a non-empty
    eligible_lu mask, adds the separately-normalized pseudo-supervision
    term from weak_edge_supervision.tex:

        L_edge = (1/|E_LL|) sum_{E_LL} ell(q,t) + lambda_p * (1/|E_LU|) sum_{E_LU} ell(q,t~)

    Each term's own `.mean()` reduction (via F.binary_cross_entropy_with_logits
    / F.mse_loss's default reduction) already implements the "normalize
    separately" requirement -- pseudo edges being far more numerous than
    exact edges cannot dominate simply by outnumbering them, since each sum
    is divided by its own edge count before lambda_p is applied.
    """
    model = EdgeHeterophilyPredictor(feat_dim, struct_dim, hidden_dim).to(z.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    z_ll, t_ll = z[eligible], t[eligible]
    pos_weight_ll = _pos_weight(t_ll)

    use_lu = lambda_p > 0 and eligible_lu is not None and eligible_lu.sum().item() > 0
    if use_lu:
        z_lu_e, t_lu_e = z_lu[eligible_lu], t_lu[eligible_lu]
        pos_weight_lu = _pos_weight(t_lu_e)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits_ll = model(z_ll)
        bce_ll = F.binary_cross_entropy_with_logits(logits_ll, t_ll, pos_weight=pos_weight_ll)
        brier_ll = F.mse_loss(torch.sigmoid(logits_ll), t_ll)
        loss = bce_ll + brier_ll

        if use_lu:
            logits_lu = model(z_lu_e)
            bce_lu = F.binary_cross_entropy_with_logits(logits_lu, t_lu_e, pos_weight=pos_weight_lu)
            brier_lu = F.mse_loss(torch.sigmoid(logits_lu), t_lu_e)
            loss = loss + lambda_p * (bce_lu + brier_lu)

        loss.backward()
        opt.step()
    model.eval()
    return model


def build_pseudo_lu_targets(labels, u_labeled, v_labeled, src, dst):
    """weak_edge_supervision.tex, Case 2/3: for an E_LU edge with exactly
    one labeled endpoint w (label y_w), the unlabeled endpoint is assumed
    normal (anomalies are rare), so the pseudo heterophily target reduces
    to t~_uv = y_w directly:
        - labeled normal (y_w=0) -- unlabeled  -> pseudo-homophilic (t~=0)
        - labeled anomaly (y_w=1) -- unlabeled -> pseudo-heterophilic (t~=1)
    Symmetric in which endpoint is the labeled one. Only meaningful where
    exactly one of u_labeled/v_labeled holds -- the caller masks by that.
    """
    labels_src = labels[src].float()
    labels_dst = labels[dst].float()
    return torch.where(u_labeled, labels_src, labels_dst)


def cross_fitted_edge_calibration(graph, feat, labels, train_mask, k=5, hidden_dim=32,
                                   epochs=150, lr=1e-2, weight_decay=1e-5, seed=2,
                                   max_edges_for_common_neighbors=5_000_000,
                                   lambda_p=0.0, oracle_lu=False, device=None):
    """Runs Algorithm 1 steps 1-3 and returns a frozen, per-edge calibrated
    heterophily probability tensor q (shape [num_edges], no grad), plus a
    diagnostics dict for reporting edge-calibration quality (Sec 5, G1/G2).

    lambda_p>0 additionally supervises q_psi on labeled-unlabeled edges
    E_LU with pseudo labels (weak_edge_supervision.tex). lambda_p=0 (the
    default) exactly reproduces the original E_LL-only calibrator.

    oracle_lu=True replaces the pseudo target on E_LU with the *true*
    hidden edge label (1[y_u != y_v], using labels that would not normally
    be visible for unlabeled nodes) -- an offline-analysis-only upper
    bound on how much the pseudo-supervision idea could help if the
    "unlabeled endpoint is normal" assumption were replaced by ground
    truth. Never use oracle_lu for a result reported as achievable at
    deployment time; it exists purely to bound Exp. "Oracle-E_LU" in the
    ablation.

    device: where the (potentially millions-of-edges) MLP training/
    inference runs. The structural-stats/fold-assignment preprocessing
    below is numpy/scipy/sklearn-based and always runs on CPU regardless
    of `device` (a one-time, comparatively cheap cost); only the tensors
    that feed the repeated per-epoch MLP forward/backward passes are moved
    to `device`. Defaults to `feat`'s own device if not given.
    """
    if device is None:
        device = feat.device
    graph_cpu = graph.to('cpu') if graph.device.type != 'cpu' else graph
    feat_cpu = feat.detach().cpu()
    labels_cpu = labels.detach().cpu()
    train_mask_cpu = train_mask.detach().cpu()

    struct = compute_structural_stats(graph_cpu, max_edges_for_common_neighbors)
    z = build_edge_descriptors(graph_cpu, feat_cpu, struct)
    src, dst = graph_cpu.edges()
    self_loop_mask = src == dst

    fold_of = stratified_fold_assignment(labels_cpu, train_mask_cpu, k, seed)
    fu, fv = fold_of[src], fold_of[dst]
    u_labeled = fu >= 0
    v_labeled = fv >= 0
    both_labeled = u_labeled & v_labeled          # E_LL
    exactly_one_labeled = u_labeled ^ v_labeled    # E_LU

    t = (labels_cpu[src] != labels_cpu[dst]).float()  # exact target, valid for any edge
    if oracle_lu:
        t_lu_target = t
    else:
        t_lu_target = build_pseudo_lu_targets(labels_cpu, u_labeled, v_labeled, src, dst)

    # fold of whichever endpoint is labeled, for E_LU edges (the other
    # endpoint has fold_of == -1 and never gates exclusion)
    labeled_fold = torch.where(u_labeled, fu, fv)

    # Move everything the MLP training/inference loop touches to `device`;
    # graph_cpu/fold_of/src/dst above stay CPU-only bookkeeping.
    z_dev = z.to(device)
    t_dev = t.to(device)
    t_lu_dev = t_lu_target.to(device)
    fu_dev, fv_dev = fu.to(device), fv.to(device)
    both_labeled_dev = both_labeled.to(device)
    exactly_one_labeled_dev = exactly_one_labeled.to(device)
    labeled_fold_dev = labeled_fold.to(device)

    models = [None] * k
    for j in range(k):
        eligible_ll = both_labeled_dev & (fu_dev != j) & (fv_dev != j)
        # Node-wise cross-fitting extended to E_LU (weak_edge_supervision.tex,
        # point 4): exclude every edge whose pseudo target depends on a
        # labeled node u in the held-out fold j, exactly as for E_LL.
        eligible_lu = exactly_one_labeled_dev & (labeled_fold_dev != j)

        if eligible_ll.sum().item() < 4 or t_dev[eligible_ll].sum().item() == 0 or (~t_dev[eligible_ll].bool()).sum().item() == 0:
            continue
        models[j] = _train_fold_predictor(
            z_dev, t_dev, eligible_ll, feat.shape[1], struct.shape[1], hidden_dim, epochs, lr, weight_decay,
            z_lu=z_dev, t_lu=t_lu_dev, eligible_lu=eligible_lu, lambda_p=lambda_p)

    with torch.no_grad():
        E = z_dev.shape[0]
        probs = torch.stack([
            torch.sigmoid(models[j](z_dev)) if models[j] is not None else torch.full((E,), 0.5, device=device)
            for j in range(k)
        ], dim=0)  # [K, E]

        # Unchanged from the original: "touch" is defined purely by fold
        # membership of fu/fv, so it already generalizes correctly to E_LU
        # edges (an unlabeled endpoint has fold -1 and never touches any
        # fold) without any code change here.
        fold_ids = torch.arange(k, device=device).unsqueeze(1)  # [K,1]
        touch = (fold_ids == fu_dev.unsqueeze(0)) | (fold_ids == fv_dev.unsqueeze(0))  # [K,E]
        no_touch = ~touch.any(dim=0)
        mask = touch.float()
        mask[:, no_touch] = 1.0  # neither endpoint labeled: ensemble average of all folds
        weights = mask / mask.sum(dim=0, keepdim=True).clamp(min=1.0)
        q = (weights * probs).sum(dim=0)

        q = torch.where(self_loop_mask.to(device), torch.zeros_like(q), q)  # a node is trivially homophilic with itself
        q_cpu = symmetrize_edge_values(graph_cpu, q.cpu())  # graph structural query (edge_ids) is CPU-only
        q = q_cpu.to(device).clamp(0.0, 1.0)

    diagnostics = evaluate_edge_calibration(q_cpu.clamp(0.0, 1.0), t, both_labeled)
    diagnostics['num_trained_folds'] = sum(m is not None for m in models)
    diagnostics['num_E_LL'] = int(both_labeled.sum().item())
    diagnostics['num_E_LU'] = int(exactly_one_labeled.sum().item())
    diagnostics['num_E_LL_hetero'] = int(t[both_labeled].sum().item())  # |E_LL^hetero|: true-heterophilic among E_LL
    diagnostics['lambda_p'] = lambda_p
    diagnostics['oracle_lu'] = oracle_lu

    # Pseudo-label accuracy of the "unlabeled endpoint assumed normal"
    # assumption itself (measured against true labels, mode-independent of
    # whether oracle_lu was actually used for training) -- both overall over
    # E_LU and restricted to the "anomaly-related" subset where the
    # unlabeled endpoint's TRUE label is 1 (anomalous), i.e. exactly where
    # the assumption is wrong and pseudo-supervision could mislead q_psi.
    pseudo_targets_for_eval = build_pseudo_lu_targets(labels_cpu, u_labeled, v_labeled, src, dst)
    unlabeled_true_label = torch.where(u_labeled, labels_cpu[dst], labels_cpu[src]).float()
    lu_mask = exactly_one_labeled
    if lu_mask.sum().item() > 0:
        correct = (pseudo_targets_for_eval == t).float()
        diagnostics['pseudo_lu_accuracy'] = float(correct[lu_mask].mean().item())
        anomaly_related_mask = lu_mask & (unlabeled_true_label == 1)
        diagnostics['num_E_LU_anomaly_related'] = int(anomaly_related_mask.sum().item())
        if anomaly_related_mask.sum().item() > 0:
            diagnostics['pseudo_lu_accuracy_anomaly_related'] = float(correct[anomaly_related_mask].mean().item())
    return q, diagnostics


def evaluate_edge_calibration(q, t, both_labeled_mask):
    """Reports out-of-fold edge-calibration quality on labeled-labeled edges:
    BCE, Brier score, and AUC where computable. Useful for Sec 5's G1/G2
    discussion (edge-estimator reliability) even though these are not used
    as a training loss once q is frozen."""
    diag = {}
    if both_labeled_mask.sum().item() == 0:
        return diag
    q_e = q[both_labeled_mask].clamp(1e-6, 1 - 1e-6)
    t_e = t[both_labeled_mask]
    diag['edge_bce'] = F.binary_cross_entropy(q_e, t_e).item()
    diag['edge_brier'] = F.mse_loss(q_e, t_e).item()
    diag['edge_ece'] = expected_calibration_error(q_e, t_e)
    if t_e.unique().numel() == 2:
        if roc_auc_score is not None:
            diag['edge_auc'] = float(roc_auc_score(t_e.numpy(), q_e.detach().numpy()))
        if average_precision_score is not None:
            diag['edge_ap'] = float(average_precision_score(t_e.numpy(), q_e.detach().numpy()))
    return diag
