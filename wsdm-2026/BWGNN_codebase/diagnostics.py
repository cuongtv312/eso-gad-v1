"""Score-level quantities of the theory (Section 5 of the paper), computed on a
real graph from the edge scores and the ground-truth labels.

Everything here uses ALL labels and is for analysis only: nothing in this
file may feed training or model selection.

Conventions (the paper's):
    - edges are ordered pairs with A~_uv = 1, self-loops included;
    - t*_uv = 1[y_u != y_v] is the true heterophily indicator, s*_uv = 1 - 2 t*_uv;
    - W_uv = 1 - 2 eta q_uv, with q_uu = 0 so a self-loop always has weight 1;
    - sums "over the edges of u" run over the edges with source u, which for a
      symmetric graph is the row of u.
`valid` is an edge mask: edges outside it are ignored everywhere, which is
how nodes with unknown labels (Amazon's first 3,305) are left out -- the
quantities are then those of the subgraph induced on the labeled nodes.
"""
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def true_heterophily(src, dst, labels):
    return (labels[src] != labels[dst]).float()


def valid_edges(src, dst, known):
    """Edges with both endpoints of known label."""
    return known[src] & known[dst]


def edge_weight(q, eta, variant='sign', tau=None, self_loop=None):
    if variant == 'sign':
        w = 1.0 - 2.0 * eta * q
    elif variant == 'reweight':
        w = 1.0 - q
    elif variant == 'prune':
        w = (q < tau).to(q.dtype)
    else:
        raise ValueError(variant)
    if self_loop is not None:
        w = torch.where(self_loop, torch.ones_like(w), w)
    return w


def _node_sum(values, src, n, valid=None):
    if valid is not None:
        values = values * valid.to(values.dtype)
    return torch.zeros(n, dtype=values.dtype, device=values.device).index_add_(0, src, values)


# --------------------------------------------------------------------------
# Theorem 2, Corollary 1: frustration and label frequency
# --------------------------------------------------------------------------

def frustration(w, t, valid=None):
    """Fr = sum |w| 1[sign(w) != s*] / sum |w| over the valid ordered edges.
    The label frequency of Theorem 2 is 2 * Fr."""
    absw = w.abs()
    wrong = ((w < 0) != (t > 0.5)) & (w != 0)
    if valid is not None:
        absw = absw * valid.to(absw.dtype)
    return float((absw * wrong.to(absw.dtype)).sum() / absw.sum())


def label_frequency_curve(q, t, etas, valid=None, self_loop=None):
    """rho_eta = 2 Fr_eta for each eta of the signed family."""
    return [2.0 * frustration(edge_weight(q, float(e), self_loop=self_loop), t, valid) for e in etas]


def label_frequency_minimiser(q, t, valid=None, self_loop=None, etas=None):
    """(eta, rho) at the lowest label frequency on a grid of eta."""
    if etas is None:
        etas = np.round(np.arange(0.0, 2.0001, 0.01), 4)
    rho = label_frequency_curve(q, t, etas, valid, self_loop)
    i = int(np.argmin(rho))
    return float(etas[i]), float(rho[i])


# --------------------------------------------------------------------------
# Assumption 2, Corollary 3: reliability of the scores
# --------------------------------------------------------------------------

def reliability_slope(q, t, mask=None):
    """Least-squares slope through the origin of the true indicator on the
    score, eta_0 = sum(q t) / sum(q^2): the eta_0 of p(q) = eta_0 q."""
    if mask is not None:
        q, t = q[mask], t[mask]
    return float((q * t).sum() / (q * q).sum().clamp(min=1e-12))


def reliability_curve(q, t, mask=None, bins=10):
    """Per equal-width score bin: (mean score, share of heterophilic edges, count)."""
    if mask is not None:
        q, t = q[mask], t[mask]
    idx = (q * bins).long().clamp(max=bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt:
            out.append((float(q[m].mean()), float(t[m].mean()), cnt))
    return out


def edge_quality(q, t, mask, src=None, dst=None, labels=None):
    """Edge AUROC / AUPRC / Brier / calibration error of the scores against
    true heterophily on the masked edges, and the mean score by edge type."""
    qm, tm = q[mask].cpu().numpy(), t[mask].cpu().numpy()
    out = {'num_edges': int(mask.sum()), 'heterophily': float(tm.mean()) if len(tm) else float('nan'),
           'mean_score': float(qm.mean()) if len(qm) else float('nan')}
    if len(tm) and 0 < tm.sum() < len(tm):
        out['auroc'] = float(roc_auc_score(tm, qm))
        out['auprc'] = float(average_precision_score(tm, qm))
    out['brier'] = float(((qm - tm) ** 2).mean()) if len(qm) else float('nan')
    edges = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        b = (qm >= edges[i]) & (qm <= edges[i + 1]) if i == 0 else (qm > edges[i]) & (qm <= edges[i + 1])
        if b.any():
            ece += b.mean() * abs(qm[b].mean() - tm[b].mean())
    out['ece'] = float(ece)
    out['slope'] = reliability_slope(q, t, mask)
    if labels is not None:
        a_s, a_d = labels[src] == 1, labels[dst] == 1
        for name, m in (('nn', ~a_s & ~a_d), ('an', a_s ^ a_d), ('aa', a_s & a_d)):
            m = m & mask
            out['mean_score_' + name] = float(q[m].mean()) if m.any() else float('nan')
            out['num_' + name] = int(m.sum())
    return out


# --------------------------------------------------------------------------
# Theorem 3: node-level quantities of one propagation step G = D^-1 W
# --------------------------------------------------------------------------

def local_heterophily(src, t, n, valid=None):
    """(h~_u, d~_u): share of heterophilic edges of u and its degree, self-loop included."""
    ones = torch.ones_like(t)
    deg = _node_sum(ones, src, n, valid)
    return _node_sum(t, src, n, valid) / deg.clamp(min=1.0), deg


def node_factors(src, w, t, n, valid=None):
    """Margin m_u = sum_v w s* / sum_v |w|, effective degree (sum|w|)^2 / sum w^2
    and node factor chi_u = m_u sqrt(d_eff,u) of Theorem 3(a) for G = D^-1 W."""
    s_star = 1.0 - 2.0 * t
    abs_sum = _node_sum(w.abs(), src, n, valid).clamp(min=1e-12)
    margin = _node_sum(w * s_star, src, n, valid) / abs_sum
    d_eff = abs_sum ** 2 / _node_sum(w * w, src, n, valid).clamp(min=1e-12)
    return margin, d_eff, margin * d_eff.sqrt()


def node_lift(src, q, t, n, valid=None):
    """Node-wise lift: mean score on u's heterophilic edges minus mean score on
    its homophilic edges (the self-loop is homophilic with score 0). NaN for a
    node with no heterophilic edge."""
    het = _node_sum(t, src, n, valid)
    hom = _node_sum(1.0 - t, src, n, valid)
    q_het = _node_sum(q * t, src, n, valid) / het.clamp(min=1.0)
    q_hom = _node_sum(q * (1.0 - t), src, n, valid) / hom.clamp(min=1.0)
    lift = q_het - q_hom
    lift[(het == 0) | (hom == 0)] = float('nan')
    return lift


def first_order_term(src, q, t, n, valid=None):
    """d chi_u / d eta at eta = 0 (Theorem 3(d)): 4 sqrt(d~_u) h~_u (1 - h~_u) lift_u."""
    h, deg = local_heterophily(src, t, n, valid)
    lift = torch.nan_to_num(node_lift(src, q, t, n, valid), nan=0.0)
    return 4.0 * deg.sqrt() * h * (1.0 - h) * lift


def _mean(x, mask):
    x = x[mask]
    x = x[~torch.isnan(x)]
    return float(x.mean()) if x.numel() else float('nan')


def summarize_scores(graph, q, labels, known, eval_edges=None, etas=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """All score-level diagnostics of one score vector on one graph.

    eval_edges: edge mask for the edge-quality numbers (for example the edges
    with both endpoints outside the labeled set). The node-level and spectral
    quantities always use every valid edge.
    """
    src, dst = graph.edges()
    src, dst = src.cpu(), dst.cpu()
    q, labels, known = q.detach().cpu().float(), labels.cpu(), known.cpu()
    n = graph.num_nodes()
    self_loop = src == dst
    valid = valid_edges(src, dst, known)
    t = true_heterophily(src, dst, labels)
    anomalies, normals = known & (labels == 1), known & (labels == 0)

    out = {}
    q_edges = valid & ~self_loop if eval_edges is None else eval_edges.cpu() & valid & ~self_loop
    out['edge'] = edge_quality(q, t, q_edges, src, dst, labels)
    out['edge_all'] = edge_quality(q, t, valid & ~self_loop, src, dst, labels)

    eta_min, rho_min = label_frequency_minimiser(q, t, valid, self_loop)
    out['label_freq'] = {'by_eta': dict(zip([str(e) for e in etas], label_frequency_curve(q, t, etas, valid, self_loop))),
                         'min_eta': eta_min, 'min_value': rho_min}

    h, deg = local_heterophily(src, t, n, valid)
    lift = node_lift(src, q, t, n, valid)
    first = first_order_term(src, q, t, n, valid)
    out['node'] = {
        'h_anomalies': _mean(h, anomalies), 'h_normals': _mean(h, normals),
        'anomalies_h_above_half': float((h[anomalies] > 0.5).float().mean()),
        'lift_anomalies': _mean(lift, anomalies), 'lift_normals': _mean(lift, normals),
        'anomalies_positive_lift': float((lift[anomalies] > 0).float().mean()),
        'first_order_anomalies': _mean(first, anomalies), 'first_order_normals': _mean(first, normals),
    }
    chi = {}
    for e in etas:
        _, _, c = node_factors(src, edge_weight(q, e, self_loop=self_loop), t, n, valid)
        chi[str(e)] = {'anomalies': _mean(c, anomalies), 'normals': _mean(c, normals),
                       'anomalies_wrong_side': float((c[anomalies] < 0).float().mean())}
    out['chi_by_eta'] = chi

    w1 = edge_weight(q, 1.0, self_loop=self_loop)
    e_ok = valid & ~self_loop
    het, hom = e_ok & (t > 0.5), e_ok & (t < 0.5)
    a_s, a_d = labels[src] == 1, labels[dst] == 1
    out['weights_at_eta1'] = {
        'mean_heterophilic': float(w1[het].mean()), 'mean_homophilic': float(w1[hom].mean()),
        'negative_heterophilic': float((w1[het] < 0).float().mean()),
        'negative_aa': float((w1[e_ok & a_s & a_d] < 0).float().mean()) if (e_ok & a_s & a_d).any() else float('nan'),
        'negative_nn': float((w1[e_ok & ~a_s & ~a_d] < 0).float().mean()),
    }
    return out
