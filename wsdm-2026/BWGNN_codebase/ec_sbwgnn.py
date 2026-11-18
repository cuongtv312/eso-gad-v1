"""EC-SBWGNN: Edge-Calibrated Signed Beta-Wavelet GNN.

Implements Sections 2.2-2.3 of the EC-SBWGNN note on top of BWGNN's own
PolyConv trick: BWGNN already computes a sum of Beta/Bernstein polynomials
of the (unsigned) normalized Laplacian by expanding each channel's fixed
`theta` coefficients via repeated application of L. SignedBetaWaveletConv
below reuses that exact recurrence but replaces the implicit unsigned,
unweighted L = I - D^{-1/2} A D^{-1/2} with the general signed, calibrated
L_eta = I - D^{-1/2} W^(eta) D^{-1/2}, whose per-edge normalized weight is
supplied externally (it depends on the learned eta and the frozen
edge-heterophily estimate q, so it cannot be derived from bare in-degrees
the way PolyConv does).

No edge score is used to gate node-to-node influence beyond this single,
symmetric, node-pair-derived sign; there is no PPR, restart vector, or
subgraph feedback loop anywhere in this file, matching the note's explicit
scope (Sec 1: "does not reuse Personalized PageRank, adaptive restart,
feedback walks, or subgraph extraction").
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from BWGNN import calculate_theta2


VARIANTS = ('sign', 'reweight', 'prune', 'none')


def build_signed_normalized_weight(graph, q, eta, self_loop_mask, variant='sign', tau=None):
    """Constructs the per-edge normalized signed weight

        w_norm[u,v] = W^(eta)_uv / sqrt(D^(eta)_u D^(eta)_v),  W^(eta)_uv = (1-eta) + eta*(1-2q_uv),

    so that message passing with w_norm computes (D^{-1/2} W^(eta) D^{-1/2}) h
    directly (edges not present in the graph implicitly have A_uv = 0 and
    never enter the sum, so the A_uv factor from the note is implicit here
    rather than multiplied explicitly). q is treated as a constant (frozen
    after cross-fitting); eta is the only differentiable quantity entering W.

    variant selects the map from the same frozen q to the edge weight, with
    everything else shared (the sign / reweight / prune comparison):
        sign     : W_uv = 1 - 2*eta*q_uv   (the method; eta=0 is the unsigned backbone)
        reweight : W_uv = 1 - q_uv         (the sign family at eta=1/2)
        prune    : W_uv = 1[q_uv < tau]
        none     : W_uv = 0 off the diagonal, so L=0 and every channel is a
                   multiple of the encoder output (the feature-only reference
                   with the same encoder and head)
    Self-loops always keep weight 1.
    """
    if variant == 'sign':
        s_hat = 1.0 - 2.0 * q
        s_hat = torch.where(self_loop_mask, torch.ones_like(s_hat), s_hat)  # a node cannot be heterophilic with itself
        W = (1.0 - eta) + eta * s_hat
    elif variant == 'reweight':
        W = torch.where(self_loop_mask, torch.ones_like(q), 1.0 - q)
    elif variant == 'prune':
        if tau is None:
            raise ValueError("variant='prune' requires tau")
        W = torch.where(self_loop_mask, torch.ones_like(q), (q < tau).to(q.dtype))
    elif variant == 'none':
        W = self_loop_mask.to(q.dtype)
    else:
        raise ValueError('unknown variant {!r}'.format(variant))
    absW = W.abs()

    with graph.local_scope():
        graph.edata['absW'] = absW.unsqueeze(-1)
        graph.update_all(fn.copy_e('absW', 'm'), fn.sum('m', 'h'))
        D = graph.ndata['h'].squeeze(-1).clamp(min=1e-6)

    D_invsqrt = torch.pow(D, -0.5)
    src, dst = graph.edges()
    w_norm = (W * D_invsqrt[src] * D_invsqrt[dst]).unsqueeze(-1)
    return w_norm, W, D


@torch.no_grad()
def estimate_spectral_range(graph, w_norm, num_iters=50):
    """Optional sanity check for Proposition 1 (sigma(L_eta) subset [0,2]):
    power-iterates L_eta and (2I - L_eta) to estimate the largest and
    smallest eigenvalues of L_eta. Not used during training."""
    n = graph.num_nodes()

    def apply_L(x):
        with graph.local_scope():
            graph.ndata['h'] = x
            graph.edata['w_norm'] = w_norm
            graph.update_all(fn.u_mul_e('h', 'w_norm', 'm'), fn.sum('m', 'h'))
            return x - graph.ndata['h']

    def power_iter(matvec):
        x = torch.randn(n, 1)
        x = x / x.norm()
        lam = 0.0
        for _ in range(num_iters):
            x = matvec(x)
            norm = x.norm()
            if norm < 1e-12:
                return 0.0
            x = x / norm
            lam = norm.item()
        return lam

    lambda_max = power_iter(apply_L)
    lambda_min_shifted = power_iter(lambda x: 2 * x - apply_L(x))  # largest eig of (2I - L)
    lambda_min = 2.0 - lambda_min_shifted
    return lambda_min, lambda_max


class SignedBetaWaveletConv(nn.Module):
    """Computes one Bernstein/Beta channel B_k^(C)(L_eta) h via the same
    theta-coefficient expansion BWGNN uses for the unsigned Laplacian,
    applied instead to the signed, eta-calibrated L_eta."""

    def __init__(self, theta):
        super().__init__()
        self._theta = theta
        self._k = len(theta)

    def forward(self, graph, feat, w_norm):
        with graph.local_scope():
            graph.edata['w_norm'] = w_norm

            def apply_L(x):
                graph.ndata['h'] = x
                graph.update_all(fn.u_mul_e('h', 'w_norm', 'm'), fn.sum('m', 'h'))
                return x - graph.ndata['h']

            h = self._theta[0] * feat
            cur = feat
            for k in range(1, self._k):
                cur = apply_L(cur)
                h = h + self._theta[k] * cur
        return h


class EC_SBWGNN(nn.Module):
    """q_calibrated is frozen (a buffer). eta and the channel gate are the only
    operator-side parameters.

    learn_eta=True  : eta = eta_max * sigmoid(eta_raw), started at eta_init
                      (eta_max=1 is the paper's range; Proposition 1 holds for
                      every eta >= 0, so eta_max=2 is a valid ablation).
    learn_eta=False : eta is fixed at eta_init exactly (0 and 1 included).
    variant / tau   : see build_signed_normalized_weight.
    gate_mode       : 'learnable' (softmax of trained logits) or 'uniform'.
    """

    def __init__(self, in_feats, h_feats, num_classes, graph, q_calibrated,
                 d=2, eta_init=0.5, gate_prior=None, learn_eta=True, eta_max=1.0,
                 variant='sign', tau=None, gate_mode='learnable'):
        super().__init__()
        self.g = graph
        src, dst = graph.edges()
        self.register_buffer('q', q_calibrated)
        self.register_buffer('self_loop_mask', src == dst)
        self.variant = variant
        self.tau = tau

        self.thetas = calculate_theta2(d=d)
        self.num_channels = len(self.thetas)
        self.convs = nn.ModuleList([SignedBetaWaveletConv(theta) for theta in self.thetas])

        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * self.num_channels, h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()

        self.learn_eta = learn_eta
        self.eta_max = float(eta_max)
        if learn_eta:
            frac = min(max(eta_init / self.eta_max, 1e-3), 1 - 1e-3)
            self.eta_raw = nn.Parameter(torch.logit(torch.tensor(float(frac))))
        else:
            self.register_buffer('eta_fixed', torch.tensor(float(eta_init)))

        self.gate_mode = gate_mode
        if gate_mode == 'uniform':
            self.register_buffer('gate_logits', torch.zeros(self.num_channels))
        else:
            self.gate_logits = nn.Parameter(torch.zeros(self.num_channels))
        if gate_prior is None:
            gate_prior = torch.full((self.num_channels,), 1.0 / self.num_channels)
        self.register_buffer('gate_prior', gate_prior)

    @property
    def eta(self):
        if self.learn_eta:
            return self.eta_max * torch.sigmoid(self.eta_raw)
        return self.eta_fixed

    def gate(self):
        return torch.softmax(self.gate_logits, dim=-1)

    def gate_kl_regularizer(self):
        """D_KL(g || g0), computed as kl_div(log g0, g) = sum g*(log g - log g0)."""
        g = self.gate()
        log_g0 = torch.log(self.gate_prior.clamp(min=1e-12))
        return F.kl_div(log_g0, g, reduction='sum')

    def encode(self, in_feat):
        """Encoder output H^(0), before any propagation."""
        h = self.act(self.linear(in_feat))
        return self.act(self.linear2(h))

    def channels(self, h, w_norm):
        """All Beta channels B_k(L_eta) h. The powers L_eta^j h are computed
        once and shared across channels (P sparse products per forward pass
        in place of (P+1)P)."""
        with self.g.local_scope():
            self.g.edata['w_norm'] = w_norm
            powers = [h]
            for _ in range(len(self.thetas[0]) - 1):
                self.g.ndata['h'] = powers[-1]
                self.g.update_all(fn.u_mul_e('h', 'w_norm', 'm'), fn.sum('m', 'h'))
                powers.append(powers[-1] - self.g.ndata['h'])
        return [sum(theta[j] * powers[j] for j in range(len(theta))) for theta in self.thetas]

    def forward(self, in_feat, return_diagnostics=False, eta=None, q=None):
        """eta / q override the model's own strength and scores for this call
        only (used to evaluate one trained model under another operator)."""
        eta = self.eta if eta is None else eta
        q = self.q if q is None else q
        w_norm, W, D = build_signed_normalized_weight(self.g, q, eta, self.self_loop_mask,
                                                      variant=self.variant, tau=self.tau)

        h = self.encode(in_feat)
        channel_outputs = self.channels(h, w_norm)
        g = self.gate()
        scaled = [torch.sqrt(g[k].clamp(min=1e-12)) * channel_outputs[k] for k in range(self.num_channels)]
        h_final = torch.cat(scaled, dim=-1)

        h = self.act(self.linear3(h_final))
        logits = self.linear4(h)

        if not return_diagnostics:
            return logits

        with torch.no_grad():
            energies = torch.stack([torch.sum(c ** 2) for c in channel_outputs])
            # r_- = |{(u,v): W^(eta)_uv < 0}| / |E| -- fraction of edges the
            # calibrated signing actually flips to heterophilic/contrastive.
            r_minus = (W < 0).float().mean()
        eta_t = eta.detach() if torch.is_tensor(eta) else torch.tensor(float(eta))
        diag = {'eta': eta_t, 'gate': g.detach(), 'channel_energy': energies, 'r_minus': r_minus.detach()}
        return logits, diag
