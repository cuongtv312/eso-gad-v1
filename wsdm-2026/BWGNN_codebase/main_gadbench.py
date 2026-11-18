"""ESO-GAD under the GADBench semi-supervised protocol (see gadbench.py).

One trial = Phase 1 (cross-fitted edge calibration, frozen) + Phase 2 (signed
Beta-wavelet model with early stopping on validation AUPRC). The default
configuration is the paper's: P=2, hidden 64, K=5, alpha=0.01, the edge target
of Sec. 4.1 (one loss over E_2 and E_1, BCE + Brier without class weight), a
learned eta started at 0.5.

Examples:
    python main_gadbench.py --dataset amazon --trials 0                 # one split
    python main_gadbench.py --dataset all --out runs/default.jsonl      # 7 datasets x 10 splits
    python main_gadbench.py --dataset yelp --fix_eta --eta_init 0       # the unsigned backbone in the same code
    python main_gadbench.py --dataset yelp --variant prune --tau 0.5
    python main_gadbench.py --dataset yelp --q_mode oracle --fix_eta --eta_init 1 --analyze

The options that reproduce the calibrator of the WSDM code are
    --edge_target legacy --lambda_p <value> --edge_pos_weight 1 --train_on_self_loops 1 --no_symmetrize
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import gadbench
from ec_sbwgnn import EC_SBWGNN, VARIANTS
from edge_calibration import cross_fitted_edge_calibration, cross_fitted_node_posterior, stratified_fold_assignment

Q_MODES = ('calibrated', 'oracle', 'shuffled', 'flipped', 'product')


def build_parser():
    p = argparse.ArgumentParser(description='ESO-GAD, GADBench semi-supervised protocol')
    p.add_argument('--dataset', type=str, default='amazon', help="comma-separated names, or 'all'")
    p.add_argument('--trials', type=str, default='0-9', help="e.g. '0', '0-9', '0,3,5'")
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--no_symmetrize', action='store_true',
                   help='Keep Weibo/Tolokers/Questions directed, as shipped (what the WSDM runs used)')
    p.add_argument('--out', type=str, default=None, help='Append one JSON line per trial to this file')
    p.add_argument('--save_q', type=str, default=None, help='Directory for the frozen scores, one file per dataset and trial')
    p.add_argument('--analyze', action='store_true',
                   help='Add the score-level diagnostics of diagnostics.py (uses ALL labels; analysis only)')
    p.add_argument('--tag', type=str, default='', help='Free-text label stored with each result')

    # Phase 2
    p.add_argument('--hid_dim', type=int, default=64)
    p.add_argument('--order', type=int, default=2, help='Beta-wavelet order P')
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--lambda_g', type=float, default=0.01, help='Weight alpha of the gate KL term')
    p.add_argument('--gate_mode', type=str, default='learnable', choices=('learnable', 'uniform'))
    p.add_argument('--eta_init', type=float, default=0.5)
    p.add_argument('--fix_eta', action='store_true', help='Keep eta fixed at --eta_init')
    p.add_argument('--eta_max', type=float, default=1.0, help='Upper end of the range of a learned eta')
    p.add_argument('--eta_lr', type=float, default=None, help='Separate learning rate for eta')
    p.add_argument('--variant', type=str, default='sign', choices=VARIANTS)
    p.add_argument('--tau', type=float, default=None, help="Threshold of --variant prune")

    # Phase 1
    p.add_argument('--edge_target', type=str, default='e1e2', choices=('e2', 'e1e2', 'soft', 'legacy'))
    p.add_argument('--edge_pos_weight', type=int, default=0, help='1 = class-weighted BCE in the calibrator loss')
    p.add_argument('--train_on_self_loops', type=int, default=0)
    p.add_argument('--lambda_p', type=float, default=0.0, help="Weight of the E_1 term with --edge_target legacy")
    p.add_argument('--edge_folds', type=int, default=5)
    p.add_argument('--edge_hidden', type=int, default=32)
    p.add_argument('--edge_epochs', type=int, default=150)
    p.add_argument('--edge_lr', type=float, default=1e-2)
    p.add_argument('--edge_wd', type=float, default=1e-5)
    p.add_argument('--max_edges_cn', type=int, default=5_000_000)
    p.add_argument('--exact_overlap', action='store_true',
                   help='Exact Salton/Jaccard above --max_edges_cn (chunked on the GPU) instead of zeros')
    p.add_argument('--q_mode', type=str, default='calibrated', choices=Q_MODES,
                   help='calibrated: Phase 1 scores. oracle: true heterophily (analysis only). shuffled: the '
                        'calibrated scores permuted over edges. flipped: 1 - q. product: no edge model, '
                        'pi_u + pi_v - 2 pi_u pi_v from the cross-fitted node posterior')
    p.add_argument('--oracle_flip', type=float, default=0.0, help='Share of oracle scores flipped at random')
    return p


def default_config(**overrides):
    cfg = build_parser().parse_args([])
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError('unknown option {!r}'.format(k))
        setattr(cfg, k, v)
    return cfg


def parse_trials(spec):
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def undirected_pair_index(graph):
    """Index of the unordered pair {u, v} of every directed edge."""
    src, dst = graph.edges()
    n = graph.num_nodes()
    key = torch.minimum(src, dst) * n + torch.maximum(src, dst)
    _, inverse = torch.unique(key, return_inverse=True)
    return inverse


def calibrate(graph, labels, train_mask, cfg, seed, device, known=None):
    """Phase 1. Returns (q on device, diagnostics, extras)."""
    feat = graph.ndata['feature']
    src, dst = graph.edges()
    self_loop = src == dst
    extras = {}
    if cfg.q_mode == 'oracle':
        q = (labels[src] != labels[dst]).float()
        if cfg.oracle_flip > 0:
            pair = undirected_pair_index(graph)
            flip = torch.rand(int(pair.max()) + 1, generator=torch.Generator().manual_seed(seed)) < cfg.oracle_flip
            q = torch.where(flip.to(q.device)[pair], 1.0 - q, q)
        q = torch.where(self_loop, torch.zeros_like(q), q)
        return q.to(device), {'q_mode': 'oracle', 'oracle_flip': cfg.oracle_flip}, extras

    if cfg.q_mode == 'product':
        fold_of = stratified_fold_assignment(labels.cpu(), train_mask.cpu(), cfg.edge_folds, seed)
        _, pi = cross_fitted_node_posterior(feat, labels, train_mask, fold_of, cfg.edge_folds, device=device)
        pi = pi.to(src.device)
        q = pi[src] + pi[dst] - 2.0 * pi[src] * pi[dst]
        q = torch.where(self_loop, torch.zeros_like(q), q)
        return q.to(device), {'q_mode': 'product'}, {'pi_oof': pi.cpu(), 'fold_of': fold_of}

    legacy = cfg.edge_target == 'legacy'
    q, diag, extras = cross_fitted_edge_calibration(
        graph, feat, labels, train_mask,
        k=cfg.edge_folds, hidden_dim=cfg.edge_hidden, epochs=cfg.edge_epochs,
        lr=cfg.edge_lr, weight_decay=cfg.edge_wd, seed=seed,
        max_edges_for_common_neighbors=cfg.max_edges_cn,
        lambda_p=cfg.lambda_p if legacy else 0.0, device=device,
        edge_target=None if legacy else cfg.edge_target,
        pos_weight=bool(cfg.edge_pos_weight), train_on_self_loops=bool(cfg.train_on_self_loops),
        exact_overlap=cfg.exact_overlap, return_extras=True)
    diag = {k: v for k, v in diag.items() if isinstance(v, (int, float, bool, str)) or v is None}

    if cfg.q_mode == 'shuffled':
        pair = undirected_pair_index(graph).to(q.device)
        num_pairs = int(pair.max()) + 1
        per_pair = torch.zeros(num_pairs, device=q.device).scatter_(0, pair, q)
        loop_pair = torch.zeros(num_pairs, dtype=torch.bool, device=q.device).scatter_(0, pair, self_loop.to(q.device))
        idx = (~loop_pair).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(idx.numel(), generator=torch.Generator().manual_seed(seed)).to(idx.device)]
        shuffled = per_pair.clone()
        shuffled[idx] = per_pair[perm]
        q = shuffled[pair]
    elif cfg.q_mode == 'flipped':
        q = torch.where(self_loop.to(q.device), torch.zeros_like(q), 1.0 - q)
    diag['q_mode'] = cfg.q_mode
    return q, diag, extras


def train_node_model(graph, labels, masks, q, cfg, device, return_model=False, log_eta=False):
    """Phase 2 with GADBench's loop: one full-batch step per epoch, then an
    evaluation pass; the test scores are those at the best validation AUPRC."""
    train_mask, val_mask, test_mask = (m.to(device) for m in masks)
    feat = graph.ndata['feature']
    model = EC_SBWGNN(feat.shape[1], cfg.hid_dim, 2, graph, q, d=cfg.order, eta_init=cfg.eta_init,
                      learn_eta=not cfg.fix_eta, eta_max=cfg.eta_max, variant=cfg.variant, tau=cfg.tau,
                      gate_mode=cfg.gate_mode).to(device)
    if cfg.eta_lr is not None and model.learn_eta:
        optimizer = torch.optim.Adam([
            {'params': [p for n, p in model.named_parameters() if n != 'eta_raw'], 'lr': cfg.lr},
            {'params': [model.eta_raw], 'lr': cfg.eta_lr}])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    y_train = labels[train_mask]
    weight = (1 - y_train).sum().item() / max(y_train.sum().item(), 1)
    ce_weight = torch.tensor([1., weight], device=device)
    y_val, y_test = labels[val_mask].cpu().numpy(), labels[test_mask].cpu().numpy()

    best = {'val': {'AUPRC': -1.0}}
    best_state, patience_count, eta_log = None, 0, []
    t0 = time.time()
    epochs_run = 0
    for epoch in range(cfg.epochs):
        model.train()
        logits = model(feat)
        node_loss = F.cross_entropy(logits[train_mask], y_train, weight=ce_weight)
        loss = node_loss + cfg.lambda_g * model.gate_kl_regularizer()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epochs_run = epoch + 1

        model.eval()
        with torch.no_grad():
            logits, diag = model(feat, return_diagnostics=True)
            probs = logits.softmax(1)[:, 1]
        if log_eta:
            eta_log.append(float(diag['eta']))
        val = gadbench.evaluate(y_val, probs[val_mask])
        if val['AUPRC'] > best['val']['AUPRC']:
            patience_count = 0
            best = {'val': val, 'test': gadbench.evaluate(y_test, probs[test_mask]), 'epoch': epoch,
                    'eta': float(diag['eta']), 'r_minus': float(diag['r_minus']),
                    'gate': diag['gate'].tolist(), 'channel_energy': diag['channel_energy'].tolist(),
                    'train_loss': float(node_loss)}
            if return_model:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count > cfg.patience:
                break
    train_time = time.time() - t0

    result = dict(best)
    result.update({'final_eta': float(diag['eta']), 'final_train_loss': float(node_loss),
                   'epochs_run': epochs_run, 'train_time_s': train_time,
                   'time_per_epoch_s': train_time / max(epochs_run, 1)})
    if log_eta:
        result['eta_log'] = eta_log
    if return_model:
        model.load_state_dict(best_state)
        return result, model
    return result


def run_trial(graph, labels, masks, cfg, seed, device, known=None, return_model=False):
    """graph, labels and masks may live on any device; they are moved to `device`."""
    gadbench.set_seed(seed)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    graph_dev = graph.to(device)
    labels_dev = labels.to(device)
    t0 = time.time()
    q, edge_diag, extras = calibrate(graph, labels.cpu(), masks[0].cpu(), cfg, seed, device, known)
    phase1 = time.time() - t0

    out = train_node_model(graph_dev, labels_dev, masks, q, cfg, device, return_model=return_model)
    result, model = out if return_model else (out, None)
    result.update({'seed': seed, 'phase1_time_s': phase1, 'edge': edge_diag,
                   'num_edges': graph.num_edges(),
                   'peak_gpu_gb': torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == 'cuda' else 0.0})
    if cfg.analyze:
        from diagnostics import summarize_scores
        src, dst = graph.edges()
        labeled = masks[0].cpu()
        unlabeled_pair = ~labeled[src.cpu()] & ~labeled[dst.cpu()]
        result['oracle_diag'] = summarize_scores(graph, q, labels.cpu(), known.cpu(), eval_edges=unlabeled_pair)
    if return_model:
        return result, q, model
    return result, q


def summarize(results):
    """Mean and population standard deviation over trials, in percent."""
    out = {}
    for metric in ('AUROC', 'AUPRC', 'RecK'):
        vals = np.array([r['test'][metric] for r in results]) * 100
        out[metric] = (float(vals.mean()), float(vals.std()))
    return out


def main():
    cfg = build_parser().parse_args()
    device = torch.device(cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    names = gadbench.DATASETS if cfg.dataset == 'all' else cfg.dataset.split(',')
    trials = parse_trials(cfg.trials)
    config = {k: v for k, v in vars(cfg).items() if k not in ('out', 'save_q', 'device')}
    print(config)

    for name in names:
        graph = gadbench.load_graph(name, symmetrize=not cfg.no_symmetrize)
        labels = graph.ndata['label']
        known = gadbench.known_label_mask(graph)
        results = []
        for trial in trials:
            masks = gadbench.split_masks(graph, trial)
            seed = gadbench.trial_seed(trial)
            result, q = run_trial(graph, labels, masks, cfg, seed, device, known)
            result.update({'dataset': name, 'trial': trial, 'config': config})
            results.append(result)
            print('{} trial {}: test AUROC {:.2f} AUPRC {:.2f} RecK {:.2f} | val AUPRC {:.2f} | best epoch {} of {} | '
                  'eta {:.3f} r_- {:.4f} | phase 1 {:.1f}s, phase 2 {:.1f}s ({:.3f} s/epoch), peak {:.1f} GB'.format(
                      name, trial, 100 * result['test']['AUROC'], 100 * result['test']['AUPRC'],
                      100 * result['test']['RecK'], 100 * result['val']['AUPRC'], result['epoch'],
                      result['epochs_run'], result['eta'], result['r_minus'], result['phase1_time_s'],
                      result['train_time_s'], result['time_per_epoch_s'], result['peak_gpu_gb']), flush=True)
            if cfg.out:
                os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)
                with open(cfg.out, 'a') as f:
                    f.write(json.dumps(result) + '\n')
            if cfg.save_q:
                os.makedirs(cfg.save_q, exist_ok=True)
                torch.save({'q': q.cpu(), 'dataset': name, 'trial': trial, 'seed': seed, 'config': config},
                           os.path.join(cfg.save_q, '{}_trial{}.pt'.format(name, trial)))
        s = summarize(results)
        print('{} over {} trial(s): AUROC {:.2f} +- {:.2f}  AUPRC {:.2f} +- {:.2f}  RecK {:.2f} +- {:.2f}'.format(
            name, len(results), *s['AUROC'], *s['AUPRC'], *s['RecK']), flush=True)


if __name__ == '__main__':
    main()
