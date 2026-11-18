"""Why the loss is flat in eta, and what would remove the flat region.

Per dataset and random draw of a GADBench-style labeled set (100 labels):

  part 1 (paper's edge target): the node factor chi_u of Theorem 3(a) by class
      on a grid of eta, the first-order term of Theorem 3(d), the node-wise
      lift, the weights at eta = 1 by edge type, and test AUPRC of the model at
      eta = 0, at eta = 1 with the calibrated scores, with oracle signs at
      eta = 1 and with oracle pruning;
  part 2 (targets): for the paper's target, the soft-imputed target and the
      node-posterior product: the lift of anomalies, chi_u at eta = 1 and test
      AUPRC at a fixed eta; next to them the feature-only reference (the same
      encoder and head with no graph) and the cross-fitted node posterior.

Uses ALL labels for the chi / lift / oracle quantities: analysis only.

    python flat_eta_check.py --datasets amazon,yelp --out_dir ../../output/phase1/flat_eta
"""
import argparse
import json
import os

import numpy as np
import torch

import diagnostics as dg
import gadbench
import main_gadbench as mg


def _train(graph_dev, labels_dev, masks, q, seed, device, epochs, **kw):
    cfg = mg.default_config(epochs=epochs, patience=epochs, **kw)
    gadbench.set_seed(seed)
    return mg.train_node_model(graph_dev, labels_dev, masks, q, cfg, device)['test']['AUPRC']


def run_draw(graph, labels, known, draw, device, epochs=100, eta_targets=0.75, edge_epochs=150, exact_overlap=False):
    masks = gadbench.random_split(labels, known, np.random.RandomState(draw))
    seed = gadbench.trial_seed(draw)
    graph_dev, labels_dev = graph.to(device), labels.to(device)
    src, dst = graph.edges()
    unl_pair = ~masks[0][src] & ~masks[0][dst]
    base = dict(edge_epochs=edge_epochs, exact_overlap=exact_overlap)
    row = {'draw': draw, 'seed': seed}

    def scores_for(**kw):
        cfg = mg.default_config(**base, **kw)
        gadbench.set_seed(seed)
        return mg.calibrate(graph, labels, masks[0], cfg, seed, device, known)

    # ---- part 1: the paper's target --------------------------------------------
    q, _, _ = scores_for(edge_target='e1e2')
    row['paper'] = dg.summarize_scores(graph, q, labels, known, eval_edges=unl_pair)
    q_oracle, _, _ = scores_for(q_mode='oracle')
    row['oracle'] = dg.summarize_scores(graph, q_oracle, labels, known)
    t = dg.true_heterophily(src, dst, labels)
    valid = dg.valid_edges(src, dst, known)
    an = known & (labels == 1)
    chi_prune = dg.node_factors(src, dg.edge_weight(t, None, 'prune', 0.5, src == dst), t, graph.num_nodes(), valid)[2]
    row['oracle']['chi_prune_anomalies'] = float(chi_prune[an].mean())
    row['auprc'] = {
        'eta0': _train(graph_dev, labels_dev, masks, q, seed, device, epochs, fix_eta=True, eta_init=0.0),
        'calibrated_eta1': _train(graph_dev, labels_dev, masks, q, seed, device, epochs, fix_eta=True, eta_init=1.0),
        'oracle_sign_eta1': _train(graph_dev, labels_dev, masks, q_oracle, seed, device, epochs, fix_eta=True, eta_init=1.0),
        'oracle_prune': _train(graph_dev, labels_dev, masks, q_oracle, seed, device, epochs, variant='prune', tau=0.5),
        'feature_only_same_head': _train(graph_dev, labels_dev, masks, q, seed, device, epochs, variant='none'),
    }

    # ---- part 2: targets --------------------------------------------------------
    row['targets'] = {}
    for name, kw in (('paper', dict(edge_target='e1e2')), ('soft', dict(edge_target='soft')),
                     ('product', dict(q_mode='product'))):
        q_t, _, extras = scores_for(**kw)
        s = dg.summarize_scores(graph, q_t, labels, known, eval_edges=unl_pair)
        row['targets'][name] = {
            'lift_anomalies': s['node']['lift_anomalies'], 'anomalies_positive_lift': s['node']['anomalies_positive_lift'],
            'chi_anomalies_eta1': s['chi_by_eta']['1.0']['anomalies'], 'edge_auroc': s['edge'].get('auroc'),
            'auprc_fixed_eta': _train(graph_dev, labels_dev, masks, q_t, seed, device, epochs,
                                      fix_eta=True, eta_init=eta_targets)}
        if extras.get('pi_oof') is not None and 'node_posterior_auprc' not in row:
            test = masks[2]
            row['node_posterior_auprc'] = gadbench.evaluate(labels[test], extras['pi_oof'][test])['AUPRC']
    return row


def summarize(rows):
    def mean(f):
        vals = [f(r) for r in rows]
        return float(np.mean(vals))
    etas = list(rows[0]['paper']['chi_by_eta'])
    print('  chi of anomalies by eta (calibrated): ' + ' / '.join(
        '{}: {:+.1f}'.format(e, mean(lambda r: r['paper']['chi_by_eta'][e]['anomalies'])) for e in etas))
    print('  anomalies on the wrong side at eta=1: {:.0%}'.format(
        mean(lambda r: r['paper']['chi_by_eta']['1.0']['anomalies_wrong_side'])))
    print('  chi of anomalies, oracle signs / oracle pruning: {:+.1f} / {:+.1f}'.format(
        mean(lambda r: r['oracle']['chi_by_eta']['1.0']['anomalies']), mean(lambda r: r['oracle']['chi_prune_anomalies'])))
    print('  first-order term, anomalies / normal nodes: {:+.2f} / {:+.2f}'.format(
        mean(lambda r: r['paper']['node']['first_order_anomalies']), mean(lambda r: r['paper']['node']['first_order_normals'])))
    print('  node-wise lift, anomalies / normal nodes: {:+.2f} / {:+.2f}'.format(
        mean(lambda r: r['paper']['node']['lift_anomalies']), mean(lambda r: r['paper']['node']['lift_normals'])))
    w = 'weights_at_eta1'
    print('  mean weight at eta=1, heterophilic / homophilic: {:+.2f} / {:+.2f}'.format(
        mean(lambda r: r['paper'][w]['mean_heterophilic']), mean(lambda r: r['paper'][w]['mean_homophilic'])))
    print('  negative at eta=1, heterophilic / anomaly-anomaly / normal-normal: {:.0%} / {:.0%} / {:.0%}'.format(
        mean(lambda r: r['paper'][w]['negative_heterophilic']), mean(lambda r: r['paper'][w]['negative_aa']),
        mean(lambda r: r['paper'][w]['negative_nn'])))
    print('  test AUPRC: ' + ', '.join('{} {:.3f}'.format(k, mean(lambda r: r['auprc'][k])) for k in rows[0]['auprc']))
    for name in rows[0]['targets']:
        tg = lambda r: r['targets'][name]
        print('  target {:<8} lift of anomalies {:+.2f}, positive {:.0%}, chi at eta=1 {:+.1f}, AUPRC at fixed eta {:.3f}'.format(
            name, mean(lambda r: tg(r)['lift_anomalies']), mean(lambda r: tg(r)['anomalies_positive_lift']),
            mean(lambda r: tg(r)['chi_anomalies_eta1']), mean(lambda r: tg(r)['auprc_fixed_eta'])))
    print('  node posterior alone (features, no graph), test AUPRC: {:.3f}'.format(mean(lambda r: r['node_posterior_auprc'])))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datasets', type=str, default='amazon,yelp')
    p.add_argument('--draws', type=int, default=3)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--eta_targets', type=float, default=0.75)
    p.add_argument('--exact_overlap', action='store_true')
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--out_dir', type=str, default=None)
    args = p.parse_args()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    for name in args.datasets.split(','):
        graph = gadbench.load_graph(name)
        labels, known = graph.ndata['label'], gadbench.known_label_mask(graph)
        rows = [run_draw(graph, labels, known, d, device, args.epochs, args.eta_targets, exact_overlap=args.exact_overlap)
                for d in range(args.draws)]
        print('\n== {} ({} draws)'.format(name, len(rows)))
        summarize(rows)
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            with open(os.path.join(args.out_dir, name + '.json'), 'w') as f:
                json.dump(rows, f, indent=1)


if __name__ == '__main__':
    main()
