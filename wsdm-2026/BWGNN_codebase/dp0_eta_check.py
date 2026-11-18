"""DP0: can the learned eta be read, and does a fixed eta matter?

For each dataset and each of several random draws of a GADBench-style labeled
set (100 training nodes with 20 anomalies, the same again for validation; NOT
the shipped splits), Phase 1 runs once and then, on the same frozen scores:

    (a) eta is learned from several starting values for a long run without
        early stopping; eta at epoch 100 and at the end, and the training
        loss at epoch 100, are recorded;
    (b) eta is fixed on a grid and the model is trained under the GADBench
        loop (selection by validation AUPRC);
    (c) the reliability slope of the scores and the eta that minimises the
        label frequency are computed with all labels (analysis only).

    python dp0_eta_check.py --datasets amazon,yelp --out ../../output/phase1/dp0/dp0.jsonl
    python dp0_eta_check.py --summarize ../../output/phase1/dp0/dp0.jsonl
"""
import argparse
import json
import os

import numpy as np
import torch

import diagnostics as dg
import gadbench
import main_gadbench as mg

STARTS = (0.2, 0.5, 0.8)
GRID = tuple(np.round(np.arange(0.0, 1.0001, 0.125), 3))


def run_draw(graph, labels, known, draw, device, cfg_overrides, starts=STARTS, grid=GRID,
             long_epochs=500, grid_epochs=100):
    """All runs of one draw on one graph. Returns a JSON-serializable dict."""
    masks = gadbench.random_split(labels, known, np.random.RandomState(draw))
    seed = gadbench.trial_seed(draw)
    graph_dev, labels_dev = graph.to(device), labels.to(device)

    cfg = mg.default_config(**cfg_overrides)
    gadbench.set_seed(seed)
    q, edge_diag, _ = mg.calibrate(graph, labels, masks[0], cfg, seed, device, known)

    src, dst = graph.edges()
    unl_pair = ~masks[0][src] & ~masks[0][dst]
    scores = dg.summarize_scores(graph, q, labels, known, eval_edges=unl_pair)
    row = {'draw': draw, 'seed': seed, 'edge': edge_diag,
           'slope': scores['edge']['slope'], 'edge_auroc': scores['edge'].get('auroc'),
           'label_freq_min_eta': scores['label_freq']['min_eta'], 'label_freq_min': scores['label_freq']['min_value'],
           'lift_anomalies': scores['node']['lift_anomalies'], 'learned': {}, 'fixed': {}}

    for start in starts:                                    # (a) learned eta, no early stopping
        c = mg.default_config(epochs=long_epochs, patience=long_epochs, eta_init=float(start), **cfg_overrides)
        gadbench.set_seed(seed)
        r = mg.train_node_model(graph_dev, labels_dev, masks, q, c, device, log_eta=True)
        log = r['eta_log']
        row['learned'][str(start)] = {'eta_100': log[min(99, len(log) - 1)], 'eta_end': log[-1],
                                      'test_auprc': r['test']['AUPRC'], 'final_train_loss': r['final_train_loss']}
    c = mg.default_config(epochs=100, patience=100, **cfg_overrides)
    gadbench.set_seed(seed)
    row['train_loss_100'] = mg.train_node_model(graph_dev, labels_dev, masks, q, c, device)['final_train_loss']

    for eta in grid:                                        # (b) fixed eta
        c = mg.default_config(epochs=grid_epochs, patience=grid_epochs, fix_eta=True, eta_init=float(eta), **cfg_overrides)
        gadbench.set_seed(seed)
        r = mg.train_node_model(graph_dev, labels_dev, masks, q, c, device)
        row['fixed'][str(eta)] = {'test_auprc': r['test']['AUPRC'], 'test_auroc': r['test']['AUROC'],
                                  'val_auprc': r['val']['AUPRC']}
    return row


def summarize(path):
    from scipy.stats import wilcoxon
    rows = [json.loads(line) for line in open(path)]
    for name in sorted({r['dataset'] for r in rows}):
        rs = [r for r in rows if r['dataset'] == name]
        starts = sorted(rs[0]['learned'], key=float)
        grid = sorted(rs[0]['fixed'], key=float)
        print('\n== {} ({} draws)'.format(name, len(rs)))
        for key in ('eta_100', 'eta_end'):
            agree = sum(max(r['learned'][s][key] for s in starts) - min(r['learned'][s][key] for s in starts) <= 0.05
                        for r in rs)
            print('  draws in which the starts end within 0.05 of each other ({}): {} of {}'.format(key, agree, len(rs)))
        print('  mean final eta by start: ' + ' / '.join(
            '{} -> {:.3f}'.format(s, np.mean([r['learned'][s]['eta_end'] for r in rs])) for s in starts))
        moved = [abs(r['learned'][s]['eta_end'] - float(s)) for r in rs for s in starts]
        print('  distance of eta from its start: mean {:.3f}, largest {:.3f}'.format(np.mean(moved), np.max(moved)))
        print('  training loss at epoch 100: {:.4f}'.format(np.mean([r['train_loss_100'] for r in rs])))
        slopes = np.array([r['slope'] for r in rs])
        print('  reliability slope: {:.2f} (sd {:.2f}); label-frequency minimiser: {:.2f} (sd {:.2f})'.format(
            slopes.mean(), slopes.std(), np.mean([r['label_freq_min_eta'] for r in rs]),
            np.std([r['label_freq_min_eta'] for r in rs])))
        ap = np.array([[r['fixed'][g]['test_auprc'] for g in grid] for r in rs])
        print('  test AUPRC by fixed eta: ' + ' / '.join('{}: {:.3f}'.format(g, m) for g, m in zip(grid, ap.mean(0))))
        print('  sd over draws by fixed eta: {:.3f} to {:.3f}'.format(ap.std(0).min(), ap.std(0).max()))
        best = int(np.argmax(ap.mean(0)[1:])) + 1
        diff = ap[:, best] - ap[:, 0]
        p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
        print('  best fixed eta {} against eta=0: mean gain {:+.3f}, draws won {} of {}, paired Wilcoxon p = {:.2f}'.format(
            grid[best], diff.mean(), int((diff > 0).sum()), len(rs), p))
        best_per_draw = np.array([float(grid[i]) for i in ap.argmax(1)])
        if best_per_draw.std() > 0 and slopes.std() > 0:
            print('  correlation over draws of the best fixed eta with the slope: {:.2f}'.format(
                np.corrcoef(best_per_draw, slopes)[0, 1]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datasets', type=str, default='amazon,yelp')
    p.add_argument('--draws', type=int, default=10)
    p.add_argument('--long_epochs', type=int, default=500)
    p.add_argument('--grid_epochs', type=int, default=100)
    p.add_argument('--edge_pos_weight', type=int, default=0)
    p.add_argument('--edge_target', type=str, default='e1e2')
    p.add_argument('--exact_overlap', action='store_true')
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--out', type=str, default=None)
    p.add_argument('--summarize', type=str, default=None, help='Print the summary of an existing output file and exit')
    args = p.parse_args()
    if args.summarize:
        return summarize(args.summarize)

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    overrides = dict(edge_pos_weight=args.edge_pos_weight, edge_target=args.edge_target,
                     exact_overlap=args.exact_overlap)
    for name in args.datasets.split(','):
        graph = gadbench.load_graph(name)
        labels, known = graph.ndata['label'], gadbench.known_label_mask(graph)
        for draw in range(args.draws):
            row = run_draw(graph, labels, known, draw, device, overrides,
                           long_epochs=args.long_epochs, grid_epochs=args.grid_epochs)
            row.update({'dataset': name, 'config': overrides})
            print('{} draw {}: slope {:.2f}, minimiser {:.2f}, learned {} , fixed AUPRC {}'.format(
                name, draw, row['slope'], row['label_freq_min_eta'],
                {s: round(v['eta_end'], 3) for s, v in row['learned'].items()},
                {g: round(v['test_auprc'], 3) for g, v in row['fixed'].items()}), flush=True)
            if args.out:
                os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                with open(args.out, 'a') as f:
                    f.write(json.dumps(row) + '\n')
    if args.out:
        summarize(args.out)


if __name__ == '__main__':
    main()
