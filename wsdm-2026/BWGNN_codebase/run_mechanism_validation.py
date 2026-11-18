"""Mechanism-validation study for weak_edge_supervision.tex's E_LU extension,
on GADBench datasets beyond Amazon/YelpChi.

Tests the hypothesized causal chain

    |E_LU|/|E_LL| -> pseudo-label quality -> Edge AP/Brier -> r_- -> Node AP

across reddit, weibo, tolokers, tfinance at several label ratios, comparing:
  - Original EC             : lambda_p = 0.0  (E_LL only)
  - EC + Weak (lambda_p=0.5): lambda_p = 0.5  (E_LL + E_LU, main config)
  - Oracle-E_LU              : lambda_p = 1.0, oracle_lu=True (offline upper bound)

Writes one CSV row per (dataset, config, ratio, run) with the full
diagnostic set: |E_LL|, |E_LU|, |E_LL^hetero|, pseudo-label accuracy
(overall + anomaly-related), Edge AP/AUROC/Brier/ECE, learned eta, r_-,
and Node AP/AUROC/Macro-F1.
"""
import argparse
import csv
import os
import time
import warnings

import torch

import main_ec_sbwgnn as mem
from dataset import Dataset

warnings.filterwarnings("ignore")

CONFIGS = [
    ('Original EC', 0.00, False),
    ('EC + Weak (0.5)', 0.50, False),
    ('Oracle-E_LU', 1.00, True),
]
RATIOS = [0.005, 0.01, 0.05, 0.1, 0.4]
DATASETS = ['reddit', 'weibo', 'tolokers', 'tfinance']

FIELDNAMES = [
    'dataset', 'config', 'lambda_p', 'oracle_lu', 'train_ratio', 'run',
    'node_auroc', 'node_ap', 'node_macro_f1', 'node_recall', 'node_precision',
    'edge_auc', 'edge_ap', 'edge_brier', 'edge_bce', 'edge_ece',
    'num_E_LL', 'num_E_LU', 'num_E_LL_hetero', 'num_E_LU_anomaly_related',
    'pseudo_lu_accuracy', 'pseudo_lu_accuracy_anomaly_related',
    'num_trained_folds', 'eta', 'r_minus',
    'edge_cf_time_s', 'node_train_time_s',
]


def make_args(train_ratio, lambda_p, oracle_lu, edge_epochs, epoch, edge_folds, device=None):
    return argparse.Namespace(
        train_ratio=train_ratio, hid_dim=64, order=2, epoch=epoch,
        edge_folds=edge_folds, edge_hidden=32, edge_epochs=edge_epochs,
        edge_lr=1e-2, edge_wd=1e-5, max_edges_cn=5_000_000,
        eta_init=0.5, lambda_g=0.01, lambda_p=lambda_p, oracle_lu=oracle_lu,
        device=device,
    )


def run_sweep(datasets, out_path, edge_epochs, epoch, edge_folds, runs_per_cell, ratios, configs, seed0=0, device=None):
    write_header = not os.path.exists(out_path)
    f = open(out_path, 'a', newline='')
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
        f.flush()

    for dataset_name in datasets:
        print(f'\n===== dataset={dataset_name} =====', flush=True)
        mem.dataset_name = dataset_name
        g = Dataset(dataset_name, homo=True).graph
        n_anom = int(g.ndata['label'].sum().item())
        print(f'  N={g.num_nodes()} E={g.num_edges()} anomalies={n_anom} ({100*n_anom/g.num_nodes():.2f}%)', flush=True)

        for ratio in ratios:
            for cfg_name, lambda_p, oracle_lu in configs:
                for run_idx in range(runs_per_cell):
                    seed = seed0 + run_idx
                    torch.manual_seed(seed)
                    args = make_args(ratio, lambda_p, oracle_lu, edge_epochs, epoch, edge_folds, device=device)
                    t0 = time.time()
                    try:
                        result = mem.train(g, args, return_dict=True)
                    except Exception as e:
                        print(f'  FAILED dataset={dataset_name} cfg={cfg_name} ratio={ratio} run={run_idx}: {e}', flush=True)
                        continue
                    dt = time.time() - t0
                    row = {
                        'dataset': dataset_name, 'config': cfg_name, 'lambda_p': lambda_p,
                        'oracle_lu': oracle_lu, 'train_ratio': ratio, 'run': run_idx,
                    }
                    row.update({k: result.get(k) for k in FIELDNAMES if k in result})
                    writer.writerow(row)
                    f.flush()
                    print('  [{}] cfg={:18s} ratio={:<6} run={} node_auroc={:.4f} node_ap={:.4f} '
                          'edge_ap={} r_-={} eta={} |E_LL|={} |E_LU|={} pseudo_acc={} ({:.1f}s)'.format(
                              dataset_name, cfg_name, ratio, run_idx,
                              result['node_auroc'], result['node_ap'],
                              round(result['edge_ap'], 4) if result['edge_ap'] is not None else None,
                              round(result['r_minus'], 4) if result['r_minus'] is not None else None,
                              round(result['eta'], 4) if result['eta'] is not None else None,
                              result['num_E_LL'], result['num_E_LU'],
                              round(result['pseudo_lu_accuracy'], 4) if result.get('pseudo_lu_accuracy') is not None else None,
                              dt), flush=True)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default=','.join(DATASETS), help='comma-separated dataset names')
    parser.add_argument('--out', type=str, default='mechanism_validation_results.csv')
    parser.add_argument('--edge_epochs', type=int, default=60)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--edge_folds', type=int, default=5)
    parser.add_argument('--runs_per_cell', type=int, default=1)
    parser.add_argument('--seed0', type=int, default=0)
    parser.add_argument('--device', type=str, default=None, help='torch device, e.g. cuda / cpu')
    cli = parser.parse_args()

    run_sweep(cli.datasets.split(','), cli.out, cli.edge_epochs, cli.epoch, cli.edge_folds,
              cli.runs_per_cell, RATIOS, CONFIGS, cli.seed0, device=cli.device)
    print('\nDone. Results ->', cli.out)
