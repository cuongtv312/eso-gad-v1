"""Multi-seed confirmation at the GADBench-scale label ratio (~100 labeled
nodes), comparing Original EC against the pseudo-supervision configs, to
turn the single-seed exploratory sweep's winner into a statistically
defensible result at the ratio that actually matters for ECHO-GAD's own
semi-supervised protocol.
"""
import argparse
import csv
import os
import time
import warnings

import torch

import main_ec_sbwgnn as mem
from dataset import Dataset
from run_weak_edge_supervision_sweep import make_args, FIELDNAMES

warnings.filterwarnings("ignore")

# ~100 labeled nodes worth of train_ratio, computed against each dataset's
# usable index range (Amazon skips the first 3305 unlabeled-by-construction
# nodes; Yelp uses the full node set).
GADBENCH_SCALE_RATIO = {
    'amazon': 100 / (11944 - 3305),
    'yelp': 100 / 45954,
}

CONFIGS = [
    ('Original EC', 0.00, False),
    ('EC + Pseudo', 1.00, False),
    ('EC + Pseudo-0.5', 0.50, False),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--edge_epochs', type=int, default=60)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--edge_folds', type=int, default=5)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--device', type=str, default=None, help='torch device, e.g. cuda / cpu')
    args = parser.parse_args()

    out_path = args.out or f'weak_edge_confirmation_{args.dataset}.csv'
    ratio = GADBENCH_SCALE_RATIO[args.dataset]
    print(f'dataset={args.dataset} GADBench-scale train_ratio={ratio:.5f}')

    mem.dataset_name = args.dataset
    g = Dataset(args.dataset, homo=True).graph

    write_header = not os.path.exists(out_path)
    f = open(out_path, 'a', newline='')
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
        f.flush()

    for seed in range(args.seeds):
        for cfg_name, lambda_p, oracle_lu in CONFIGS:
            torch.manual_seed(seed)
            run_args = make_args(ratio, lambda_p, oracle_lu, args.edge_epochs, args.epoch, args.edge_folds,
                                  device=args.device)
            t0 = time.time()
            try:
                result = mem.train(g, run_args, return_dict=True)
            except Exception as e:
                print(f'  FAILED cfg={cfg_name} seed={seed}: {e}', flush=True)
                continue
            dt = time.time() - t0
            row = {'dataset': args.dataset, 'config': cfg_name, 'lambda_p': lambda_p,
                   'oracle_lu': oracle_lu, 'train_ratio': ratio, 'run': seed}
            row.update({k: result.get(k) for k in FIELDNAMES if k in result})
            writer.writerow(row)
            f.flush()
            print('  seed={} cfg={:16s} node_auroc={:.4f} node_ap={:.4f} edge_auc={} edge_ap={} ({:.1f}s)'.format(
                seed, cfg_name, result['node_auroc'], result['node_ap'],
                round(result['edge_auc'], 4) if result['edge_auc'] is not None else None,
                round(result['edge_ap'], 4) if result['edge_ap'] is not None else None, dt), flush=True)
    f.close()
    print('Done ->', out_path)


if __name__ == '__main__':
    main()
