"""GADBench semi-supervised protocol for ESO-GAD.

What GADBench's own benchmark.py / BaseDetector do, reduced to what this
code needs: the graph files with their shipped splits, the per-trial seeds,
and the three scores. The numbers in the paper's tables use this protocol:

    - 100 labeled training nodes (20 anomalies) and 100 validation nodes
      (20 anomalies): mask columns 10-19 of each dataset file, one column
      per trial;
    - at most 200 epochs, stopped when the validation AUPRC has not improved
      for more than 50 epochs; test scores taken at the best validation AUPRC;
    - Adam at 0.01, class-weighted cross-entropy;
    - seed 3407 + 10 * trial;
    - mean and (population) standard deviation over the trials.
"""
import os
import random

import dgl
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from dataset import Dataset

DATASETS = ('reddit', 'weibo', 'amazon', 'yelp', 'tfinance', 'tolokers', 'questions')
SEMI_OFFSET = 10      # columns 0-9 are the fully-supervised splits, 10-19 the 100-label ones
NUM_TRIALS = 10


def trial_seed(trial):
    return 3407 + 10 * trial


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def is_symmetric(graph):
    src, dst = graph.edges()
    n = graph.num_nodes()
    return bool(torch.isin(dst * n + src, src * n + dst).all())


def load_graph(name, symmetrize=True, verbose=True):
    """Homogeneous GADBench graph with exactly one self-loop per node.

    symmetrize=True adds the reverse of every edge. GADBench stores Weibo,
    Tolokers and Questions as directed graphs (each edge once), while the
    signed operator and its theory are defined for a symmetric weight matrix;
    the other four files are already symmetric and are left unchanged.
    symmetrize=False keeps the files as shipped (what the WSDM runs used).
    """
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        graph = Dataset(name, homo=True).graph
    graph = dgl.remove_self_loop(graph)
    was_symmetric = is_symmetric(graph)
    if symmetrize and not was_symmetric:
        graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    if verbose:
        print('{}: {} nodes, {} directed edges incl. self-loops ({})'.format(
            name, graph.num_nodes(), graph.num_edges(),
            'symmetric as shipped' if was_symmetric else
            ('symmetrized' if symmetrize else 'directed as shipped')))
    return graph


def split_masks(graph, trial, semi_supervised=True):
    """Boolean train / val / test masks of one shipped split."""
    col = trial + (SEMI_OFFSET if semi_supervised else 0)
    return (graph.ndata['train_masks'][:, col].bool(),
            graph.ndata['val_masks'][:, col].bool(),
            graph.ndata['test_masks'][:, col].bool())


def known_label_mask(graph):
    """Nodes whose label is known: those that appear in a shipped split.
    Only Amazon has nodes outside (its first 3,305 nodes are unlabeled and
    stored with label 0); analyses that use all ground-truth labels must
    restrict to this mask."""
    col = SEMI_OFFSET
    return (graph.ndata['train_masks'][:, col].bool() | graph.ndata['val_masks'][:, col].bool()
            | graph.ndata['test_masks'][:, col].bool())


def random_split(labels, known, rng, n_train=100, n_train_anomalies=20, n_val=100, n_val_anomalies=20):
    """A random draw of a GADBench-style labeled set (not one of the shipped
    splits): n_train training nodes and n_val validation nodes with the given
    numbers of anomalies, the rest of the known-label nodes as test."""
    labels = labels.cpu().numpy()
    known = known.cpu().numpy()
    anomalies = rng.permutation(np.where(known & (labels == 1))[0])
    normals = rng.permutation(np.where(known & (labels == 0))[0])
    a1, a2 = n_train_anomalies, n_train_anomalies + n_val_anomalies
    b1, b2 = n_train - n_train_anomalies, n_train - n_train_anomalies + n_val - n_val_anomalies
    masks = [np.zeros(labels.shape[0], dtype=bool) for _ in range(3)]
    masks[0][anomalies[:a1]] = True
    masks[0][normals[:b1]] = True
    masks[1][anomalies[a1:a2]] = True
    masks[1][normals[b1:b2]] = True
    masks[2][anomalies[a2:]] = True
    masks[2][normals[b2:]] = True
    return tuple(torch.from_numpy(m) for m in masks)


def evaluate(labels, probs):
    """AUROC, AUPRC and Rec@K (K = number of anomalies), as GADBench computes them."""
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    if torch.is_tensor(probs):
        probs = probs.detach().cpu().numpy()
    labels = np.asarray(labels)
    k = int(labels.sum())
    return {'AUROC': float(roc_auc_score(labels, probs)),
            'AUPRC': float(average_precision_score(labels, probs)),
            'RecK': float(labels[probs.argsort()[-k:]].sum() / labels.sum())}
