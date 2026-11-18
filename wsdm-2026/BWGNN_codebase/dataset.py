import os

from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

# GADBench's pre-built homogeneous graph files, each shipped with self-loops
# and the 20 split columns (train_masks/val_masks/test_masks; columns 10-19
# are the semi-supervised 100-label splits). Override with ESOGAD_DATA_DIR.
GADBENCH_DATASETS_DIR = os.environ.get(
    'ESOGAD_DATA_DIR', os.path.normpath(os.path.join(_HERE, '..', 'GADBench-master', 'datasets')))
GADBENCH_DATASETS = ('reddit', 'weibo', 'tolokers', 'questions', 'amazon', 'yelp', 'tfinance', 'elliptic')

# BWGNN's own files (one-hot T-Finance labels, T-Social), used only by the
# original BWGNN options below.
BWGNN_DATASETS_DIR = os.path.join(_HERE, 'dataset')


class Dataset:
    def __init__(self, name='tfinance', homo=True, anomaly_alpha=None, anomaly_std=None, source='gadbench'):
        """source='gadbench' (default) reads the homogeneous graph from
        GADBENCH_DATASETS_DIR whenever GADBench ships one. source='dgl' keeps
        BWGNN's original loaders (DGL's multi-relation Amazon/YelpChi, BWGNN's
        T-Finance file); homo=False and the T-Finance anomaly_* options
        always use them."""
        self.name = name
        graph = None
        use_gadbench = (source == 'gadbench' and homo and name in GADBENCH_DATASETS
                        and not anomaly_alpha and not anomaly_std)
        if use_gadbench:
            graph, label_dict = load_graphs(os.path.join(GADBENCH_DATASETS_DIR, name))
            graph = graph[0]

        elif name == 'tfinance':
            graph, label_dict = load_graphs(os.path.join(BWGNN_DATASETS_DIR, 'tfinance'))
            graph = graph[0]
            graph.ndata['label'] = graph.ndata['label'].argmax(1)

            if anomaly_std:
                graph, label_dict = load_graphs(os.path.join(BWGNN_DATASETS_DIR, 'tfinance'))
                graph = graph[0]
                feat = graph.ndata['feature'].numpy()
                anomaly_id = graph.ndata['label'][:,1].nonzero().squeeze(1)
                feat = (feat-np.average(feat,0)) / np.std(feat,0)
                feat[anomaly_id] = anomaly_std * feat[anomaly_id]
                graph.ndata['feature'] = torch.tensor(feat)
                graph.ndata['label'] = graph.ndata['label'].argmax(1)

            if anomaly_alpha:
                graph, label_dict = load_graphs(os.path.join(BWGNN_DATASETS_DIR, 'tfinance'))
                graph = graph[0]
                feat = graph.ndata['feature'].numpy()
                anomaly_id = list(graph.ndata['label'][:, 1].nonzero().squeeze(1))
                normal_id = list(graph.ndata['label'][:, 0].nonzero().squeeze(1))
                label = graph.ndata['label'].argmax(1)
                diff = anomaly_alpha * len(label) - len(anomaly_id)
                import random
                new_id = random.sample(normal_id, int(diff))
                # new_id = random.sample(anomaly_id, int(diff))
                for idx in new_id:
                    aid = random.choice(anomaly_id)
                    # aid = random.choice(normal_id)
                    feat[idx] = feat[aid]
                    label[idx] = 1  # 0

        elif name == 'tsocial':
            graph, label_dict = load_graphs(os.path.join(BWGNN_DATASETS_DIR, 'tsocial'))
            graph = graph[0]

        elif name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        else:
            print('no such dataset')
            exit(1)

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        print(graph)

        self.graph = graph
