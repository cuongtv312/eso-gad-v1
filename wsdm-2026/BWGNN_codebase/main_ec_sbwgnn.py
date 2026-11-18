import argparse
import time

import numpy
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, average_precision_score

from dataset import Dataset
from edge_calibration import cross_fitted_edge_calibration
from ec_sbwgnn import EC_SBWGNN
from main import get_best_f1


def _resolve_device(args):
    dev_str = getattr(args, 'device', None) or ('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(dev_str)


def train(g, args, return_dict=False):
    device = _resolve_device(args)
    g = g.to(device)
    features = g.ndata['feature']
    labels = g.ndata['label']
    labels_cpu = labels.detach().cpu()

    index = list(range(len(labels_cpu)))
    if dataset_name == 'amazon':
        index = list(range(3305, len(labels_cpu)))

    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels_cpu[index], stratify=labels_cpu[index],
                                                              train_size=args.train_ratio,
                                                              random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                              test_size=0.67,
                                                              random_state=2, shuffle=True)
    train_mask = torch.zeros([len(labels_cpu)]).bool()
    val_mask = torch.zeros([len(labels_cpu)]).bool()
    test_mask = torch.zeros([len(labels_cpu)]).bool()
    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())

    print('--- cross-fitted edge-heterophily calibration (Algorithm 1, steps 1-3'
          '{}) --- [device={}]'.format(
              ', + weak E_LU supervision (lambda_p={})'.format(args.lambda_p) if args.lambda_p > 0 else '', device))
    time_cf_start = time.time()
    q_calibrated, edge_diag = cross_fitted_edge_calibration(
        g, features, labels, train_mask,
        k=args.edge_folds, hidden_dim=args.edge_hidden, epochs=args.edge_epochs,
        lr=args.edge_lr, weight_decay=args.edge_wd, seed=2,
        max_edges_for_common_neighbors=args.max_edges_cn,
        lambda_p=args.lambda_p, oracle_lu=args.oracle_lu, device=device,
    )
    edge_cf_time = time.time() - time_cf_start
    print('edge calibration ({} of {} folds trained, |E_LL|={}, |E_LU|={}): {}'.format(
        edge_diag.get('num_trained_folds'), args.edge_folds,
        edge_diag.get('num_E_LL'), edge_diag.get('num_E_LU'),
        {k: round(v, 4) for k, v in edge_diag.items()
         if k not in ('num_trained_folds', 'num_E_LL', 'num_E_LU', 'lambda_p', 'oracle_lu')}))
    print('cross-fitting time cost: {:.2f}s'.format(edge_cf_time))

    in_feats = features.shape[1]
    num_classes = 2
    model = EC_SBWGNN(in_feats, args.hid_dim, num_classes, g, q_calibrated,
                       d=args.order, eta_init=args.eta_init).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_f1 = 0.
    final_trec, final_tpre, final_tmf1, final_tauc, final_tap = 0., 0., 0., 0., 0.

    weight = (1 - labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    ce_weight = torch.tensor([1., weight], device=device)
    val_mask_cpu, test_mask_cpu = val_mask.cpu(), test_mask.cpu()
    labels_val_cpu, labels_test_cpu = labels_cpu[val_mask_cpu].numpy(), labels_cpu[test_mask_cpu].numpy()

    time_start = time.time()
    for e in range(args.epoch):
        model.train()
        logits = model(features)
        node_loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=ce_weight)
        gate_kl = model.gate_kl_regularizer()
        loss = node_loss + args.lambda_g * gate_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = logits.softmax(1)
        probs_cpu = probs.detach().cpu()
        f1, thres = get_best_f1(labels_val_cpu, probs_cpu[val_mask_cpu].numpy())
        preds = numpy.zeros(labels_cpu.shape[0])
        preds[probs_cpu[:, 1].numpy() > thres] = 1
        preds_test = preds[test_mask_cpu.numpy()]
        trec = recall_score(labels_test_cpu, preds_test)
        tpre = precision_score(labels_test_cpu, preds_test)
        tmf1 = f1_score(labels_test_cpu, preds_test, average='macro')
        test_probs_pos = probs_cpu[test_mask_cpu][:, 1].numpy()
        tauc = roc_auc_score(labels_test_cpu, test_probs_pos)
        tap = average_precision_score(labels_test_cpu, test_probs_pos)

        if best_f1 < f1:
            best_f1 = f1
            final_trec, final_tpre, final_tmf1, final_tauc, final_tap = trec, tpre, tmf1, tauc, tap

        print('Epoch {}, loss: {:.4f} (node {:.4f}, gate-KL {:.4f}), val mf1: {:.4f}, (best {:.4f})'.format(
            e, loss.item(), node_loss.item(), gate_kl.item(), f1, best_f1))

    node_train_time = time.time() - time_start
    print('time cost: ', node_train_time, 's')
    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f} AP {:.2f}'.format(
        final_trec * 100, final_tpre * 100, final_tmf1 * 100, final_tauc * 100, final_tap * 100))

    with torch.no_grad():
        _, diag = model(features, return_diagnostics=True)
    print('final eta (calibration strength): {:.4f}'.format(diag['eta'].item()))
    print('final r_- (negative-edge fraction): {:.4f}'.format(diag['r_minus'].item()))
    print('final channel gate g: {}'.format([round(v, 4) for v in diag['gate'].tolist()]))
    print('final channel energy ||B_k H^(0)||_F^2: {}'.format([round(v, 2) for v in diag['channel_energy'].tolist()]))

    if not return_dict:
        return final_tmf1, final_tauc

    result = {
        'node_auroc': final_tauc, 'node_ap': final_tap, 'node_macro_f1': final_tmf1,
        'node_recall': final_trec, 'node_precision': final_tpre,
        'edge_auc': edge_diag.get('edge_auc'), 'edge_ap': edge_diag.get('edge_ap'),
        'edge_brier': edge_diag.get('edge_brier'), 'edge_bce': edge_diag.get('edge_bce'),
        'edge_ece': edge_diag.get('edge_ece'),
        'num_E_LL': edge_diag.get('num_E_LL'), 'num_E_LU': edge_diag.get('num_E_LU'),
        'num_E_LL_hetero': edge_diag.get('num_E_LL_hetero'),
        'num_E_LU_anomaly_related': edge_diag.get('num_E_LU_anomaly_related'),
        'pseudo_lu_accuracy': edge_diag.get('pseudo_lu_accuracy'),
        'pseudo_lu_accuracy_anomaly_related': edge_diag.get('pseudo_lu_accuracy_anomaly_related'),
        'num_trained_folds': edge_diag.get('num_trained_folds'),
        'eta': diag['eta'].item(), 'gate': diag['gate'].tolist(), 'r_minus': diag['r_minus'].item(),
        'edge_cf_time_s': edge_cf_time, 'node_train_time_s': node_train_time,
        'lambda_p': args.lambda_p, 'oracle_lu': args.oracle_lu, 'train_ratio': args.train_ratio,
    }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EC-SBWGNN')
    parser.add_argument("--dataset", type=str, default="amazon",
                        help="Dataset for this model (amazon/yelp/tfinance/reddit/weibo/tolokers/questions/tsocial)")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Wavelet order C in Beta/Bernstein bank")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")
    parser.add_argument("--device", type=str, default=None,
                        help="torch device, e.g. cuda / cuda:0 / cpu. Defaults to cuda if available.")

    parser.add_argument("--edge_folds", type=int, default=5, help="K, number of cross-fitting folds")
    parser.add_argument("--edge_hidden", type=int, default=32, help="Hidden dim of the edge predictor q_psi")
    parser.add_argument("--edge_epochs", type=int, default=150, help="Epochs to train each fold's q_psi")
    parser.add_argument("--edge_lr", type=float, default=1e-2, help="Learning rate for q_psi")
    parser.add_argument("--edge_wd", type=float, default=1e-5, help="Weight decay for q_psi")
    parser.add_argument("--max_edges_cn", type=int, default=5_000_000,
                        help="Skip exact common-neighbor structural stats above this many edges")
    parser.add_argument("--eta_init", type=float, default=0.5, help="Initial global calibration strength eta")
    parser.add_argument("--lambda_g", type=float, default=0.01, help="Weight of the gate KL regularizer")

    parser.add_argument("--lambda_p", type=float, default=0.0,
                        help="Weight of the weak E_LU pseudo-supervision term in the edge-calibration loss "
                             "(weak_edge_supervision.tex). 0.0 = original E_LL-only calibrator.")
    parser.add_argument("--oracle_lu", action="store_true",
                        help="Offline-analysis-only: replace E_LU pseudo labels with true hidden edge labels "
                             "to estimate the upper bound of the pseudo-supervision idea.")

    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset
    graph = Dataset(dataset_name, homo=True).graph

    if args.run == 1:
        train(graph, args)
    else:
        final_mf1s, final_aucs = [], []
        for _ in range(args.run):
            mf1, auc = train(graph, args)
            final_mf1s.append(mf1)
            final_aucs.append(auc)
        final_mf1s = numpy.array(final_mf1s)
        final_aucs = numpy.array(final_aucs)
        print('MF1-mean: {:.2f}, MF1-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}'.format(
            100 * numpy.mean(final_mf1s), 100 * numpy.std(final_mf1s),
            100 * numpy.mean(final_aucs), 100 * numpy.std(final_aucs)))
