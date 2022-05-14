import pickle
import warnings
import sklearn.metrics as metrics
from torch.autograd import Variable
import time
from scipy.io import loadmat
from collections import defaultdict
import numpy as np
import torch
from hinets.model_ehigcn import EHIgcn2 as HiGCN
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='tehi.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif score >= self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# load label
def load_label(name):

    label_path = 'data/correlation&label/871_label_' + name + '.txt'
    graph_labels = []

    with open(label_path) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            graph_labels.append(val)
    graph_labels = np.array(graph_labels)
    # print(graph_labels)

    return graph_labels


def load_correlation(name):

    data_path = 'data/correlation&label/pcc_correlation_871_'
    data_dict = loadmat(data_path + name + '_.mat')
    data_array = data_dict['connectivity']

    return data_array


def cal_pcc(thr, pcc):
    len_pcc = pcc.shape[0]
    len_node = pcc.shape[1]
    corr_matrix = []
    for bb in range(len_pcc):
        for i in range(len_node):
            pcc[bb][i][i] = 0
        corr_mat = np.arctanh(pcc[bb])
        corr_matrix.append(corr_mat)
    pcc_array = np.array(corr_matrix)

    where_are_nan = np.isnan(pcc_array)
    where_are_inf = np.isinf(pcc_array)
    for bb in range(0, len_pcc):
        for i in range(0, len_node):
            for j in range(0, len_node):
                if where_are_nan[bb][i][j]:
                    pcc_array[bb][i][j] = 0
                if where_are_inf[bb][i][j]:
                    pcc_array[bb][i][j] = 0.8

    for bb in range(len_pcc):
        for i in range(len_node):
            for j in range(i+1, len_node):
                if np.abs(pcc_array[bb][i][j]) >= thr:
                    continue
                else:
                    pcc_array[bb][i][j] = 0
                    pcc_array[bb][j][i] = 0
    print('pcc_array::::', pcc_array.shape)

    corr_p = np.maximum(pcc_array, 0)
    corr_n = 0 - np.minimum(pcc_array, 0)
    pcc_array = [corr_p, corr_n]
    pcc_array = np.array(pcc_array)
    pcc_array = np.transpose(pcc_array, (1, 0, 2, 3))

    return pcc_array


def load_graph_kernel(len_pcc, name):
    final_graph = np.ones((len_pcc, len_pcc))
    with open('graph_kernel/graph_kernel_' + name + '.txt', 'r') as f:
        count = 0
        for line in f:
            line.strip('\n')
            line = line.split()
            for columns in range(len_pcc):
                final_graph[count][columns] = float(line[columns])
            count += 1

    adj = np.zeros((len_pcc, len_pcc))
    for i in range(len_pcc):
        for j in range(i + 1, len_pcc):
            adj[i][j] = 1
            adj[j][i] = 1

    print('final graph:', final_graph.shape)
    final_graph = np.abs(final_graph)

    return final_graph


def get_node_adj(train_nodes, train_val_nodes, train_test_nodes, node_adj):
    train_adj = torch.zeros(len(train_nodes), len(train_nodes)).to(device)
    for i in range(len(train_nodes)):
        for j in range(i + 1, len(train_nodes)):
            train_adj[i][j] = node_adj[train_nodes[i]][train_nodes[j]]
            train_adj[j][i] = train_adj[i][j]

    train_val_adj = torch.zeros(len(train_val_nodes), len(train_val_nodes)).to(device)
    print(len(train_val_nodes) - len(train_nodes))
    for i in range(len(train_val_nodes)):
        for j in range(len(train_val_nodes) - len(train_nodes), len(train_val_nodes)):
            train_val_adj[i][j] = node_adj[train_val_nodes[i]][train_val_nodes[j]]
            train_val_adj[j][i] = train_val_adj[i][j]

    train_test_adj = torch.zeros(len(train_test_nodes), len(train_test_nodes)).to(device)
    for i in range(len(train_test_nodes)):
        for j in range(len(train_test_nodes) - len(train_nodes), len(train_test_nodes)):
            train_test_adj[i][j] = node_adj[train_test_nodes[i]][train_test_nodes[j]]
            train_test_adj[j][i] = train_test_adj[i][j]

    return train_adj, train_val_adj, train_test_adj


def train(name, brain_region_num, device='cuda'):
    ts_folder = StratifiedKFold(n_splits=10, random_state=0, shuffle=True).split(graph_adj1, labels)
    ts_result = []
    ts = 0

    for t_idx, s_idx in ts_folder:
        test = s_idx
        train_val = t_idx

        best_acc = 0
        train_val_graph_adj = graph_adj1[t_idx]
        train_val_labels = labels[t_idx]

        tv_folder = StratifiedKFold(n_splits=10, random_state=0, shuffle=True).split(train_val_graph_adj,
                                                                                     train_val_labels)
        for tt_idx, vv_idx in tv_folder:
            val = train_val[vv_idx]
            train = train_val[tt_idx]
        train_test = np.append(test, train)
        train_val = np.append(val, train)

        train_node_adj, train_val_node_adj, train_test_node_adj = get_node_adj(train, train_val, train_test,
                                                                               node_adj_data)
        model = HiGCN(in_dim=5, hidden_dim=5, graph_adj1=graph_adj1, graph_adj2=graph_adj2, graph_adj3=graph_adj3,
                      graph_adj4=graph_adj4, graph_adj5=graph_adj5, graph_adj6=graph_adj6, graph_adj7=graph_adj7,
                      graph_adj8=graph_adj8, graph_adj9=graph_adj9, graph_adj10=graph_adj10, num=ts, thr1='1', thr2='2',
                      thr3='3', thr4='4', thr5='5', thr6='05', thr7='15', thr8='25', thr9='35', thr10='45',
                      brain_region_num=brain_region_num, name=name)

        ts += 1
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)

        early_stopping = EarlyStopping(patience=500, verbose=True)

        for epoch in range(2000):
            for k, v in model.named_parameters():
                v.requires_grad = True
            model.train()

            batch_nodes = train
            model.zero_grad()

            train_out, cluster_loss = model(batch_nodes, train_node_adj)
            seg_loss = model.loss(train_out, torch.LongTensor(labels[np.array(batch_nodes)]).to(device))

            loss = seg_loss + cluster_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            CosineLR.step()

            train_result = {
                'loss': seg_loss.item(),
                'prec': metrics.precision_score(labels[batch_nodes], train_out.cpu().data.numpy().argmax(axis=1)),
                'recall': metrics.recall_score(labels[batch_nodes], train_out.cpu().data.numpy().argmax(axis=1)),
                'acc': metrics.accuracy_score(labels[batch_nodes], train_out.cpu().data.numpy().argmax(axis=1)),
                'F1': metrics.f1_score(labels[batch_nodes], train_out.data.cpu().numpy().argmax(axis=1)),
                'auc': metrics.roc_auc_score(labels[batch_nodes], train_out.cpu().data.numpy().argmax(axis=1), sample_weight=None)
            }
            print('train_result:', train_result)

            with torch.no_grad():
                val_out, _ = model.forward(train_val, train_val_node_adj)
                val_output = val_out.cpu().data.numpy().argmax(axis=1)[:len(val)]
                val_loss = model.loss(val_out[:len(val)], torch.LongTensor(labels[np.array(val)]).to(device))

                result = {
                    'prec': metrics.precision_score(labels[val], val_output, average='micro'),
                    'recall': metrics.recall_score(labels[val], val_output, average='micro'),
                    'acc': metrics.accuracy_score(labels[val], val_output),
                    'F1': metrics.f1_score(labels[val], val_output, average="macro"),
                    'auc': metrics.roc_auc_score(labels[val], val_output, average='macro', sample_weight=None)
                }

                if result['acc'] >= best_acc:
                    best_result = result
                    best_epoch = epoch
                    best_acc = result['acc']

                    torch.save(model.state_dict(), 'tehi.pt')
                    torch.save(model.state_dict(), 'model/ehigcn/ASD_' + name + '/ehigcn_' + str(ts) + '.pt')
                print('te epoch', ts, epoch)
                # print('Val epoch', best_epoch, best_result)
                # print('val', result)
                early_stopping(0 - result['acc'], model)
                print('eeeee:', 0 - result['acc'])
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        model.load_state_dict(torch.load('model/ehigcn/ASD_' + name + '/ehigcn_' + str(ts) + '.pt'))

        test_out, _ = model.forward(train_test, train_test_node_adj)
        test_output = test_out.cpu().data.numpy().argmax(axis=1)[:len(test)]

        test_result = {'prec': metrics.precision_score(labels[test], test_output),
                       'recall': metrics.recall_score(labels[test], test_output),
                       'acc': metrics.accuracy_score(labels[test], test_output),
                       'F1': metrics.f1_score(labels[test], test_output),
                       'auc': metrics.roc_auc_score(labels[test], test_output, sample_weight=None)
                       }

        ts_result.append(test_result)
        print('test', test_result)
        del test_result

    print(ts_result)

    return


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Device: ', device)

    name = 'aal'
    num_feats = 50
    num_class = 2

    pcc = load_correlation(name)
    labels = load_label(name)
    brain_region_num= pcc.shape[1]

    print('pcc shape:', pcc.shape)

    graph_adj1 = cal_pcc(thr=0.1, pcc=pcc)
    graph_adj2 = cal_pcc(thr=0.2, pcc=pcc)
    graph_adj3 = cal_pcc(thr=0.3, pcc=pcc)
    graph_adj4 = cal_pcc(thr=0.4, pcc=pcc)
    graph_adj5 = cal_pcc(thr=0.5, pcc=pcc)
    graph_adj6 = cal_pcc(thr=0.05, pcc=pcc)
    graph_adj7 = cal_pcc(thr=0.15, pcc=pcc)
    graph_adj8 = cal_pcc(thr=0.25, pcc=pcc)
    graph_adj9 = cal_pcc(thr=0.35, pcc=pcc)
    graph_adj10 = cal_pcc(thr=0.45, pcc=pcc)

    node_adj_data = load_graph_kernel(pcc.shape[0], name)

    graph_adj1 = Variable(torch.FloatTensor(graph_adj1), requires_grad=False).to(device)
    graph_adj2 = Variable(torch.FloatTensor(graph_adj2), requires_grad=False).to(device)
    graph_adj3 = Variable(torch.FloatTensor(graph_adj3), requires_grad=False).to(device)
    graph_adj4 = Variable(torch.FloatTensor(graph_adj4), requires_grad=False).to(device)
    graph_adj5 = Variable(torch.FloatTensor(graph_adj5), requires_grad=False).to(device)
    graph_adj6 = Variable(torch.FloatTensor(graph_adj6), requires_grad=False).to(device)
    graph_adj7 = Variable(torch.FloatTensor(graph_adj7), requires_grad=False).to(device)
    graph_adj8 = Variable(torch.FloatTensor(graph_adj8), requires_grad=False).to(device)
    graph_adj9 = Variable(torch.FloatTensor(graph_adj9), requires_grad=False).to(device)
    graph_adj10 = Variable(torch.FloatTensor(graph_adj10), requires_grad=False).to(device)

    node_adj_data = Variable(torch.FloatTensor(node_adj_data), requires_grad=False).to(device)

    train(name, brain_region_num)
