from torch.autograd.variable import Variable
import pickle
import warnings
import sklearn.metrics as metrics
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import math
from torch.nn import init
from scipy.io import loadmat
import pandas as pd
import os
import tqdm
import numpy as np
import torch
import random
from Graph_sample import datasets
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint2.pt', trace_func=print):
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
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, neg_penalty):
        super(GCN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.neg_penalty = neg_penalty
        self.kernel = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        init.xavier_uniform_(self.kernel)
        self.c = 0.85
        self.losses = []

    def forward(self, x, adj):
        # print(adj.shape)
        feature_dim = int(adj.shape[-1])
        eye = torch.eye(feature_dim).cuda()
        if x is None:
            AXW = torch.tensordot(adj, self.kernel, [[-1], [0]])  # batch_size * num_node * feature_dim
        else:
            XW = torch.tensordot(x, self.kernel, [[-1], [0]])  # batch *  num_node * feature_dim
            AXW = torch.matmul(adj, XW)  # batch *  num_node * feature_dim
        # print(AXW.shape)
        I_cAXW = eye+self.c * AXW
        y_relu = torch.nn.functional.relu(I_cAXW)
        temp = torch.mean(input=y_relu, dim=-2, keepdim=True) + 1e-6
        col_mean = temp.repeat([1, feature_dim, 1])
        y_norm = torch.divide(y_relu, col_mean)
        output = torch.nn.functional.softplus(y_norm)
        if self.neg_penalty != 0:
            neg_loss = torch.multiply(torch.tensor(self.neg_penalty),
                                      torch.sum(torch.nn.functional.relu(1e-6 - self.kernel)))
            self.losses.append(neg_loss)
        return output


class model_gnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(model_gnn, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.gcn1_p = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_p = GCN(hidden_dim, hidden_dim, 0.2)
        self.gcn3_p = GCN(hidden_dim, hidden_dim, 0.2)
        self.gcn1_n = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_n = GCN(hidden_dim, hidden_dim, 0.2)
        self.gcn3_n = GCN(hidden_dim, hidden_dim, 0.2)
        self.kernel_p = nn.Parameter(torch.FloatTensor(116, in_dim))  #
        self.kernel_n = nn.Parameter(torch.FloatTensor(116, in_dim))
        init.xavier_uniform_(self.kernel_p)
        init.xavier_uniform_(self.kernel_n)
        self.lin1 = nn.Linear(2 * in_dim*hidden_dim, 16)
        self.lin2 = nn.Linear(16, self.out_dim)
        self.losses = []

    def dim_reduce(self, adj_matrix, num_reduce,
                   ortho_penalty, variance_penalty, neg_penalty, kernel):
        kernel_p = torch.nn.functional.relu(kernel)
        batch_size = int(adj_matrix.shape[0])
        AF = torch.tensordot(adj_matrix, kernel_p, [[-1], [0]])
        reduced_adj_matrix = torch.transpose(
            torch.tensordot(kernel_p, AF, [[0], [1]]),  # num_reduce*batch*num_reduce
            1, 0)  # num_reduce*batch*num_reduce*num_reduce
        kernel_p_tran = kernel_p.transpose(-1, -2)  # num_reduce * column_dim
        gram_matrix = torch.matmul(kernel_p_tran, kernel_p)
        diag_elements = gram_matrix.diag()

        if ortho_penalty != 0:
            ortho_loss_matrix = torch.square(gram_matrix - torch.diag(diag_elements))
            ortho_loss = torch.multiply(torch.tensor(ortho_penalty), torch.sum(ortho_loss_matrix))
            self.losses.append(ortho_loss)

        if variance_penalty != 0:
            variance = diag_elements.var()
            variance_loss = torch.multiply(torch.tensor(variance_penalty), variance)
            self.losses.append(variance_loss)

        if neg_penalty != 0:
            neg_loss = torch.multiply(torch.tensor(neg_penalty),
                                      torch.sum(torch.nn.functional.relu(torch.tensor(1e-6) - kernel)))
            self.losses.append(neg_loss)
        self.losses.append(0.05 * torch.sum(torch.abs(kernel_p)))
        return reduced_adj_matrix

    def reset_weigths(self):
        """reset weights
            """
        # stdv = 1.0 / math.sqrt(116)
        for weight in self.parameters():
            init.xavier_uniform_(weight)
            # init.uniform_(weight, -stdv, stdv)

    def forward(self, A, flag=0):
        A = torch.transpose(A, 1, 0)
        s_feature_p = A[0]
        s_feature_n = A[1]
        p_reduce = self.dim_reduce(s_feature_p, self.in_dim, 0.2, 0.3, 0.1, self.kernel_p)
        p_conv1 = self.gcn1_p(None, p_reduce)
        p_conv2 = self.gcn2_p(p_conv1, p_reduce)
        p_conv3 = self.gcn3_p(p_conv2, p_reduce)
        n_reduce = self.dim_reduce(s_feature_n, self.in_dim, 0.2, 0.5, 0.1, self.kernel_n)
        n_conv1 = self.gcn1_n(None, n_reduce)
        n_conv2 = self.gcn2_n(n_conv1, n_reduce)
        n_conv3 = self.gcn3_n(n_conv2, n_reduce)

        conv_concat = torch.cat([p_conv3, n_conv3], -1).reshape([-1, self.in_dim*self.hidden_dim*2])
        output = self.lin2(self.lin1(conv_concat))
        output = torch.softmax(output, dim=1)
        loss = torch.sum(torch.tensor(self.losses))
        self.losses.clear()
        return output, loss


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

    return graph_labels


def load_correlation(name):

    data_path = 'data/correlation&label/pcc_correlation_871_'
    data_dict = loadmat(data_path + name + '_.mat')
    data_array = data_dict['connectivity']

    len_data = data_array.shape[0]

    return data_array, len_data


def cal_pcc(thr, pcc, len_pcc):
    len_node = pcc.shape[1]
    corr_matrix = []
    print(pcc[0])
    for bb in range(len_pcc):
        for i in range(len_node):
            pcc[bb][i][i] = 0
        corr_mat = np.arctanh(pcc[bb])
        corr_matrix.append(corr_mat)
    pcc_array = np.array(corr_matrix)   # 871 116 116

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


def cross_val(A, labels):

    kf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    zip_list = list(zip(A, labels))
    random.Random(0).shuffle(zip_list)
    A, labels = zip(*zip_list)
    test_data_loader = []
    train_data_loader = []
    valid_data_loader = []
    A = np.array(A)
    labels = np.array(labels)
    for kk, (train_index, test_index) in enumerate(kf.split(A, labels)):
        train_val_adj, test_adj = A[train_index], A[test_index]
        train_val_labels, test_labels = labels[train_index], labels[test_index]
        tv_folder = StratifiedKFold(n_splits=10, random_state=0, shuffle=True).split(train_val_adj, train_val_labels)
        for t_idx, v_idx in tv_folder:
            train_adj, train_labels = train_val_adj[t_idx], train_val_labels[t_idx]
            val_adj, val_labels = train_val_adj[v_idx], train_val_labels[v_idx]
        dataset_sampler = datasets(test_adj, test_labels)
        test_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=32,
            shuffle=False,
            num_workers=0)
        test_data_loader.append(test_dataset_loader)
        dataset_sampler = datasets(train_adj, train_labels)
        train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=32,
            shuffle=True,
            num_workers=0)
        train_data_loader.append(train_dataset_loader)
        dataset_sampler = datasets(val_adj, val_labels)
        val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=32,
            shuffle=False,
            num_workers=0)
        valid_data_loader.append(val_dataset_loader)

    return train_data_loader, valid_data_loader, test_data_loader


def evaluate(dataset, model, name='Validation', max_num_examples=None, device='cpu'):
    model.eval()
    avg_loss = 0.0
    preds = []
    labels = []

    with torch.no_grad():
        for batch_idx, data in enumerate(dataset):
            adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
            label = Variable(data['label'].long()).to(device)
            labels.append(data['label'].long().numpy())
            ypred, loss = model(adj)
            _, indices = torch.max(ypred, 1)
            preds.append(indices.cpu().data.numpy())

            if max_num_examples is not None:
                if (batch_idx + 1) * 32 > max_num_examples:
                    break
    avg_loss /= batch_idx + 1

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    global xx
    global yy
    from sklearn.metrics import confusion_matrix
    auc = metrics.roc_auc_score(labels, preds, sample_weight=None)
    result = {'prec': metrics.precision_score(labels, preds),
              'recall': metrics.recall_score(labels, preds),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds),
              'auc': auc,
              'matrix': confusion_matrix(labels, preds)}
    xx = preds
    yy = labels

    return avg_loss, result


def train(dataset, model, val_dataset=None, test_dataset=None, device='cpu'):

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005,
                                  weight_decay=0.00001)
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.00001)

    for name in model.state_dict():
        print(name)
    iter = 0
    best_val_acc = 0.0

    early_stopping = EarlyStopping(patience=50, verbose=True)

    for epoch in range(300):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        print(epoch)
        for batch_idx, data in enumerate(dataset):
            for k, v in model.named_parameters():
                v.requires_grad = True
            time1 = time.time()
            model.zero_grad()
            adj = Variable(data['adj'].to(torch.float32), requires_grad=False).to(device)
            label = Variable(data['label'].long()).to(device)
            pred, losses = model(adj)

            loss = F.cross_entropy(pred, label, size_average=True)
            loss += losses
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)

            optimizer.step()
            CosineLR.step()
            iter += 1
            avg_loss += loss
        avg_loss /= batch_idx + 1
        print(avg_loss)
        eval_time = time.time()
        if val_dataset is not None:
            val_loss, val_result = evaluate(val_dataset, model, name='Validation', device=device)
            # print('val:', val_result)
            if val_result['acc'] >= best_val_acc:
                torch.save(model.state_dict(), 'checkpoint2.pt')
                best_val_epoch = epoch
                print('save pth.......')
                best_val_acc = val_result['acc']
            print('best::::', best_val_epoch, best_val_acc)

        early_stopping(0-val_result['acc'], model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    model.load_state_dict(torch.load('checkpoint2.pt'))
    return model


def main():
    name = 'aal'
    thr = 0.45

    labels = load_label(name)
    pcc, _ = load_correlation(name)

    adj = cal_pcc(thr, pcc=pcc, len_pcc=871)
    print('finished')
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print('Device: ', device)
    train_data_loaders, valid_data_loaders, test_data_loaders = cross_val(adj, labels)
    result = []
    for i in range(len(train_data_loaders)):
        model = model_gnn(5, 5, 2)
        model.to(device)
        print('model:', model)
        model = train(train_data_loaders[i], model, val_dataset=valid_data_loaders[i],
                                                        test_dataset=test_data_loaders[i], device='cuda')
        dir = 'model/fgcn/ASD_' + name + '/fgcn' + str(i) + '_' + str(thr) + '.pth'
        torch.save(model.state_dict(), dir)
        _, test_result = evaluate(test_data_loaders[i], model, name='Test', device='cuda')
        print('test', test_result)
        result.append(test_result)
        del model
        del test_result
    print(result)


if __name__ == "__main__":
    main()
