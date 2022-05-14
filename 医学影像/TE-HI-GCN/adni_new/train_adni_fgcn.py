import warnings
import sklearn.metrics as metrics
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
from torch.nn import init
import numpy as np
import torch
import random
from sklearn.model_selection import StratifiedKFold


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
        feature_dim = int(adj.shape[-1])
        eye = torch.eye(feature_dim).cuda()
        if x is None:
            AXW = torch.tensordot(adj, self.kernel, [[-1], [0]])  # batch_size * num_node * feature_dim
        else:
            XW = torch.tensordot(x, self.kernel, [[-1], [0]])  # batch *  num_node * feature_dim
            AXW = torch.matmul(adj, XW)  # batch *  num_node * feature_dim
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
        # self.reset_weigths()

    # cluster
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

    def loss(self, pred, label):
        return F.cross_entropy(pred, label, reduction='mean')


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpointhi.pt', trace_func=print):
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


def load_time_series():
    t_brains = []
    for i in range(133):
        with open('ADNI/'+ str(i + 1) + '.txt', 'r') as f:
            counts = 0

            tmp_list = []
            for line in f:  # 116
                tmp = np.zeros(116)
                line.strip('\n')
                line = line.split(' ')

                for columns in range(116):
                    tmp[columns] = line[columns]

                tmp_list.append(tmp)
                counts += 1

        tmp_array = np.array(tmp_list, dtype=np.float32)
        time_series = np.transpose(tmp_array)
        t_brains.append(time_series)

    print('t-series-shape:', t_brains[0].shape)

    return t_brains


# load 866 label
def load_label():
    filename_graphs = 'ADNI/label_ADNI.txt'

    graph_labels = []

    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            graph_labels.append(val)

    return graph_labels


def get_corr(data):
    corr_matrix = []
    pcc = np.zeros((len(data), 116, 116))
    for di in range(len(data)):
        corr_mat = np.corrcoef(data[di])
        corr_matrix.append(corr_mat)
        pcc[di] = corr_mat

    print(pcc.shape)

    return pcc


def ten_fold(fold):

    series = load_time_series()
    labels = load_label()
    ts_folder = StratifiedKFold(n_splits=10, random_state=0, shuffle=True).split(series, labels)
    fold_count = 0
    for t_idx, s_idx in ts_folder:

        if fold != fold_count:
            fold_count += 1
            continue
        print('fold_count:', fold_count)
        new_labels = []
        new_series = []
        test = s_idx
        train = t_idx
        for ti in range(len(train)):
            if labels[train[ti]] == 1:
                new_series.append(series[train[ti]][:, :45])
                new_series.append(series[train[ti]][:, 45:90])
                new_series.append(series[train[ti]][:, 90:])

                new_labels.append(1)
                new_labels.append(1)
                new_labels.append(1)
            else:
                new_series.append(series[train[ti]])
                new_labels.append(0)


        random.Random(0).shuffle(train)
        neww = np.array(new_labels)

        print(test, neww[test])

        print(len(new_labels), len(labels))
        print(len(new_series), len(series))

        for testi in range(len(test)):
            new_series.append(series[test[testi]])
            new_labels.append(labels[test[testi]])

        print(len(new_labels), len(labels))
        print(len(new_series), len(series))

        pcc = get_corr(new_series)

        return pcc, train, test, new_labels


def calcRbfKernel(bag1, bag2, gamma):
    '''
    This function calculates an rbf kernel for instances between two bags.
    :param bag1: ndarray [n,d].  A multiple instance bag.
    :param bag2: ndarray [m,d].  A multiple instance bag.
    :param gamma: The normalizing parameter for the radial basis function.
    return: kMat: ndarray [n,m]. The between instances kernel function.
    '''

    n = bag1.shape[0] # the number of instances in bag 1
    m = bag2.shape[0] # the number of instances in bag 2

    # suared euclidean norm
    bag1_i_norm = np.sum(bag1**2, axis=1)
    bag2_i_norm = np.sum(bag2**2, axis=1)
    distMat = np.array(np.tile(bag1_i_norm, [m,1]).transpose() + np.tile(bag2_i_norm, [n,1]) - 2*np.dot(bag1, bag2.transpose()), dtype=np.float64)

    # radial basis function
    kMat = np.exp(-gamma * distMat)

    return kMat


def calcKernelEntry(bag1, bag2, weightMatrix1, weightMatrix2, gamma):
    '''
    This function calculates one kg kernel entry comparing two bags.
    Differently than stated in the publication, in their implementation Zhou et al. normalized by taking the squareroot
    of the sum over the edge coeficcients.
    :param bag1: ndarray [n,d].  A multiple instance bag.
    :param bag2: ndarray [m,d].  A multiple instance bag.
    :param gamma: The normalizing parameter for the radial basis function.
    return: kMat: ndarray [n,m]. The between instances kernel function.
    '''

    n = bag1.shape[0]  # the number of instances in bag 1
    m = bag2.shape[0]  # the number of instances in bag 2

    activeEdgesCount1 = np.sum(weightMatrix1, axis=1)  # number of edges per instance
    # print(activeEdgesCount1.shape)  # 116
    activeEdgesCount2 = np.sum(weightMatrix2, axis=1)  # number of edges per instance

    activeEdgesCoef1 = 1. / (activeEdgesCount1 + 1e-3)    # offset to avoid division by zero if e.g. just one instance in a bag
    activeEdgesCoef2 = 1. / (activeEdgesCount2 + 1e-3)

    k = calcRbfKernel(bag1, bag2, gamma=gamma)

    k = np.tile(activeEdgesCoef1, [m,1]).transpose() * np.tile(activeEdgesCoef2, [n,1]) * k

    k = np.sum(k) / np.sqrt(np.sum(activeEdgesCoef1)) / np.sqrt(np.sum(activeEdgesCoef2))

    return k


def calcDistMatrix(bag, method='gaussian', gamma=1.):
    '''
    This function calculates the inner bag distance matrix. This matrix represents a graph of connected instances
    Differently than stated in the publication, in their implementation Zhou et al. used not the gaussian distance
    but the squared euclidiean distance.
    :param bag: ndarray [n,D]. One multiple instance bag.
    :param method: Norm used for distance calculation.
    :param gamma: Parameter for RBF kernel for gaussian distance.
    :return distMat: ndarray [n,n].  Distance matrix
    '''

    n = bag.shape[0] # the number of instances

    if method == 'sqeuclidean':
        # squared euclidean norm
        bag_i_norm = np.sum(bag**2, axis=1)
        distMat = np.tile(bag_i_norm, [n,1]) + np.tile(bag_i_norm, [n,1]).transpose() -2*np.dot(bag, bag.transpose())
    elif method == 'gaussian':
        bag_i_norm = np.sum(bag**2, axis=1)
        distMat = np.tile(bag_i_norm, [n,1]) + np.tile(bag_i_norm, [n,1]).transpose() -2*np.dot(bag, bag.transpose())
        distMat = 1 - np.exp(-gamma * distMat)

    #distMat = squareform(pdist(bag, 'sqeuclidean'))

    return distMat


def get_graph_kernel(fold):
    pcc, _, _, _ = ten_fold(fold)

    len_data = pcc.shape[0]

    graph_kernel = np.zeros((len_data, len_data))
    for i in range(len_data):
        for j in range(i + 1, len_data):
            we1 = calcDistMatrix(pcc[i])
            we2 = calcDistMatrix(pcc[j])
            mat = calcKernelEntry(pcc[i], pcc[j], we1, we2, 0.5)
            graph_kernel[i][j] = mat
            graph_kernel[j][i] = mat

    print('mat:', graph_kernel)

    where_are_nan = np.isnan(graph_kernel)
    where_are_inf = np.isinf(graph_kernel)
    for i in range(0, graph_kernel.shape[0]):
        for j in range(0, graph_kernel.shape[1]):
            if where_are_nan[i][j]:
                graph_kernel[i][j] = 0
            if where_are_inf[i][j]:
                graph_kernel[i][j] = 0.8

    return graph_kernel


def cal_pcc(pcc, thr):
    corr_matrix = []
    print(pcc[0])
    len_pcc = pcc.shape[0]
    for bb in range(len_pcc):
        for i in range(116):
            pcc[bb][i][i] = 0
        corr_mat = np.arctanh(pcc[bb])
        corr_matrix.append(corr_mat)
    pcc_array = np.array(corr_matrix)

    where_are_nan = np.isnan(pcc_array)
    where_are_inf = np.isinf(pcc_array)
    for bb in range(0, len_pcc):
        for i in range(0, 116):
            for j in range(0, 116):
                if where_are_nan[bb][i][j]:
                    # print(bb)
                    pcc_array[bb][i][j] = 0
                if where_are_inf[bb][i][j]:
                    pcc_array[bb][i][j] = 0.8

    for bb in range(len_pcc):
        for i in range(pcc_array.shape[1]):
            for j in range(i + 1, pcc_array.shape[1]):
                if np.abs(pcc_array[bb][i][j]) >= thr:
                    continue
                else:
                    pcc_array[bb][i][j] = 0
                    pcc_array[bb][j][i] = 0
    print('pcc_array::::', pcc_array[0])

    corr_p = np.maximum(pcc_array, 0)
    corr_n = 0 - np.minimum(pcc_array, 0)
    pcc_array = [corr_p, corr_n]
    pcc_array = np.array(pcc_array)
    pcc_array = np.transpose(pcc_array, (1, 0, 2, 3))

    return pcc_array


def get_node_adj(train_nodes, train_test_nodes, node_adj):
    train_adj = torch.zeros(len(train_nodes), len(train_nodes)).to(device)
    for i in range(len(train_nodes)):
        for j in range(i+1, len(train_nodes)):
            train_adj[i][j] = node_adj[train_nodes[i]][train_nodes[j]]
            train_adj[j][i] = train_adj[i][j]

    train_test_adj = torch.zeros(len(train_test_nodes), len(train_test_nodes)).to(device)
    for i in range(len(train_test_nodes)):
        for j in range(len(train_test_nodes)-len(train_nodes), len(train_test_nodes)):
            train_test_adj[i][j] = node_adj[train_test_nodes[i]][train_test_nodes[j]]
            train_test_adj[j][i] = train_test_adj[i][j]

    return train_adj, train_test_adj


def train():
    thr = 0.5
    ts_result = []
    for fold in range(10):
        pcc, train, test, new_labels = ten_fold(fold)
        new_labels = np.array(new_labels)
        graph_kenel = get_graph_kernel(fold)
        graph_adj = cal_pcc(pcc, thr)

        graph_adj = Variable(torch.FloatTensor(graph_adj), requires_grad=False).to(device)
        node_adj_data = Variable(torch.FloatTensor(graph_kenel), requires_grad=False).to(device)

        train_test = np.append(test, train)

        train_node_adj, train_test_node_adj = get_node_adj(train,  train_test, node_adj_data)
        model = model_gnn(in_dim=5, hidden_dim=5, out_dim=2)
        # ts += 1
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)
        stop_epoch = 0

        early_stopping = EarlyStopping(patience=500, verbose=True)

        for epoch in range(500):
            for k, v in model.named_parameters():
                v.requires_grad = True
            model.train()

            batch_nodes = train
            start_time = time.time()
            model.zero_grad()

            train_out, cluster_loss = model(graph_adj[batch_nodes])
            print(new_labels[np.array(batch_nodes)])
            seg_loss = model.loss(train_out, torch.LongTensor(new_labels[np.array(batch_nodes)]).to(device))

            # # l2正则化
            regularization_loss = 0
            for param in model.parameters():
                regularization_loss += torch.sum(abs(param))
            print(regularization_loss)

            loss = seg_loss + cluster_loss + 0.0001 * regularization_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

            optimizer.step()
            CosineLR.step()

            train_result = {
                'loss': seg_loss.item(),
                'prec': metrics.precision_score(new_labels[batch_nodes], train_out.cpu().data.numpy().argmax(axis=1)),
                'recall': metrics.recall_score(new_labels[batch_nodes], train_out.cpu().data.numpy().argmax(axis=1)),
                'acc': metrics.accuracy_score(new_labels[batch_nodes], train_out.cpu().data.numpy().argmax(axis=1)),
                'F1': metrics.f1_score(new_labels[batch_nodes], train_out.data.cpu().numpy().argmax(axis=1)),
                'auc': metrics.roc_auc_score(new_labels[batch_nodes], train_out.cpu().data.numpy().argmax(axis=1),
                                             sample_weight=None)
            }
            print('train_result:', fold, epoch, train_result)

            with torch.no_grad():
                torch.save(model.state_dict(), 'fgcn/fgcn_ADNI' + str(fold) + '_' + str(thr) + '.pt')

        model.load_state_dict(torch.load('fgcn/fgcn_ADNI' + str(fold) + '_' + str(thr) + '.pt'))

        test_out, _ = model.forward(graph_adj[test])
        test_output = test_out.cpu().data.numpy().argmax(axis=1)[:len(test)]
        precs.append(metrics.precision_score(new_labels[test], test_output))
        recalls.append(metrics.recall_score(new_labels[test], test_output))
        accs.append(metrics.accuracy_score(new_labels[test], test_output))
        f1s.append(metrics.f1_score(new_labels[test], test_output))
        aucs.append(metrics.roc_auc_score(new_labels[test], test_output, sample_weight=None))

        test_result = {'prec': metrics.precision_score(new_labels[test], test_output),
                       'recall': metrics.recall_score(new_labels[test], test_output),
                       'acc': metrics.accuracy_score(new_labels[test], test_output),
                       'F1': metrics.f1_score(new_labels[test], test_output),
                       'auc': metrics.roc_auc_score(new_labels[test], test_output, sample_weight=None)
                       }

        ts_result.append(test_result)
        print('test', test_result)
        print('test predict', test_output)
        del test_result

    print(ts_result)

    return


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Device: ', device)

    precs =[]
    recalls =[]
    accs =[]
    f1s =[]
    aucs =[]
    for cv in range(1):

        train()

        print('cv', cv, precs, recalls, accs, f1s, aucs)

    prec = np.mean(precs)
    prec_std = np.var(precs)

    recall = np.mean(recalls)
    recall_std = np.var(recalls)

    acc = np.mean(accs)
    acc_std = np.var(accs)

    f1 = np.mean(f1s)
    f1_std = np.var(f1s)

    auc = np.mean(aucs)
    auc_std = np.var(aucs)

    print(prec, prec_std)
    print(recall, recall_std)
    print(acc, acc_std)
    print(f1, f1_std)
    print(auc, auc_std)




