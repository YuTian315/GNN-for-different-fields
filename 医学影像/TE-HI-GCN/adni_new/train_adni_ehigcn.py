import warnings
import sklearn.metrics as metrics
from torch.autograd import Variable
import numpy as np
import torch
import random
from hinets.model_ehigcn_adni import EHIgcn2 as HiGCN
from sklearn.model_selection import StratifiedKFold


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
    ts_result = []

    for fold in range(10):
        pcc, train, test, new_labels = ten_fold(fold)
        new_labels = np.array(new_labels)
        graph_kenel = get_graph_kernel(fold)

        graph_adj1 = cal_pcc(pcc=pcc, thr=0.1)
        graph_adj2 = cal_pcc(pcc=pcc, thr=0.2)
        graph_adj3 = cal_pcc(pcc=pcc, thr=0.3)
        graph_adj4 = cal_pcc(pcc=pcc, thr=0.4)
        graph_adj5 = cal_pcc(pcc=pcc, thr=0.5)
        graph_adj6 = cal_pcc(pcc=pcc, thr=0.05)
        graph_adj7 = cal_pcc(pcc=pcc, thr=0.15)
        graph_adj8 = cal_pcc(pcc=pcc, thr=0.25)
        graph_adj9 = cal_pcc(pcc=pcc, thr=0.35)
        graph_adj10 = cal_pcc(pcc=pcc, thr=0.45)

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

        node_adj_data = Variable(torch.FloatTensor(graph_kenel), requires_grad=False).to(device)

        train_test = np.append(test, train)

        train_node_adj, train_test_node_adj = get_node_adj(train,  train_test, node_adj_data)

        model = HiGCN(in_dim=5, hidden_dim=5, graph_adj1=graph_adj1, graph_adj2=graph_adj2, graph_adj3=graph_adj3,
                      graph_adj4=graph_adj4, graph_adj5=graph_adj5, graph_adj6=graph_adj6, graph_adj7=graph_adj7,
                      graph_adj8=graph_adj8, graph_adj9=graph_adj9, graph_adj10=graph_adj10, num=fold, thr1='1', thr2='2',
                      thr3='3', thr4='4', thr5='5', thr6='05', thr7='15', thr8='25', thr9='35', thr10='45')

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)

        for epoch in range(100):
            for k, v in model.named_parameters():
                v.requires_grad = True
            model.train()

            batch_nodes = train
            model.zero_grad()

            train_out, cluster_loss = model(batch_nodes, train_node_adj)
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
                torch.save(model.state_dict(), 'ehi/ehigcn_ADNI' + str(fold) + '_0.0' + '.pt')

        model.load_state_dict(torch.load('ehi/ehigcn_ADNI' + str(fold) + '_0.0' + '.pt'))

        test_out, _ = model.forward(train_test, train_test_node_adj)
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

    precs = []
    recalls = []
    accs = []
    f1s = []
    aucs = []
    for cv in range(10):
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




