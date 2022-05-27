# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import time
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from scipy import sparse
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from scipy.spatial import distance
from sklearn.linear_model import RidgeClassifier
import sklearn.metrics
import scipy.io as sio
from sklearn import svm
import ABIDEParser as Reader
import train_GCN as Train
# import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.manifold.t_sne import TSNE
from sklearn.metrics import confusion_matrix
# Prepares the training/test data for each cross validation fold and trains the GCN
from sklearn import metrics
import matplotlib.pylab as plt
def train_fold_0(train_ind, test_ind, val_ind, graph_feat, features, y, y_data, params, subject_IDs):
    print(len(train_ind))
    print("----------------------------------------ceshijiyangben-----------------------------")
    print(test_ind)

    # selection of a subset of data if running experiments with a subset of the training set
    labeled_ind = Reader.site_percentage(train_ind, params['num_training'], subject_IDs)
    # x_data = Reader.feature_selection(features, y, labeled_ind, params['num_features'])
    x_data = Reader.feature_mrmr(features, y, labeled_ind, params['num_features'])
    # x_data = features
    fold_size = len(test_ind)

    # # Calculate all pairwise distances
    # distv = distance.pdist(x_data, metric='correlation')
    # # Convert to a square symmetric distance matrix
    # dist = distance.squareform(distv)
    # sigma = np.mean(dist)
    # # Get affinity from similarity matrix
    # sparse_graph_rb = np.exp(- dist ** 2 / (2 * sigma ** 2))
    # final_graph = graph_feat * sparse_graph_rb

    num_nodes = len(subject_IDs)
    sparse_graph = np.zeros((num_nodes, num_nodes))
    for xx in range(num_nodes):
        for yy in range(num_nodes):
            aa = np.array(x_data[xx] + x_data[yy])
            bb = (x_data[xx] - x_data[yy])
            sparse_graph[xx][yy] = np.dot(aa, bb)
            # sparse_graph[xx][yy] = np.fabs(sparse_graph[xx][yy])
    sparse_graph = np.fabs(sparse_graph)
    # amin = sparse_graph.min()
    # amax = sparse_graph.max()
    # sparse_graph_guiyi = (sparse_graph - amin) / (amax - amin)
    final_graph = graph_feat * sparse_graph#*sparse_graph_rb



    # features = np.around(features, decimals=5)
    # final_graph = np.around(final_graph, decimals=5)

    # np.savetxt("final_graph.csv", final_graph, delimiter=',')

    #
    # # Linear classifier
    clf = RidgeClassifier(alpha=0.1, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='svd', random_state=None)
    clf.fit(x_data[train_ind, :], y[train_ind].ravel())
    # Compute the accuracy
    lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
    # Compute the AUC
    pred = clf.decision_function(x_data[test_ind, :])

    y_hat = clf.predict(x_data[test_ind, :])

    print("test_ind")
    for fff in test_ind:
        print(fff)

    print("pred")
    for fff in pred:
        print(fff)
    print("y[test_ind] - 1")

    for fff in y[test_ind] - 1:
        print(fff)

    # fpr, tpr, thresholds = sklearn.metrics.roc_curve(y[test_ind] - 1, pred)
    #
    # roc_auc = sklearn.metrics.auc(fpr, tpr)
    # print(roc_auc)
    #
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    # plt.legend(loc='lower right')
    # # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([-0.1, 1.1])
    # plt.ylim([-0.1, 1.1])
    # plt.xlabel('False Positive Rate')  # 横坐标是fpr
    # plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    # plt.title('Receiver operating characteristic example')
    # plt.show()

    lin_auc = sklearn.metrics.roc_auc_score(y[test_ind] - 1, pred)

    lin_confusion =confusion_matrix(y[test_ind],y_hat)
    lin_TP = lin_confusion[1,1]
    lin_TN = lin_confusion[0,0]
    lin_FP = lin_confusion[0,1]
    lin_FN = lin_confusion[1,0]
    lin_sen = float (lin_TP)/ float (lin_TP+lin_FN)
    lin_spe = float(lin_TN)/float (lin_TN+lin_FP)
    print("lin_ACC",lin_acc)
    print("lin_sen",lin_sen)
    print("lin_spe", lin_spe)

    # svm
    # clf = svm.SVC(C=1, cache_size=100, class_weight=None, coef0=0.0,
    #                   decision_function_shape='ovr', degree=4, gamma='scale', kernel='linear',
    #                   max_iter=-1, probability=False, random_state=None, shrinking=True,
    #                   tol=0.001, verbose=False)
    #
    # clf.fit(x_data[train_ind, :], y[train_ind].ravel())
    # # Compute the accuracy
    # lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
    # # Compute the AUC
    # pred = clf.decision_function(x_data[test_ind, :])
    # y_hat = clf.predict(x_data[test_ind, :])
    # lin_auc = sklearn.metrics.roc_auc_score(y[test_ind] - 1, pred)
    # lin_confusion =confusion_matrix(y[test_ind],y_hat)
    # lin_TP = lin_confusion[1,1]
    # lin_TN = lin_confusion[0,0]
    # lin_FP = lin_confusion[0,1]
    # lin_FN = lin_confusion[1,0]
    # lin_sen = float (lin_TP)/ float (lin_TP+lin_FN)
    # lin_spe = float(lin_TN)/float (lin_TN+lin_FP)

    #rbf
    # clf= svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #                   decision_function_shape='ovr', degree=3, gamma=0.5, kernel='rbf',
    #                   max_iter=-1, probability=False, random_state=None, shrinking=True,
    #                   tol=0.001, verbose=False)
    # clf.fit(x_data[train_ind, :], y[train_ind].ravel())
    # # Compute the accuracy
    # lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
    # # Compute the AUC
    # pred = clf.decision_function(x_data[test_ind, :])
    # y_hat = clf.predict(x_data[test_ind, :])
    # lin_auc = sklearn.metrics.roc_auc_score(y[test_ind] - 1, pred)
    # lin_confusion = confusion_matrix(y[test_ind], y_hat)
    # lin_TP = lin_confusion[1, 1]
    # lin_TN = lin_confusion[0, 0]
    # lin_FP = lin_confusion[0, 1]
    # lin_FN = lin_confusion[1, 0]
    # lin_sen = float(lin_TP) / float(lin_TP + lin_FN)
    # lin_spe = float(lin_TN) / float(lin_TN + lin_FP)

    #RF
    # clf =RandomForestClassifier(max_depth=None, min_samples_split=2,random_state=0)
    # clf.fit(x_data[train_ind, :], y[train_ind].ravel())
    # lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
    # pred = clf.predict(x_data[test_ind, :])
    # y_hat = clf.predict(x_data[test_ind, :])
    # lin_auc = sklearn.metrics.roc_auc_score(y[test_ind] - 1, pred)
    # lin_confusion = confusion_matrix(y[test_ind], y_hat)
    # lin_TP = lin_confusion[1, 1]
    # lin_TN = lin_confusion[0, 0]
    # lin_FP = lin_confusion[0, 1]
    # lin_FN = lin_confusion[1, 0]
    # lin_sen = float(lin_TP) / float(lin_TP + lin_FN)
    # lin_spe = float(lin_TN) / float(lin_TN + lin_FP)


    #DC
    # clf =DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
    # clf.fit(x_data[train_ind, :], y[train_ind].ravel())
    # lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
    # pred = clf.predict(x_data[test_ind, :])
    # y_hat = clf.predict(x_data[test_ind, :])
    # lin_auc = sklearn.metrics.roc_auc_score(y[test_ind] - 1, pred)
    # lin_confusion = confusion_matrix(y[test_ind], y_hat)
    # lin_TP = lin_confusion[1, 1]
    # lin_TN = lin_confusion[0, 0]
    # lin_FP = lin_confusion[0, 1]
    # lin_FN = lin_confusion[1, 0]
    # lin_sen = float(lin_TP) / float(lin_TP + lin_FN)
    # lin_spe = float(lin_TN) / float(lin_TN + lin_FP)

    #GBDT
    # clf =GradientBoostingClassifier(random_state=10)
    # clf.fit(x_data[train_ind, :], y[train_ind].ravel())
    # lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
    # pred = clf.predict(x_data[test_ind, :])
    # y_hat = clf.predict(x_data[test_ind, :])
    # lin_auc = sklearn.metrics.roc_auc_score(y[test_ind] - 1, pred)
    # lin_confusion = confusion_matrix(y[test_ind], y_hat)
    # lin_TP = lin_confusion[1, 1]
    # lin_TN = lin_confusion[0, 0]
    # lin_FP = lin_confusion[0, 1]
    # lin_FN = lin_confusion[1, 0]
    # lin_sen = float(lin_TP) / float(lin_TP + lin_FN)
    # lin_spe = float(lin_TN) / float(lin_TN + lin_FP)

    #XG
    # clf =xgb.XGBClassifier(max_depth=4,learning_rate= 0.001, verbosity=1, objective='binary:logistic',random_state=1)
    # clf.fit(x_data[train_ind, :], y[train_ind].ravel())
    # lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
    # pred = clf.predict(x_data[test_ind, :])
    # y_hat = clf.predict(x_data[test_ind, :])
    # lin_auc = sklearn.metrics.roc_auc_score(y[test_ind] - 1, pred)
    # lin_confusion = confusion_matrix(y[test_ind], y_hat)
    # lin_TP = lin_confusion[1, 1]
    # lin_TN = lin_confusion[0, 0]
    # lin_FP = lin_confusion[0, 1]
    # lin_FN = lin_confusion[1, 0]
    # lin_sen = float(lin_TP) / float(lin_TP + lin_FN)
    # lin_spe = float(lin_TN) / float(lin_TN + lin_FP)



    # Classification with GCNs
    # test_acc, test_auc = Train.run_training(final_graph, sparse.coo_matrix(x_data).tolil(), y_data, train_ind, val_ind,
    #                                         test_ind, params)

    # test_acc, test_auc = Train.run_training(final_graph, sparse.coo_matrix(x_data).tolil(), y_data, train_ind, val_ind,
    #                                         test_ind, params)

    # print(test_acc)
    test_acc=0
    test_auc=0
    # return number of correctly classified samples instead of percentage
    test_acc = int(round(test_acc * len(test_ind)))
    lin_acc = int(round(lin_acc * len(test_ind)))

    # return test_acc, test_auc, lin_acc, lin_auc, fold_size

    return test_acc, test_auc, lin_acc, lin_auc, fold_size,lin_sen,lin_spe

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Graph CNNs for population graphs: '
                                                 'classification of the ABIDE dataset')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='Dropout rate (1 - keep probability) (default: 0.3)')
    parser.add_argument('--decay', default=5e-4, type=float,
                        help='Weight for L2 loss on embedding matrix (default: 5e-4)')
    parser.add_argument('--hidden', default=16, type=int, help='Number of filters in hidden layers (default: 16)')
    parser.add_argument('--lrate', default=0.005, type=float, help='Initial learning rate (default: 0.005)')
    parser.add_argument('--atlas', default='dosenbach160', help='atlas for network construction (node definition) (default: ho, '
                                                      'see preprocessed-connectomes-project.org/abide/Pipelines.html '
                                                      'for more options )')
    parser.add_argument('--epochs', default=150, type=int, help='Number of epochs to train')


    parser.add_argument('--num_features', default=8, type=int, help='Number of features to keep for '
                                                                       'the feature selection step (default: 2000)')


    parser.add_argument('--num_training', default=1.0, type=float, help='Percentage of training set used for '
                                                                        'training (default: 1.0)')
    parser.add_argument('--depth', default=0, type=int, help='Number of additional hidden layers in the GCN. '
                                                             'Total number of hidden layers: 1+depth (default: 0)')
    # parser.add_argument('--model', default='gcn_cheby', help='gcn model used (default: gcn_cheby, '
    #                                                          'uses chebyshev polynomials, '
    #                                                          'options: gcn, gcn_cheby, dense )')

    parser.add_argument('--model', default='dense', help='gcn model used (default: gcn_cheby, '
                                                             'uses chebyshev polynomials, '
                                                             'options: gcn, gcn_cheby, dense )')
    parser.add_argument('--seed', default=123, type=int, help='Seed for random initialisation (default: 123)')
    parser.add_argument('--folds', default=11, type=int, help='For cross validation, specifies which fold will be '
                                                             'used. All folds are used if set to 11 (default: 11)')
    parser.add_argument('--save', default=1, type=int, help='Parameter that specifies if results have to be saved. '
                                                            'Results will be saved if set to 1 (default: 1)')
    parser.add_argument('--connectivity', default='correlation', help='Type of connectivity used for network '
                                                                      'construction (default: correlation, '
                                                                      'options: correlation, partial correlation, '
                                                                      'tangent)')
    args = parser.parse_args()
    start_time = time.time()
    params = dict()
    params['model'] = args.model                    # gcn model using chebyshev polynomials
    params['lrate'] = args.lrate                    # Initial learning rate
    params['epochs'] = args.epochs                  # Number of epochs to train
    params['dropout'] = args.dropout                # Dropout rate (1 - keep probability)
    params['hidden'] = args.hidden                  # Number of units in hidden layers
    params['decay'] = args.decay                    # Weight for L2 loss on embedding matrix.
    params['early_stopping'] = 2    # Tolerance for early stopping (# of epochs). No early stopping if set to param.epochs
    params['max_degree'] = 3                        # Maximum Chebyshev polynomial degree.
    params['depth'] = args.depth                    # number of additional hidden layers in the GCN. Total number of hidden layers: 1+depth
    params['seed'] = args.seed                      # seed for random initialisation

    # GCN Parameters
    params['num_features'] = args.num_features      # number of features for feature selection step
    params['num_training'] = args.num_training      # percentage of training set used for training
    atlas = args.atlas                              # atlas for network construction (node definition)
    connectivity = args.connectivity                # type of connectivity used for network construction

    # Get class labels
    subject_IDs = Reader.get_ids()
    labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')

    # Get acquisition site
    sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
    unique = np.unique(list(sites.values())).tolist()

    num_classes = 2
    # print("subjects----------------------------")
    # print(subject_IDs)
    num_nodes = len(subject_IDs)

    # Initialise variables for class labels and acquisition sites
    y_data = np.zeros([num_nodes, num_classes])
    y = np.zeros([num_nodes, 1])
    site = np.zeros([num_nodes, 1], dtype=np.int)

    # Get class labels and acquisition site for all subjects
    for i in range(num_nodes):
        y_data[i, int(labels[subject_IDs[i]])-1] = 1
        y[i] = int(labels[subject_IDs[i]])
        site[i] = unique.index(sites[subject_IDs[i]])
    # ho_features = np.loadtxt(os.path.join(os.path.abspath('.')+'/path/to/data',"ho_features.csv"),delimiter=",",skiprows=0)
    # cc200_features = np.loadtxt(os.path.join(os.path.abspath('.') + '/path/to/data', "cc200_features.csv"), delimiter=",",skiprows=0)
    # ez_features = np.loadtxt(os.path.join(os.path.abspath('.') + '/path/to/data', "ez_features.csv"), delimiter=",",skiprows=0)
    # aal_features = np.loadtxt(os.path.join(os.path.abspath('.') + '/path/to/data', "aal_features.csv"), delimiter=",",skiprows=0)
    # tt_features = np.loadtxt(os.path.join(os.path.abspath('.') + '/path/to/data', "tt_features.csv"), delimiter=",",skiprows=0)
    # dosenbach160_features = np.loadtxt(os.path.join(os.path.abspath('.') + '/path/to/data', "dosenbach160_features.csv"), delimiter=",",skiprows=0)
    mlp_features = np.loadtxt(os.path.join(os.path.abspath('.') + '/path/to/data', "mlp_onlysexagegraph_sex_site_depth0.csv"), delimiter=",",skiprows=0)
    # mlp_features = mlp_features[:,0:12]
    # mlp_features = np.loadtxt(os.path.join(os.path.abspath('.') + '/path/to/data', "mlp_onlysexagegraph_sex_site_depth0.csv"), delimiter=",",skiprows=0)
    # mlp_features = mlp_features[:,2:4`]
    # ho_zuhe = np.hstack((ho_features, cc200_features,ez_features,aal_features,tt_features,dosenbach160_features))
    graph = Reader.create_affinity_graph_from_scores(['SEX','SITE_ID'], subject_IDs)#7154
    # graph = Reader.create_affinity_graph_from_scores(['HANDEDNESS_CATEGORY', 'SITE_ID'], subject_IDs)7586
    print(os.path.abspath('.'))
    # skf = StratifiedKFold(n_splits=10)
    # skf = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
    skf = StratifiedKFold(n_splits=10, random_state=11, shuffle=True)
    # skf = StratifiedKFold(n_splits=10, random_state=12, shuffle=True)
    # skf = StratifiedKFold(n_splits=10, random_state=13, shuffle=True)
    if args.folds == 11:  # run cross validation on all folds

        list3=[]
        # for train_ind, test_ind in(skf.split(np.zeros(num_nodes), np.squeeze(y))):
        #
        #     scores=train_fold_0(train_ind, test_ind, test_ind, graph, mlp_features, y, y_data, params, subject_IDs)
        #     list3.append(scores)
            # l2 = list(set(list1))
            # print(len(l2))
        scores = Parallel(n_jobs=10)(delayed(train_fold_0)(train_ind, test_ind, test_ind, graph, mlp_features, y, y_data,
                                                         params, subject_IDs)
                                     for train_ind, test_ind in
                                     reversed(list(skf.split(np.zeros(num_nodes), np.squeeze(y)))))
        params = dict()
        print(scores)
        print(list3)
        scores_acc = [x[0] for x in scores]
        scores_auc = [x[1] for x in scores]
        scores_lin = [x[2] for x in scores]
        scores_auc_lin = [x[3] for x in scores]
        fold_size = [x[4] for x in scores]
        scores_sen = [x[5] for x in scores]
        scores_spe = [x[6] for x in scores]

        print('overall linear accuracy %f' + str(np.sum(scores_lin) * 1. / num_nodes))
        print('overall sen %f' + str(np.mean(scores_sen)))
        print('overall spe %f' + str(np.mean(scores_spe)))
        print('overall linear AUC %f' + str(np.mean(scores_auc_lin)))
        print('overall accuracy %f' + str(np.sum(scores_acc) * 1. / num_nodes))
        print('overall AUC %f' + str(np.mean(scores_auc)))


        # scores = Parallel(n_jobs=10)(delayed(train_fold_0)(train_ind, test_ind, test_ind, graph, tt_features, y, y_data,
        #                                                  params, subject_IDs)
        #                              for train_ind, test_ind in reversed(list(skf.split(np.zeros(num_nodes), np.squeeze(y)))))
        # params = dict()
        # print(scores)
        #
        # scores_acc = [x[0] for x in scores]
        # scores_auc = [x[1] for x in scores]
        # scores_lin = [x[2] for x in scores]
        # scores_auc_lin = [x[3] for x in scores]
        # fold_size = [x[4] for x in scores]
        # scores_sen = [x[5] for x in scores]
        # scores_spe = [x[6] for x in scores]
        #
        # print('overall linear accuracy %f' + str(np.sum(scores_lin) * 1. / num_nodes))
        # print('overall sen %f' + str(np.mean(scores_sen)))
        # print('overall spe %f' + str(np.mean(scores_spe)))
        # print('overall linear AUC %f' + str(np.mean(scores_auc_lin)))
        # print('overall accuracy %f' + str(np.sum(scores_acc) * 1. / num_nodes))
        # print('overall AUC %f' + str(np.mean(scores_auc)))


    else:  # compute results for only one fold

        cv_splits = list(skf.split(cc200_features, np.squeeze(y)))

        train = cv_splits[args.folds][0]
        test = cv_splits[args.folds][1]
        val = test
        scores_acc, scores_auc, scores_lin, scores_auc_lin, fold_size = train_fold_0(train, test, val, graph, cc200_features, y,
                                                         y_data, params, subject_IDs)

        print('overall linear accuracy %f' + str(np.sum(scores_lin) * 1. / fold_size))
        print('overall linear AUC %f' + str(np.mean(scores_auc_lin)))
        print('overall accuracy %f' + str(np.sum(scores_acc) * 1. / fold_size))
        print('overall AUC %f' + str(np.mean(scores_auc)))