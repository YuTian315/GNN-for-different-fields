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


import os
import csv
import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from nilearn import connectome
from sklearn.feature_selection import f_classif
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
# from skfeature.function.information_theoretical_based import MRMR
# Reading and computing the input data
from sklearn.manifold.t_sne import TSNE
from skfeature.function.information_theoretical_based import LCSI
# Selected pipeline
pipeline = 'cpac'

# Input data variables
root_folder = os.path.abspath('.')+'/path/to/data'
data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal')
phenotype = os.path.join(root_folder, 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')
struture_folder = os.path.join(root_folder, 'structure')

def get_structre():
    all_structure_networks=[]
    for root, dirs, files in os.walk(struture_folder):
        for file in files:
            singlestructure = pd.read_csv(os.path.join(struture_folder, file),usecols=[1],header=None).T
            all_structure_networks.append(singlestructure)

    df = pd.concat(all_structure_networks,ignore_index=True)

    print(df.isnull().sum())

    my_imputer = Imputer()
    data_imputed = my_imputer.fit_transform(df)
    print(type(data_imputed))
    # array转换成df
    df_data_imputed = pd.DataFrame(data_imputed, columns=df.columns)
    print(df_data_imputed)

    std = MinMaxScaler()
    df_biaozhun = std.fit_transform(df_data_imputed)

    return df_biaozhun
def fetch_filenames(subject_IDs, file_type):

    """
        subject_list : list of short subject IDs in string format
        file_type    : must be one of the available file types

    returns:

        filenames    : list of filetypes (same length as subject_list)
    """

    import glob

    # Specify file mappings for the possible file types
    filemapping = {'func_preproc': '_func_preproc.nii.gz',
                   'rois_dosenbach160': '_rois_dosenbach160.1D'}

    # The list to be filled
    filenames = []

    # Fill list with requested file paths
    for i in range(len(subject_IDs)):
        os.chdir(data_folder)  # os.path.join(data_folder, subject_IDs[i]))
        try:
            filenames.append(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
        except IndexError:
            # Return N/A if subject ID is not found
            filenames.append('N/A')

    return filenames


# Get timeseries arrays for list of subjects
def get_timeseries(subject_list, atlas_name):
    """
        subject_list : list of short subject IDs in string format
        atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200

    returns:
        time_series  : list of timeseries arrays, each of shape (timepoints x regions)
    """

    timeseries = []
    for i in range(len(subject_list)):
        subject_folder = os.path.join(data_folder, subject_list[i])
        ro_file = [f for f in os.listdir(subject_folder) if f.endswith('_rois_' + atlas_name + '.1D')]
        fl = os.path.join(subject_folder, ro_file[0])
        print("Reading timeseries file %s" %fl)
        timeseries.append(np.loadtxt(fl, skiprows=0))

    return timeseries


# Compute connectivity matrices
def subject_connectivity(timeseries, subject, atlas_name, kind, save=True, save_path=data_folder):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        subject      : the subject ID
        atlas_name   : name of the parcellation atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder

    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    print("Estimating %s matrix for subject %s" % (kind, subject))

    if kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform([timeseries])[0]

        np.savetxt(os.path.join(save_path, subject,
                                    subject + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.csv'),connectivity, delimiter=',')

    if save:
        subject_file = os.path.join(save_path, subject,
                                    subject + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
        # np.savetxt(subject_file, {'connectivity': connectivity}, delimiter=',')
        sio.savemat(subject_file, {'connectivity': connectivity})

        # np.loadtxt(os.path.join(os.path.abspath('.'), "power_871.csv"), delimiter=",", skiprows=0)

    return connectivity


# Get the list of subject IDs
def get_ids(num_subjects=None):
    """

    return:
        subject_IDs    : list of all subject IDs
    """

    subject_IDs = np.genfromtxt(os.path.join(data_folder, 'subject_IDs.txt'), dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs


# Get phenotype values for a list of subjects
def get_subject_score(subject_list, score):
    scores_dict = {}

    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                scores_dict[row['SUB_ID']] = row[score]

    return scores_dict


# Dimensionality reduction step for the feature vector using a ridge classifier
def feature_selection(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    estimator = RidgeClassifier()
    selector = RFE(estimator, fnum, step=100, verbose=1)

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)

    print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])

    return x_data  #949*2000


def feature_tsne(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    model_pca = TSNE(n_components=fnum)
    x_data = model_pca.fit_transform(matrix)

    return x_data


def feature_mrmr(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    train_set = matrix[train_ind,:].astype(float)

    labels = np.loadtxt(os.path.join(os.path.abspath('.') + '/path/to/data', "label.csv"), delimiter=",",skiprows=0)

    train_y=labels[train_ind].astype(np.int16)
    # print(labels)
    # # labels=labels.reshape(1,949)
    # train_y=[]
    # for i in range(100):
    #     train_y.append(labels[i])

    # train_y=labels[0:100].reshape(1,).astype(np.int16)
    # train_y = np.array(labels[0:100].astype(np.int16))

    F, J_CMI, MIfy = LCSI.lcsi(train_set, train_y, gamma=0, function_name='MRMR', n_selected_features=fnum)
    #
    # feature_index= MRMR.mrmr(train_set, train_y, n_selected_features=fnum)
    # # transformed_train = train_set[:, feature_index]
    # x_data=matrix[:feature_index]
    print("Number of labeled samples %d" % len(F))
    # # print("Number of features selected %d" % x_data.shape[1])

    x_data = matrix[:,F]

    return x_data


def feature_PCA(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    model_pca = PCA(n_components=fnum)
    x_data = model_pca.fit(matrix).transform(matrix)

    return x_data

def f_score(X, y):
    """
    This function implements the anova f_value feature selection (existing method for classification in scikit-learn),
    where f_score = sum((ni/(c-1))*(mean_i - mean)^2)/((1/(n - c))*sum((ni-1)*std_i^2))

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y : {numpy array},shape (n_samples,)
        input class labels

    Output
    ------
    F: {numpy array}, shape (n_features,)
        f-score for each feature
    """
    F, pval = f_classif(X, y)
    return F

def feature_ranking(F):
    """
    Rank features in descending order according to f-score, the higher the f-score, the more important the feature is
    """
    idx = np.argsort(F)
    return idx[::-1]


def feature_selection_fscore(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]

    score = f_score(featureX, featureY)

    idx = feature_ranking(score)

    features = matrix[:, idx[0:fnum]]

    x_data = features
    # selector = selector.fit(featureX, featureY.ravel())
    # x_data = selector.transform(matrix)

    print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])

    return x_data

def feature_selection_L1(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]

    lsvc = LinearSVC(C=0.1, penalty="l1", dual=False).fit(featureX, featureY)

    model = SelectFromModel(lsvc, prefit=True)

    x_data = model.transform(matrix)


    # selector = selector.fit(featureX, featureY.ravel())
    # x_data = selector.transform(matrix)

    print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])

    return x_data


def feature_selection_GBDT(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]

    gbdt = GradientBoostingClassifier(
        init=None,
        learning_rate=0.1,
        loss='deviance',
        max_depth=3,
        max_features=None,
        max_leaf_nodes=None,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_estimators=100,
        random_state=None,
        subsample=1.0,
        verbose=0,
        warm_start=False)

    gbdt.fit(featureX, featureY)

    score = gbdt.feature_importances_

    x_data = matrix[:,gbdt.feature_importances_>0]
    # selector = selector.fit(featureX, featureY.ravel())
    # x_data = selector.transform(matrix)

    print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])

    return x_data




# Make sure each site is represented in the training set when selecting a subset of the training set
def site_percentage(train_ind, perc, subject_list):
    """
        train_ind    : indices of the training samples
        perc         : percentage of training set used
        subject_list : list of subject IDs

    return:
        labeled_indices      : indices of the subset of training samples
    """

    train_list = subject_list[train_ind]
    sites = get_subject_score(train_list, score='SITE_ID')
    unique = np.unique(list(sites.values())).tolist()
    site = np.array([unique.index(sites[train_list[x]]) for x in range(len(train_list))])

    labeled_indices = []

    for i in np.unique(site):
        id_in_site = np.argwhere(site == i).flatten()

        num_nodes = len(id_in_site)
        labeled_num = int(round(perc * num_nodes))
        labeled_indices.extend(train_ind[id_in_site[:labeled_num]])

    return labeled_indices


# Load precomputed fMRI connectivity networks
def get_networks(subject_list, kind, atlas_name, variable='connectivity'):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks


    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """

    all_networks = []
    for subject in subject_list:
        fl = os.path.join(data_folder, subject,
                          subject + "_" + atlas_name + "_" + kind + ".mat")
        matrix = sio.loadmat(fl)[variable]

        # corela = np.corrcoef(matrix)
        #
        # matrix = (matrix+corela)/2
        all_networks.append(matrix)
    # all_networks=np.array(all_networks)

    idx = np.triu_indices_from(all_networks[0], 1)
    norm_networks = [np.arctanh(mat) for mat in all_networks]
    vec_networks = [mat[idx] for mat in norm_networks]
    matrix = np.vstack(vec_networks)

    return matrix


# Construct the adjacency matrix of the population from phenotypic scores
def create_affinity_graph_from_scores(scores, subject_list):
    """
        scores       : list of phenotypic information to be used to construct the affinity graph
        subject_list : list of subject IDs

    return:
        graph        : adjacency matrix of the population graph (num_subjects x num_subjects)
    """

    num_nodes = len(subject_list)
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = get_subject_score(subject_list, l)

        # quantitative phenotypic scores
        if l in ['AGE_AT_SCAN', 'FIQ']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[subject_list[k]]) - float(label_dict[subject_list[j]]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[subject_list[k]] == label_dict[subject_list[j]]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph
