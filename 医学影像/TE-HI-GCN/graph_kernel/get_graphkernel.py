import numpy as np
from scipy.io import loadmat


def load_correlation(name):

    data_path = '../data/different_mask_correlation&label_with_same_order/pcc_correlation_871_'
    data_dict = loadmat(data_path + name + '_.mat')
    data_array = data_dict['connectivity']

    len_data = data_array.shape[0]

    print('len:', len_data, data_array.shape)
    where_are_nan = np.isnan(data_array)
    where_are_inf = np.isinf(data_array)
    for bb in range(0, len_data):
        for i in range(0, data_array.shape[1]):
            for j in range(0, data_array.shape[1]):
                if where_are_nan[bb][i][j]:
                    data_array[bb][i][j] = 0
                if where_are_inf[bb][i][j]:
                    data_array[bb][i][j] = 0.8

    return data_array, len_data


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

    return distMat


def new(name):
    pcc, len_data = load_correlation(name)

    graph_kernel = np.zeros((len_data, len_data))
    w = np.zeros((pcc.shape[0], pcc.shape[1], pcc.shape[2]))
    for wi in range(len_data):
        w[wi] = calcDistMatrix(pcc[wi])
    for i in range(len_data):
        print('i', i)
        for j in range(i+1, len_data):
            mat = calcKernelEntry(pcc[i], pcc[j], w[i], w[j], 0.5)
            graph_kernel[i][j] = mat
            graph_kernel[j][i] = mat

    print('mat:', graph_kernel)
    np.savetxt('graph_kernel_'+name+'.txt', graph_kernel)
    return


new(name='ho')

