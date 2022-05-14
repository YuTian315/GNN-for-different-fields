from torch.autograd.variable import Variable
import pickle
# from train_old_version2 import train, evaluate
import warnings
import pandas as pd
import os
import tqdm
import numpy as np
import scipy.io as sio
warnings.filterwarnings("ignore")


def get_key(file_name):
    file_name = file_name.split('_')
    key = ''
    for i in range(len(file_name)):
        if file_name[i] == 'rois':
            key = key[:-1]
            break
        else:
            key += file_name[i]
            key += '_'
    return key


def load_data(path1, path2, name):
    profix = path1
    dirs = os.listdir(profix)
    dirss = np.sort(dirs)
    print(dirss)
    all = {}
    labels = {}
    all_data = []
    label = []
    for filename in dirss:
        a = np.loadtxt(path1 + filename)
        a = a.transpose()
        # a = a.tolist()
        all[filename] = a
        all_data.append(a)
        data = pd.read_csv(path2)
        for i in range(len(data)):
            if get_key(filename) == data['FILE_ID'][i]:
                if int(data['DX_GROUP'][i]) == 2:
                    labels[filename] = int(data['DX_GROUP'][i]-1)
                    label.append(int(data['DX_GROUP'][i]-1))
                else:
                    labels[filename] = 0
                    label.append(0)
                break
    label = np.array(label)
    np.savetxt('data/correlation&label/871_label_' + name + '.txt', label, fmt='%s')
    return all_data, label


def cal_pcc(data):

    corr_matrix = []
    for key in range(len(data)):
        corr_mat = np.corrcoef(data[key])
        corr_matrix.append(corr_mat)
    data_array = np.array(corr_matrix)
    subject_file = os.path.join('data/correlation&label/pcc_correlation_' + str(871) + '_' + name + '_.mat')
    sio.savemat(subject_file, {'connectivity': data_array})

    return data_array


name = 'ho'
data_path = 'data/ABIDE-871/' + name + '/ABIDE_pcp/cpac/filt_global/'
label_path = 'data/Phenotypic_V1_0b_preprocessed1.csv'

# 导入数据
raw_data, labels = load_data(data_path, label_path, name)  # raw_data [871 116 ?]  labels [871]
# 划分时间窗
adj = cal_pcc(raw_data)
print('finished')


