import numpy as np
from skfeature.function.information_theoretical_based import MRMR
from sklearn.datasets import load_iris  # 利用iris数据作为演示数据集
from skfeature.function.information_theoretical_based import LCSI
import os
# 载入数据集
iris = load_iris()
X, y = iris.data, iris.target

# 选择前100个观测点作为训练集
# 剩下的50个观测点作为测试集
# 由于skfeature中的mRMR仅适用于离散变量
# 因此我们通过将float转换为int而把所有连续变量转换为离散变量
# 此转换仅用于演示目的

# train_set = X[0:100,:].astype(int)
# test_set = X[100:,].astype(float)
# train_y = y[0:100].astype(int)

mlp_features = np.loadtxt(os.path.join(os.path.abspath('.') + '/path/to/data', "mlp_onlysexagegraph_sex_site_depth0.csv"), delimiter=",",skiprows=0)

train_set = mlp_features[0:100,:].astype(float)
train_y = y[0:100].astype(int)

feature_index,_,_ = MRMR.mrmr(train_set, train_y, n_selected_features=2) # 在训练集上训练

F, J_CMI, MIfy = LCSI.lcsi(train_set, train_y, gamma=0, function_name='MRMR', n_selected_features=20)

transformed_train = train_set[:,feature_index] # 转换训练集
assert np.array_equal(transformed_train, train_set[:,[2,3]])  # 其选择了第三个及第四个变量

transformed_test = test_set[:,feature_index] # 转换测试集
assert np.array_equal(transformed_test, test_set[:,[2,3]]) 