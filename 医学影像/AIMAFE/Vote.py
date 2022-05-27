import numpy as np
from Dataloader import Load_Raw_Data
from utils import Metric

_, Label = Load_Raw_Data()
idx = np.load("result/val_ind.npy")
aal = np.load("result/aal_result.npy")
cc200 = np.load("result/cc200_result.npy")
dosenbach = np.load("result/dosenbach160_result.npy")
all_result = aal + cc200 + dosenbach
all_result = np.where(all_result > 1, 1, 0)
idx_label = Label[idx]
result = Metric(idx_label, all_result, soft=True, dim=1, datatype="numpy")

print("ACC: {:.2f}%".format(result[0] * 100))
print("AUC: {:.2f}%".format(result[1] * 100))
print("SPE: {:.2f}%".format(result[2] * 100))
print("SEN: {:.2f}%".format(result[3] * 100))
print("F1: {:.2f}%".format(result[4] * 100))
print("finish")