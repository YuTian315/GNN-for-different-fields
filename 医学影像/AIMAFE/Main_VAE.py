import numpy as np
import torch.nn

from Dataloader import *
from sklearn.model_selection import StratifiedKFold, train_test_split
from utils import *
from VAE import *
import random
from scipy.special import softmax
# ============== 参数设置 =============== #
# opt = OptInit().initialize()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Using {device} in torch")
set_seed(100)
# ============== 加载数据 =============== #
print('  Loading dataset ...')
atlas = "dosenbach160"
Raw_Feature, Label = Load_Raw_Data(atlas=atlas)

ALL_val_result_list = []
ALL_test_result_list = []
for times in range(1):
    random_state_seed = random.randint(0, 1000)
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state_seed)
    cv_splits = list(skf.split(Raw_Feature, Label))

    # ============== 创建模型 =============== #
    loss_ce = torch.nn.CrossEntropyLoss()
    loss_mse = torch.nn.MSELoss()
    Labels = torch.tensor(Label, dtype=torch.long).to(device)
    # ============== 划分测试验证集 =============== #
    Best_val_list = []
    Best_test_list = []
    ALL_val_list = []
    ALL_test_list = []
    pred_list = []
    test_list = []
    for fold in range(n_folds):
        Best_pred = 0
        print("\r\n========================== Fold {} ==========================".format(fold))
        train_ind = cv_splits[fold][0]
        test_ind = cv_splits[fold][1]
        test_label = Label[test_ind]
        train_label = Label[train_ind]

        test_list.append(test_ind)
        Raw_Feature = preprocess_features(Raw_Feature)
        Feature_cuda = torch.tensor(Raw_Feature, dtype=torch.float32).to(device)

        Model = VAE(input_dim = Feature_cuda.size()[1], nhid=16, nclass=2, dropout=0.3)
        # Cls_Model = MLP(input_dim=600)
        # Cls_Model.to(device)
        Model.to(device)
        optimizer = torch.optim.Adam(Model.parameters(), lr=0.003, weight_decay=5e-5)
        # Cls_optimizer = torch.optim.Adam(Cls_Model.parameters(), lr=0.0005, weight_decay=5e-5)
    # ============== 训练模型 =============== #
        ACC = 0
        Best_Test_result = []
        for epoch in range(300):
            Model.train()  # 启用batch normalization和drop out
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                output, rec_f_loss, kl = Model(Feature_cuda)
                ce_loss = loss_ce(output[train_ind], Labels[train_ind])
                loss = ce_loss + kl + rec_f_loss
                loss.backward()
                optimizer.step()
            train_results = Metric(train_label, output[train_ind].detach().cpu().numpy(), soft=False, dim=2)
            Model.eval()
            with torch.set_grad_enabled(False):
                output, _ , _ = Model(Feature_cuda)

            test_results = Metric(test_label, output[test_ind].detach().cpu().numpy(), soft=False, dim=2)

            test_result_numpy = output[test_ind].detach().cpu().numpy()
            test_result_numpy = softmax(test_result_numpy, axis=1)
            test_result_numpy = (test_result_numpy == test_result_numpy.max(axis=1, keepdims=1)).astype(int)[:, 1]
            if test_results[0] >= ACC and epoch > 9:
                ACC = test_results[0]
                Best_epoch = epoch
                Best_Test_result = test_results
                Best_pred = test_result_numpy
            print("KL 散度: {:.4f}, rec_f_loss: {:.4f}".format(kl.item(), rec_f_loss.item()), end=" ")
            print("Epoch: {}, output loss: {:.4f}, train acc: {:.4f}, test acc: {:.4f}".format(epoch, ce_loss.item(), train_results[0].item(), test_results[0].item()))

        print("\r\n => Fold {} Best Test accuacry {:.5f}  Best_epoch {:.5f}".format(fold, Best_Test_result[0], Best_epoch))
         # 每折最好的ACC结果分数
        Best_test_list.append(Best_Test_result)
        pred_list.append(Best_pred)

    test_list = np.concatenate(test_list)
    pred_list = np.concatenate(pred_list)
    np.save(f"./result/{atlas}_result.npy", pred_list)
    np.save("./result/test_ind.npy", test_list)

    ALL_test_list.append(Best_test_list)
    ALL_test_result_list.append(Best_test_list)

    print(f"{times}-测试集结果：")
    Calc_All_Metric(ALL_test_list)

print("测试集总结果：")
Calc_All_Metric(ALL_test_result_list)
print("finish")