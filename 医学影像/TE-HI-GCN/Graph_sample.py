from torch.utils.data import Dataset


class datasets(Dataset):
    def __init__(self, adj, label):
        self.adj_all = adj
        self.labels = label

    def __getitem__(self, idx):

        adj = self.adj_all[idx]
        return_dic = {'adj': adj,
                      'label': self.labels[idx]
                      }

        return return_dic

    def __len__(self):
        return len(self.labels)


