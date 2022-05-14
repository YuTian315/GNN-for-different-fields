

import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.nn import init
import torch
import numpy as np
warnings.filterwarnings("ignore")


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

    def dim_reduce(self, adj_matrix, num_reduce,
                   ortho_penalty, variance_penalty, neg_penalty, kernel):
        kernel_p = torch.nn.functional.relu(kernel)
        np.savetxt('kernel_p.txt', kernel_p.cpu().data.numpy())
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

        out1 = self.lin1(conv_concat)
        output = self.lin2(out1)
        output = torch.softmax(output, dim=1)
        loss = torch.sum(torch.tensor(self.losses))
        self.losses.clear()
        return conv_concat, loss


class DenseGCNConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(DenseGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = nn.Parameter(torch.Tensor(self.in_channels, out_channels))
        init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.bias.data = init.constant_(self.bias.data, 0.0)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


class MyHIgcn(nn.Module):
    def __init__(self, in_dim, hidden_dim, graph_adj, num, thr=0, num_feats=50, num_nodes=871, num_classes=2, device='cuda'):
        super(MyHIgcn, self).__init__()
        self.graph_adj = graph_adj
        self.num_nodes = num_nodes
        self.num_feats = num_feats
        self.num_classes = num_classes
        self.num = num
        self.thr = thr

        self.gnn = model_gnn(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=num_classes).to(device)

        self.gcn1 = DenseGCNConv(in_channels=50, out_channels=128)

        self.fc1 = nn.Linear(128, 32).to(device)
        self.fc2 = nn.Linear(32, self.num_classes).to(device)

    def forward(self, nodes, nodes_adj, device='cuda'):

        embedding, cluster_loss = self.gnn(self.graph_adj[nodes])  # xx*50

        out1 = self.gcn1(embedding, nodes_adj)

        out = self.fc1(out1)
        out = self.fc2(out)
        out = out.squeeze()

        return out, cluster_loss

    def loss(self, pred, label):
        return F.cross_entropy(pred, label, reduction='mean')


