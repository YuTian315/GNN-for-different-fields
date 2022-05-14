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
    def __init__(self, in_dim, hidden_dim, out_dim, brain_region_num):
        super(model_gnn, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.brain_region_num = brain_region_num

        self.gcn1_p = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_p = GCN(hidden_dim, hidden_dim, 0.2)
        self.gcn3_p = GCN(hidden_dim, hidden_dim, 0.2)
        self.gcn1_n = GCN(in_dim, hidden_dim, 0.2)
        self.gcn2_n = GCN(hidden_dim, hidden_dim, 0.2)
        self.gcn3_n = GCN(hidden_dim, hidden_dim, 0.2)
        self.kernel_p = nn.Parameter(torch.FloatTensor(self.brain_region_num, in_dim))  #
        self.kernel_n = nn.Parameter(torch.FloatTensor(self.brain_region_num, in_dim))
        init.xavier_uniform_(self.kernel_p)
        init.xavier_uniform_(self.kernel_n)
        self.lin1 = nn.Linear(2 * in_dim*hidden_dim, 16)
        self.lin2 = nn.Linear(16, self.out_dim)
        self.losses = []

    def dim_reduce(self, adj_matrix, num_reduce,
                   ortho_penalty, variance_penalty, neg_penalty, kernel):
        kernel_p = torch.nn.functional.relu(kernel)
        # np.savetxt('kernel_p.txt', kernel_p.cpu().data.numpy())
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
        # np.savetxt('kernel_p' + str(flag) + '.txt', torch.nn.functional.relu(self.kernel_p).cpu().data.numpy())
        p_conv1 = self.gcn1_p(None, p_reduce)
        p_conv2 = self.gcn2_p(p_conv1, p_reduce)
        p_conv3 = self.gcn3_p(p_conv2, p_reduce)
        n_reduce = self.dim_reduce(s_feature_n, self.in_dim, 0.2, 0.5, 0.1, self.kernel_n)
        # np.savetxt('kernel_n' + str(flag) + '.txt', torch.nn.functional.relu(self.kernel_n).cpu().data.numpy())
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
            # init.xavier_uniform_(self.bias)
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
    def __init__(self, in_dim, hidden_dim, graph_adj, num, thr='0', num_feats=50, num_nodes=871, num_classes=2,
                 brain_region_num=116, name='aal', device='cuda'):
        super(MyHIgcn, self).__init__()
        self.graph_adj = graph_adj
        self.num_nodes = num_nodes
        self.num_feats = num_feats
        self.num_classes = num_classes
        self.num = num
        self.thr = thr
        self.name = name
        self.brain_region_num = brain_region_num

        self.gnn = model_gnn(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=num_classes,
                             brain_region_num=self.brain_region_num).to(device)
        self.gnn.load_state_dict(
            torch.load('model/fgcn/ASD_' + self.name + '/fgcn' + str(self.num) + '_0.' + self.thr + '.pth'))

        self.gcn1 = DenseGCNConv(in_channels=50, out_channels=128)

        self.fc1 = nn.Linear(128, 32).to(device)
        self.fc2 = nn.Linear(32, self.num_classes).to(device)

    def forward(self, nodes, nodes_adj, device='cuda'):

        embedding, cluster_loss = self.gnn(self.graph_adj[nodes], self.num)  # xx*50

        out1 = self.gcn1(embedding, nodes_adj)

        out = self.fc1(out1)
        out = self.fc2(out)

        out = out.squeeze()

        return out, cluster_loss

    def loss(self, pred, label):
        return F.cross_entropy(pred, label, reduction='mean')


class EHIgcn2(nn.Module):
    def __init__(self, in_dim, hidden_dim, graph_adj1, graph_adj2, graph_adj3, graph_adj4, graph_adj5, graph_adj6,
                 graph_adj7, graph_adj8, graph_adj9, graph_adj10, num,
                 thr1='0', thr2='0', thr3='0', thr4='0', thr5='0', thr6='0', thr7='0', thr8='0', thr9='0', thr10='0',
                 num_feats=50, num_classes=2, brain_region_num=116, name='aal', device='cuda'):
        super(EHIgcn2, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.graph_adj1 = graph_adj1
        self.graph_adj2 = graph_adj2
        self.graph_adj3 = graph_adj3
        self.graph_adj4 = graph_adj4
        self.graph_adj5 = graph_adj5
        self.graph_adj6 = graph_adj6
        self.graph_adj7 = graph_adj7
        self.graph_adj8 = graph_adj8
        self.graph_adj9 = graph_adj9
        self.graph_adj10 = graph_adj10

        self.thr1 = thr1
        self.thr2 = thr2
        self.thr3 = thr3
        self.thr4 = thr4
        self.thr5 = thr5
        self.thr6 = thr6
        self.thr7 = thr7
        self.thr8 = thr8
        self.thr9 = thr9
        self.thr10 = thr10

        self.num_feats = num_feats
        self.num_classes = num_classes
        self.num = num
        self.brain_region_num = brain_region_num
        self.name = name

        self.higcn1 = MyHIgcn(in_dim=self.in_dim, hidden_dim=self.hidden_dim, graph_adj=self.graph_adj1, num=self.num,
                              thr=self.thr1, brain_region_num=self.brain_region_num, name=self.name)
        self.higcn2 = MyHIgcn(in_dim=self.in_dim, hidden_dim=self.hidden_dim, graph_adj=self.graph_adj2, num=self.num,
                              thr=self.thr2, brain_region_num=self.brain_region_num, name=self.name)
        self.higcn3 = MyHIgcn(in_dim=self.in_dim, hidden_dim=self.hidden_dim, graph_adj=self.graph_adj3, num=self.num,
                              thr=self.thr3, brain_region_num=self.brain_region_num, name=self.name)
        self.higcn4 = MyHIgcn(in_dim=self.in_dim, hidden_dim=self.hidden_dim, graph_adj=self.graph_adj4, num=self.num,
                              thr=self.thr4, brain_region_num=self.brain_region_num, name=self.name)
        self.higcn5 = MyHIgcn(in_dim=self.in_dim, hidden_dim=self.hidden_dim, graph_adj=self.graph_adj5, num=self.num,
                              thr=self.thr5, brain_region_num=self.brain_region_num, name=self.name)
        self.higcn6 = MyHIgcn(in_dim=self.in_dim, hidden_dim=self.hidden_dim, graph_adj=self.graph_adj6, num=self.num,
                              thr=self.thr6, brain_region_num=self.brain_region_num, name=self.name)
        self.higcn7 = MyHIgcn(in_dim=self.in_dim, hidden_dim=self.hidden_dim, graph_adj=self.graph_adj7, num=self.num,
                              thr=self.thr7, brain_region_num=self.brain_region_num, name=self.name)
        self.higcn8 = MyHIgcn(in_dim=self.in_dim, hidden_dim=self.hidden_dim, graph_adj=self.graph_adj8, num=self.num,
                              thr=self.thr8, brain_region_num=self.brain_region_num, name=self.name)
        self.higcn9 = MyHIgcn(in_dim=self.in_dim, hidden_dim=self.hidden_dim, graph_adj=self.graph_adj9, num=self.num,
                              thr=self.thr9, brain_region_num=self.brain_region_num, name=self.name)
        self.higcn10 = MyHIgcn(in_dim=self.in_dim, hidden_dim=self.hidden_dim, graph_adj=self.graph_adj10, num=self.num,
                              thr=self.thr10, brain_region_num=self.brain_region_num, name=self.name)

    def forward(self, nodes, nodes_adj, device='cuda'):

        out1, cl1 = self.higcn1(nodes, nodes_adj)
        out2, cl2 = self.higcn2(nodes, nodes_adj)
        out3, cl3 = self.higcn3(nodes, nodes_adj)
        out4, cl4 = self.higcn4(nodes, nodes_adj)
        out5, cl5 = self.higcn5(nodes, nodes_adj)
        out6, cl6 = self.higcn6(nodes, nodes_adj)
        out7, cl7 = self.higcn7(nodes, nodes_adj)
        out8, cl8 = self.higcn8(nodes, nodes_adj)
        out9, cl9 = self.higcn9(nodes, nodes_adj)
        out10, cl10 = self.higcn10(nodes, nodes_adj)

        out = out1+out2+out3+out4+out5+out6+out7+out8+out9+out10

        cl = cl1 + cl2 + cl3 + cl4 + cl5 + cl6 + cl7 + cl8 + cl9 + cl10

        return out, cl

    def loss(self, pred, label):
        return F.cross_entropy(pred, label, reduction='mean')




