import torch
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, nhid, nclass, dropout):
        super(VAE, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, 2400),
            nn.Dropout(p=0.3),
            nn.ReLU()
        )
        self.deconder1 = nn.Sequential(
            nn.Linear(1200, input_dim),
            nn.Sigmoid()
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(1200, 600),
            nn.Dropout(p=0.1),
            nn.ReLU()
        )
        self.deconder2 = nn.Sequential(
            nn.Linear(600, 1200),
            nn.Sigmoid()
        )

        # self.encoder3 = nn.Sequential(
        #     nn.Linear(600, 300),
        #     nn.Dropout(p=0.1),
        #     nn.ReLU()
        # )
        # self.deconder3 = nn.Sequential(
        #     nn.Linear(300, 600),
        #     nn.Sigmoid()
        # )

        self.mse_loss = torch.nn.MSELoss()
        self.MLP = nn.Sequential(
                torch.nn.Linear(600, 128),
                torch.nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                torch.nn.Linear(128, nclass))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        # x = F.normalize(x, dim=0)
        encode_feature_1 = self.encoder1(x)
        mu, siama = encode_feature_1.chunk(2, dim=1)
        normal = torch.rand_like(siama)
        encode_feature_1 = mu + siama * normal
        decoder_feature_1 = self.deconder1(encode_feature_1)

        log_encode_feature = F.log_softmax(encode_feature_1, dim=-1)
        softmax_normal = F.softmax(normal, dim=-1)
        kl = F.kl_div(log_encode_feature, softmax_normal, reduction='sum')

        encode_feature_2 = self.encoder2(encode_feature_1)
        decoder_feature_2 = self.deconder2(encode_feature_2)

        # encode_feature_3 = self.encoder3(encode_feature_2)
        # decoder_feature_3 = self.deconder3(encode_feature_3)
        sum_mse = 0
        sum_mse += self.mse_loss(x, decoder_feature_1)
        sum_mse += self.mse_loss(encode_feature_1, decoder_feature_2)
        # sum_mse += self.mse_loss(encode_feature_2, decoder_feature_3)

        output = self.MLP(encode_feature_2)
        return output, sum_mse, kl


