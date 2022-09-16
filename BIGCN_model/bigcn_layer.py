# -*-coding:utf-8-*-

import torch
import torch.nn.functional as F
from torch import nn as nn

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.):
        super(GraphConvolution, self).__init__()

        self.fc = nn.Linear(input_dim,output_dim, bias = False)
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

    def adj_normalize(self, A):
        B, N, _ = A.size()

        mask_nonzero = torch.nonzero(torch.sum(A + A.permute(0,2,1),dim=-1),as_tuple=True)
        (batch, row) = mask_nonzero
        # add self_loop
        A[batch, row, row] = 1
        d = A.sum(-1)

        if A.equal(A.permute(0, 2, 1)):
            symmetric = True
        else:
            symmetric = False

        if symmetric:
            # D = D^-1/2
            d_s = torch.where(d!=0,torch.pow(d, -0.5),d).detach()
            if torch.any(torch.isnan(d_s)):
                print('d_s has nan')
            if torch.any(torch.isinf(d_s)):
                print('d_s has inf')
            D = torch.diag_embed(d_s)
            return torch.matmul(torch.matmul(D, A), D)

        else:
            # D=D^-1
            d_s = torch.where(d!=0, torch.pow(d, -1), d).detach()
            if torch.any(torch.isnan(d_s)):
                print('d_s has nan')
            if torch.any(torch.isinf(d_s)):
                print('d_s has inf')
            D = torch.diag_embed(d_s)
            A = torch.matmul(D,A)
        return A


    def forward(self, x, A):
        # print('inputs:', inputs)

        A = self.adj_normalize(A).detach()
        x = self.fc(x)
        x = self.leakyrelu(x)

        out = torch.matmul(A, x)
        out = self.dropout(out)

        return out


class GCN(nn.Module):

    def __init__(self, input_dim, hid_dim, output_dim, num_features_nonzero,dropout):
        super(GCN, self).__init__()

        self.input_dim = input_dim # 1433
        self.output_dim = output_dim

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        print('num_features_nonzero:', num_features_nonzero)


        self.layers = nn.Sequential(GraphConvolution(self.input_dim, hid_dim, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout= dropout,
                                                     is_sparse_inputs=True),

                                    GraphConvolution(hid_dim, output_dim, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout= dropout,
                                                     is_sparse_inputs=False),

                                    )

    def forward(self, x, A):

        x = self.layers(x, A)

        return x

    def l2_loss(self):

        layer = self.layers.children()
        layer = next(iter(layer))

        loss = None

        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss
