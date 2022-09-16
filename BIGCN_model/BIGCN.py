import sys,os
sys.path.append(os.getcwd())
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean


from BIGCN_model.bigcn_layer import GraphConvolution
import copy
from torch import nn as nn


class TDrumorGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GraphConvolution(in_feats, hid_feats, dropout)
        self.conv2 = GraphConvolution(hid_feats+in_feats, out_feats, dropout)
        self.layernorm = nn.LayerNorm(normalized_shape=hid_feats, elementwise_affine=False)
        self.dropout = dropout

    def forward(self,x, A, mask):


        x1 = copy.copy(x.float())
        x = self.conv1(x, A)
        x = F.leaky_relu(x)
        x= self.layernorm(x)
        x2 = copy.copy(x)
        root_extend = torch.repeat_interleave(x1[:,0,:].unsqueeze(dim=1), x1.size()[1], dim=1)
        mask_fea = torch.repeat_interleave(mask, x1.size()[-1], dim=-1)
        root_extend.masked_fill_(mask_fea, 0.0)
        x = torch.cat((x, root_extend), -1)
        x = F.dropout(x, p= self.dropout)
        x = self.conv2(x, A)
        x = F.leaky_relu(x)
        root_extend = torch.repeat_interleave(x2[:, 0, :].unsqueeze(dim=1), x2.size()[1], dim=1)
        mask_fea = torch.repeat_interleave(mask, x2.size()[-1], dim=-1)
        root_extend.masked_fill_(mask_fea, 0.0)
        x = torch.cat((x, root_extend), -1)
        x = torch.mean(x, dim=1)
        return x


class BUrumorGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GraphConvolution(in_feats, hid_feats, dropout)
        self.conv2 = GraphConvolution(hid_feats + in_feats, out_feats, dropout)
        self.layernorm = nn.LayerNorm(normalized_shape=hid_feats, elementwise_affine=False)

        self.dropout = dropout

    def forward(self, x, A, mask):
        x1 = copy.copy(x.float())
        x = self.conv1(x, A)
        x = F.leaky_relu(x)
        x = self.layernorm(x)
        x2 = copy.copy(x)
        root_extend = torch.repeat_interleave(x1[:, 0, :].unsqueeze(dim=1), x1.size()[1], dim=1)
        mask_fea = torch.repeat_interleave(mask, x1.size()[-1], dim=-1)
        root_extend.masked_fill_(mask_fea, 0.0)
        x = torch.cat((x, root_extend), -1)
        x = F.dropout(x, p=self.dropout)
        x = self.conv2(x, A)
        x = F.leaky_relu(x)
        root_extend = torch.repeat_interleave(x2[:, 0, :].unsqueeze(dim=1), x2.size()[1], dim=1)
        mask_fea = torch.repeat_interleave(mask, x2.size()[-1], dim=-1)
        root_extend.masked_fill_(mask_fea, 0.0)
        x = torch.cat((x, root_extend), -1)
        x = torch.mean(x, dim=1)
        return x


class BirumorGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout):
        super(BirumorGCN, self).__init__()
        self.conv1 = GraphConvolution(in_feats, hid_feats, dropout)
        self.conv2 = GraphConvolution(hid_feats + in_feats, hid_feats, dropout)
        self.layernorm = nn.LayerNorm(normalized_shape=hid_feats, elementwise_affine=False)

        self.dropout = dropout

    def forward(self, x, A, mask):
        x1 = copy.copy(x.float())
        x = self.conv1(x, A)
        x = F.leaky_relu(x)
        x = self.layernorm(x)
        x2 = copy.copy(x)
        root_extend = torch.repeat_interleave(x1[:, 0, :].unsqueeze(dim=1), x1.size()[1], dim=1)
        mask_fea = torch.repeat_interleave(mask, x1.size()[-1], dim=-1)
        root_extend.masked_fill_(mask_fea, 0.0)
        x = torch.cat((x, root_extend), -1)
        x = F.dropout(x, p=self.dropout)
        x = self.conv2(x, A)
        x = F.leaky_relu(x)
        root_extend = torch.repeat_interleave(x2[:, 0, :].unsqueeze(dim=1), x2.size()[1], dim=1)
        mask_fea = torch.repeat_interleave(mask, x2.size()[-1], dim=-1)
        root_extend.masked_fill_(mask_fea, 0.0)
        x = torch.cat((x, root_extend), -1)
        return x

class Reconstruction(nn.Module):
    def __init__(self,num_posts, in_feats, hid_feats, out_feats, dropout):
        super(Reconstruction, self).__init__()
        self.num_posts = num_posts
        self.Npost = nn.Parameter(torch.Tensor(num_posts,in_feats))
        self.recon = BirumorGCN(in_feats, hid_feats, out_feats, dropout)
        self.fc = nn.Linear(hid_feats * 2, out_feats, bias = False)

        nn.init.kaiming_uniform_(self.Npost)

    def forward(self, x, A, mask_zero):
        x = x.unsqueeze(dim=1)
        x = torch.repeat_interleave( x, self.num_posts, dim=1)
        mask_fea = torch.repeat_interleave(mask_zero, x.size()[-1], dim=-1)
        x = torch.mul(x, mask_fea)
        recx = torch.mul(x, self.Npost)
        recx = self.recon(recx, A, mask_zero)
        recx = self.fc(recx)
        return recx


class BIGCN(nn.Module):
    def __init__(self,in_feats, h_GCN, num_posts, dropout=0):
        super(BIGCN, self).__init__()
        hid_feats, out_feats = h_GCN
        self.emb = nn.Sequential(nn.Linear(in_feats, hid_feats, bias=False),
                                 nn.LeakyReLU())
        self.layernorm = nn.LayerNorm(normalized_shape= hid_feats, elementwise_affine=False)

        self.TDrumorGCN = TDrumorGCN(hid_feats, hid_feats, out_feats, dropout)
        self.BUrumorGCN = BUrumorGCN(hid_feats, hid_feats, out_feats, dropout)
        self.fc = nn.Linear((out_feats+out_feats)*2, 2)

    def forward(self, x, A):
        x= self.emb(x)
        x = self.layernorm(x)
        mask_zero = torch.sum(x,dim=-1,keepdim=True) ==  0
        TD_x = self.TDrumorGCN(x, A, mask_zero)
        BU_x = self.BUrumorGCN(x, A.permute(0,2,1), mask_zero)
        x = torch.cat((BU_x,TD_x), -1)
        pred = self.fc(x)
        #x = F.log_softmax(x, dim=1)
        return pred, x

class BIGCN_CCFD(nn.Module):
    def __init__(self, in_feats, h_GCN, num_posts, dropout=0):
        super(BIGCN_CCFD, self).__init__()
        hid_feats, out_feats = h_GCN
        self.emb = nn.Sequential(nn.Linear(in_feats,hid_feats,bias=False),
                                 nn.LeakyReLU())
        self.layernorm = nn.LayerNorm(normalized_shape=hid_feats, elementwise_affine=False)

        self.TDrumorGCN = TDrumorGCN(hid_feats, hid_feats, out_feats, dropout)
        self.BUrumorGCN = BUrumorGCN(hid_feats, hid_feats, out_feats, dropout)
        self.reconstruction = Reconstruction(num_posts, out_feats *4, hid_feats, in_feats, dropout)

        self.domaingate = nn.Sequential(nn.Linear(out_feats * 4, out_feats * 4),
                                        nn.Sigmoid())

        self.classifier_all = nn.Linear(out_feats * 4, 2)
        self.classifier_com = nn.Linear(out_feats * 4, 2)

    def gateFeature(self,x, gate):
        xc = torch.mul(x, gate)
        xs = torch.mul(x, 1 - gate)
        return xc, xs


    def forward(self, x, A):

        x = self.emb(x)
        x = self.layernorm(x)
        mask_zero = torch.sum(x, dim=-1,keepdim=True) ==  0
        TD_x = self.TDrumorGCN(x, A, mask_zero)
        BU_x = self.BUrumorGCN(x, A.permute(0,2,1), mask_zero)
        xf = torch.cat((BU_x,TD_x), -1)

        dgate = self.domaingate(xf)
        xc, xs = self.gateFeature(xf, dgate)

        xall = xc + xs
        preds = self.classifier_all(xall)

        preds_xc = self.classifier_com(xc)
        preds_xs = self.classifier_all(xc.detach() + xs)

        A_sys = copy.copy(A)
        A_sys.masked_fill_(A.permute(0,2,1)==1,1)
        x_rec = self.reconstruction(xc + xs,  A_sys, mask_zero)

        return preds, preds_xc, preds_xs, xc, xs, x_rec, dgate