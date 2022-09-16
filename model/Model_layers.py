# -*-coding:utf-8-*-
"""
@Project    : MCSFD
@Time       : 2022/5/2 20:24
@Author     : Danke Wu
@File       : Model_layers.py
"""
# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Encoder import TransformerEncoder,Decoder
import math
import copy
import torch_scatter


class sentence_embedding(nn.Module):
    def __init__(self,num_posts, h_in,h_out):
        super(sentence_embedding, self).__init__()
        self.embedding = nn.Linear(h_in, h_out, bias=False)
        self.leakrelu = nn.LeakyReLU()
        # self.insnorm = nn.InstanceNorm1d(num_posts,affine=False)

    def forward(self,x):

        x = self.embedding(x)
        x = self.leakrelu(x)
        return x

class Extractor_GAT(nn.Module):
    def __init__(self, n_head, h_in, h_hid, dropout, attn_dropout =0):
        super(Extractor_GAT,self).__init__()
        self.encoder = BatchGAT( n_head, h_in, h_hid, dropout, attn_dropout, bias =False)
        self.post_attn = Post_Attn(h_in)

    def forward(self,x, A, mask_nonzero):

        xc = self.encoder(x,A,mask_nonzero)
        xc, attn = self.post_attn(xc,mask_nonzero)

        return xc, attn


class Extractor(nn.Module):
    def __init__(self, num_layers, n_head, h_in, h_hid, dropout):
        super(Extractor,self).__init__()
        self.encoder = TransformerEncoder(num_layers, h_in, n_head,  h_hid, dropout)
        self.post_attn = Post_Attn(h_in)

    def forward(self,x, A, mask_nonzero):

        xc = self.encoder(x,A,mask_nonzero)
        xc,attn = self.post_attn(xc,mask_nonzero)

        return xc, attn

class Content_reconstruction(nn.Module):
    def __init__(self,num_layers, n_posts,  h_hid, n_head, e_hid, c_in, dropout, pos_embed=None):
        super(Content_reconstruction, self).__init__()
        self.n_posts = n_posts
        self.attn_inverse = nn.Parameter(torch.FloatTensor(n_posts,1))
        self.decoder = Decoder(num_layers, h_hid, n_head, e_hid,  dropout, pos_embed)
        self.fc = nn.Sequential(nn.Linear(h_hid, c_in, bias=False),
                                 nn.LayerNorm(c_in))
        nn.init.constant_(self.attn_inverse, 1.0)

    def forward(self, xc, xs, attn, A, mask_nonzero):

        # x = torch.cat((xc,xs), dim=-1)
        x = xc + xs
        x = torch.repeat_interleave(torch.unsqueeze(x,dim=1), self.n_posts, dim=1)
        # # index_nonzeromask
        # (batch, row) = mask_nonzero
        # mask_fea = torch.zeros((xc.size()[0], self.n_posts, xc.size()[1]), device=xc.device)
        # mask_fea[batch, row, :] = 1
        attn_ = torch.mul(attn, self.attn_inverse)
        x = torch.mul(attn_, x)

        recov_xc = self.decoder(x, A, mask_nonzero)
        recov_xc = self.fc(recov_xc)

        return recov_xc


class Post_Attn(nn.Module):
    def __init__(self,h_in):
        super(Post_Attn, self).__init__()
        self.Attn = nn.Linear(2 * h_in,1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mask_nonzero):
        (batch, row) = mask_nonzero
        root = torch.zeros_like(x,device=x.device)
        root[batch,row,:] = x[batch,0,:]

        x_plus = torch.cat([x,root],dim=-1)
        attn = self.Attn(x_plus)
        attn.masked_fill_(attn ==0, -1e20)
        attn = self.softmax(attn)
        x = torch.matmul(x.permute(0, 2, 1),attn).squeeze()
        return x, attn


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


class MCSFD(nn.Module):
    def __init__(self, c_in, num_layers, n_head, c_hid, e_hid, dropout, num_posts):
        super(MCSFD, self).__init__()

        self.embedding = sentence_embedding(num_posts, c_in, c_hid)
        self.extractor = Extractor(num_layers,n_head, c_hid, e_hid, dropout)
        self.reconstruction = Content_reconstruction(num_layers, num_posts, c_hid, n_head,
                                                     e_hid, c_in, dropout)

        self.domaingate = nn.Sequential(nn.Linear(c_hid,c_hid),
                                        nn.Sigmoid())

        self.classifier_all = nn.Linear(c_hid, 2)
        self.classifier_com = nn.Linear( c_hid, 2)

    def gateFeature(self,x, gate):

        xc = torch.mul(x, gate)
        xs = torch.mul(x, 1 - gate)

        return xc, xs

    def forward(self, x, A):

        x = self.embedding(x)
        batch_size, max_len, _= x.size()
        mask_nonzero = torch.nonzero(x.sum(-1), as_tuple=True)
        xf, attn = self.extractor(x, A, mask_nonzero)  # , stance
        # xs = self.extractor2(x, A, mask_nonzero)
        dgate = self.domaingate(xf)
        xc, xs = self.gateFeature(xf, dgate)

        xall = xc + xs
        preds = self.classifier_all(xall)
        preds_xs = self.classifier_all(xc.detach() + xs)

        preds_xc = self.classifier_com(xc)

        x_rec = self.reconstruction(xc, xs, attn, A, mask_nonzero)

        return preds, preds_xc,preds_xs, xc, xs, x_rec, dgate
