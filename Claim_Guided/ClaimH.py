# -*-coding:utf-8-*-
import torch
import torch.nn as nn
from Claim_Guided.ClaimH_layer import Post_level_Atten,BatchGAT, Opin_Diff, Event_level_Attent


class ClaimH(nn.Module):
    def __init__(self, num_heads, f_in, h_DUGAT, f_hid, attn_dropout, num_class):
        super(ClaimH, self).__init__()

        self.sent_emb = nn.Sequential(nn.Linear(f_in, f_hid, bias=False),
                                      nn.LeakyReLU())
        self.layernorm = nn.LayerNorm(normalized_shape=f_hid, elementwise_affine= False)
        self.post_level_atten = Post_level_Atten(f_hid)
        self.GAT = BatchGAT( num_heads, f_hid, h_DUGAT, dropout=0.2, attn_dropout=attn_dropout)
        self.diff = Opin_Diff( h_DUGAT[-1], f_hid, dropout=0.2)
        #self.lstm_text = LSTM_block(channels, f_hid+num_nodes, LSTM_X, batch_first=True)
        self.event_level_atten = Event_level_Attent(h_DUGAT[-1])

        self.mlp = nn.Sequential(
            nn.Linear(h_DUGAT[-1] * 2,num_class),  # h_sgg + num_nodes,h_LSTM
        )

    def forward(self, x, adj):

        mask = torch.nonzero(torch.sum(x,dim=-1),as_tuple=True)

        x = self.sent_emb(x)
        x = self.layernorm(x)
        x_1, p_atten = self.post_level_atten(x, mask)
        x = torch.cat((x, x_1),dim=-1)
        adj_sys = adj
        adj_T = adj.permute(0,2,1)
        adj_sys.data.masked_fill_(adj_T==1, 1).detach()
        x_agg = self.GAT(x, adj_sys,mask)
        if torch.any(torch.isnan(x_agg)):
            print('x_agg has nan')
        x_diff = self.diff(x_agg,mask)
        if torch.any(torch.isnan(x_diff)):
            print('x_diff has nan')
        x_agg_attn,_ = self.event_level_atten(x_agg,x_diff)
        if torch.any(torch.isnan(x_agg_attn)):
            print('x_agg_attn has nan')
        x_agg = torch.mean(x_agg,dim=1)
        x = torch.cat((x_agg, x_agg_attn),dim=-1)
        if torch.any(torch.isinf(x)):
            print('x has inf')
        if torch.any(torch.isnan(x)):
            print('x has nan')
        pred = self.mlp(x)
        return pred, x

class Reconstruction(nn.Module):
    def __init__(self,num_posts, num_heads, f_hid, h_DUGAT, dropout, attn_dropout):
        super(Reconstruction, self).__init__()
        self.num_posts = num_posts
        self.Npost = nn.Parameter(torch.Tensor(num_posts,f_hid*2))
        self.recon = BatchGAT(num_heads, f_hid, h_DUGAT, dropout, attn_dropout)

        nn.init.kaiming_uniform_(self.Npost)

    def forward(self, x, A, mask_zero, p_atten, e_atten):
        x = x.unsqueeze(dim=1)
        x = torch.repeat_interleave( x, self.num_posts, dim=1)
        e_atten_inv = torch.mul(self.Npost, e_atten)
        recx = torch.mul(x, e_atten_inv)
        recx = self.recon(recx, A, mask_zero)
        return recx

class ClaimH_CCFD(nn.Module):
    def __init__(self, num_heads, f_in, h_DUGAT, f_hid, attn_dropout, num_posts):
        super(ClaimH_CCFD, self).__init__()

        self.sent_emb = nn.Sequential(nn.Linear(f_in, f_hid,bias=False),
                                      nn.LeakyReLU())
        self.layernorm = nn.LayerNorm(normalized_shape=f_hid, elementwise_affine= False)
        self.post_level_atten = Post_level_Atten(f_hid)
        self.GAT = BatchGAT( num_heads, f_hid, h_DUGAT, dropout=0.2, attn_dropout=attn_dropout)
        self.diff = Opin_Diff( h_DUGAT[-1], f_hid, dropout=0.2)
        #self.lstm_text = LSTM_block(channels, f_hid+num_nodes, LSTM_X, batch_first=True)
        self.event_level_atten = Event_level_Attent(h_DUGAT[-1])
        recover = []
        recover.extend(h_DUGAT[:-1][::-1])
        recover.append(f_in)

        self.reconstruction = Reconstruction(num_posts, num_heads, h_DUGAT[-1], recover ,dropout=0.2, attn_dropout=attn_dropout)

        self.domaingate = nn.Sequential(nn.Linear(h_DUGAT[-1] *2, h_DUGAT[-1] *2),
                                        nn.Sigmoid())

        self.classifier_all = nn.Linear(h_DUGAT[-1] *2, 2)
        self.classifier_com = nn.Linear(h_DUGAT[-1] *2, 2)

    def gateFeature(self, x, gate):
        xc = torch.mul(x, gate)
        xs = torch.mul(x, 1 - gate)
        return xc, xs

    def forward(self, x, adj):

        mask = torch.nonzero(torch.sum(x,dim=-1),as_tuple=True)

        x = self.sent_emb(x)
        x = self.layernorm(x)
        x_1, p_atten = self.post_level_atten(x, mask)
        x = torch.cat((x, x_1),dim=-1)
        adj_sys = adj
        adj_T = adj.permute(0,2,1)
        adj_sys.data.masked_fill_(adj_T==1, 1).detach()
        x_agg = self.GAT(x, adj_sys,mask)
        x_diff = self.diff(x_agg, mask)
        x_agg_attn, e_atten = self.event_level_atten(x_agg, x_diff)
        x_agg = torch.mean(x_agg,dim=1)
        xf = torch.cat((x_agg, x_agg_attn),dim=-1)

        dgate = self.domaingate(xf)
        xc, xs = self.gateFeature(xf, dgate)

        xall = xc + xs
        preds = self.classifier_all(xall)

        preds_xc = self.classifier_com(xc)
        preds_xs = self.classifier_all(xc.detach() + xs)

        x_rec = self.reconstruction(xc + xs, adj_sys, mask, p_atten, e_atten)

        return preds, preds_xc, preds_xs, xc, xs, x_rec, dgate
