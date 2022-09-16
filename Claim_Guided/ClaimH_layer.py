# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Post_level_Atten(nn.Module):
    def __init__(self,h_dim):
        super(Post_level_Atten, self).__init__()
        self.w = nn.Parameter(torch.Tensor(h_dim, 1))
        self.u = nn.Parameter(torch.Tensor(h_dim, 1))
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.u)

    def forward(self,x, mask_nonzero):
        B, N, _ = x.size()
        (batch, row) = mask_nonzero
        root = torch.zeros_like(x, device=x.device)
        root[batch, row, :] = x[batch, 0, :]
        g = self.sigmoid( torch.matmul(x, self.w) + torch.matmul(root,self.u))
        x = torch.mul(x, g) + torch.mul(root, 1-g)
        return x,g

class BatchGAT(nn.Module):
    def __init__(self, n_heads, f_in, n_units, dropout, attn_dropout, bias =False):
        super(BatchGAT, self).__init__()
        self.n_layer = len(n_units)
        self.f_in = f_in
        self.dropout = dropout
        self.bias = bias

        self.layer_stack = nn.ModuleList()
        for i in range(self.n_layer):
            f_in = n_units[i-1] * n_heads[i] if i else self.f_in*2
            self.layer_stack.append(
                BatchMultiHeadGraphAttention( n_heads[i], f_in=f_in,
                                             f_out=n_units[i], attn_dropout=attn_dropout,bias =self.bias)
            )

    def forward(self, x, adj, mask):
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = gat_layer(x, adj, mask) # bs x c x n_head x n x f_out
            if i + 1 == self.n_layer:
                x = x.mean(dim=1)

            else:
                x = F.elu(x.permute(0, 2, 1, 3).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        return x


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias = False):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.fc = nn.Linear(f_in, f_out, bias= False)
        self.attn_src_linear = nn.Linear(f_out, 1 * n_head, bias= False )
        self.attn_dst_linear = nn.Linear(f_out, 1 * n_head, bias= False)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)


    def init_glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def forward(self, h, adj, mask):
        (batch, row) = mask
        # node_mask = h.sum(-1,keepdim=True) == 0
        # h = h.unsqueeze(1)
        h_prime = self.fc(h)
        # h_prime = torch.matmul(h, self.w) # bs x c x n_head x n x f_out
        B, N, F = h.size()  # h is of size bs x c x n x f_in


        attn_src = self.attn_src_linear(h_prime).permute(0, 2, 1).unsqueeze(dim=-1)
        attn_dst = self.attn_dst_linear(h_prime).permute(0, 2, 1).unsqueeze(dim=-1)

        attn = attn_src.expand(-1, -1,-1, N) + attn_dst.expand(-1, -1,-1, N).permute(0, 1, 3, 2) # bs  x c x n_head x n x n
        attn_all = self.leaky_relu(attn)

        adj[batch, row, row] = 1
        adj = torch.repeat_interleave(adj.unsqueeze(dim=1), self.n_head, dim=1)
        attn_all.masked_fill_(adj == 0, 0.0)
        attn_mask = attn_all[batch, :, row, :]
        attn_mask.masked_fill_(attn_mask == 0., -1e20)
        attn_mask = self.softmax(attn_mask)
        attn_all[batch, :, row, :] = attn_mask
        attn_all = self.dropout(attn_all)
        # adj_ = torch.where(adj == 1, 0.0, -1.0e12).detach()
        # attn_all = self.softmax(attn_all + adj_)

        h_prime = h_prime.unsqueeze(1)
        output = torch.matmul(attn_all, h_prime) # bs x c x n_head x n x f_out

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Opin_Diff(nn.Module):
    def __init__(self,f_in,f_out,dropout):
        super(Opin_Diff, self).__init__()
        self.linear = nn.Linear(f_in *4,f_out,bias=False)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self,x, mask_nonzero):
        (batch, row) = mask_nonzero
        root = torch.zeros_like(x, device=x.device)
        root[batch, row, :] = x[batch, 0, :]
        x_prod = torch.mul(root, x)
        x_diff = torch.abs(root - x)
        x_all = torch.cat((root,x,x_prod,x_diff),dim=-1)
        x_all = self.linear(x_all)
        x_all = self.tanh(x_all)
        x_all = self.dropout(x_all)
        return x_all

class Event_level_Attent(nn.Module):
    def __init__(self,f_in):
        super(Event_level_Attent, self).__init__()
        self.linear = nn.Linear(f_in, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-2)

    def forward(self,x_agg, x_diff):
        atten = self.tanh(self.linear(x_diff))
        atten = self.softmax(atten)
        x_agg = x_agg.permute(0,2,1)
        x_agg_att = torch.matmul(x_agg, atten)
        return x_agg_att.squeeze(),atten







