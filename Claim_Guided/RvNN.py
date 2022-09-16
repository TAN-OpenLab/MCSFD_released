import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def Tree_child_AGG(h,adj):
    d = torch.sum(adj,dim=-1)
    d = torch.where(d!=0, 1/d,  torch.zeros_like(d))
    adj = adj * d.unsqueeze(-1)
    c_agg = torch.matmul(adj, h)

    return c_agg

def Tree_Parent_AGG(h,adj):

    adj_T = adj.permute(0, 2, 1)  # bs x c x 1 x n x n
    d = torch.sum(adj, dim=-1)
    d = torch.where(d != 0, 1 / d, torch.zeros_like(d))
    adj_T= adj_T * d.unsqueeze(-1)
    p_agg = torch.matmul(adj_T, h)
    return p_agg


class RvNN_BU(nn.Module):
    def __init__(self,f_in, f_out,num_posts):
        super(RvNN_BU,self).__init__()

        self.emb = nn.Linear(f_in,f_out,bias= False)
        self.ioux = nn.Linear(f_out, f_out * 2 )
        self.iouh = nn.Linear(f_out, f_out * 2 )
        self.coux = nn.Linear(f_out, f_out )
        self.couh = nn.Linear(f_out, f_out)
        self.fc = nn.Linear(f_out, 2)

    def forward(self,h,adj):
        h_e = self.emb(h)
        #h_e = h
        h_child = Tree_child_AGG(h_e,adj)
        h_e_copy = torch.clone(h_e)
        for i in range(h.size()[1]):
            x = h_e[:,-i-1,:]
            x_child = h_child[:,-i-1,:]
            iou = self.ioux(x) + self.iouh(x_child)
            r, z = torch.split(iou, iou.size(1) // 2, dim=1)  # 取出三个门 i 输入门， o输出门
            r, z = torch.sigmoid(r), torch.sigmoid(z)
            hc = torch.tanh(self.coux(x) + self.couh(r * x_child))
            h_new = z * x_child + (1 - z) * hc

            h_e_copy[:,-i-1,:] = h_new
            h_child = Tree_child_AGG(h_e_copy,adj)

        h_e = h_e[:,0,:]
        y = self.fc(h_e)
        return y


class RvNN_TD_All_Nodes(nn.Module):
    def __init__(self,f_in, f_out,num_posts):
        super(RvNN_TD_All_Nodes,self).__init__()

        self.emb = nn.Linear(f_in,f_out,bias= False)
        self.ioux = nn.Linear(f_out, f_out * 2 )
        self.iouh = nn.Linear(f_out, f_out * 2 )
        self.coux = nn.Linear(f_out, f_out )
        self.couh = nn.Linear(f_out, f_out)
        self.fc = nn.Linear(f_out,2)

    def forward(self,h,adj):
        h_e = self.emb(h)
        #h_e= h
        h_parent = Tree_Parent_AGG(h_e,adj)
        h_e_copy = torch.clone(h_e)
        for i in range(h.size()[1]):
            x = h_e[:,i,:]
            x_parent = h_parent[:,i,:]
            iou = self.ioux(x) + self.iouh( x_parent)
            r, z = torch.split(iou, iou.size(1) // 2, dim=1)  # 取出三个门 i 输入门， o输出门
            r, z = torch.sigmoid(r), torch.sigmoid(z)
            hc = torch.tanh(self.coux(x) + self.couh(r * x_parent))
            h_new = z * x_parent + (1 - z) * hc

            h_e_copy[:,i,:] = h_new
            h_parent = Tree_Parent_AGG(h_e_copy,adj)

        h_e_copy = torch.max(h_e_copy,dim=1)[0]
        y = self.fc(h_e_copy)
        return y

class RvNN_TD_Leaf_Nodes(nn.Module):
    def __init__(self,f_in, f_out, num_posts):
        super(RvNN_TD_Leaf_Nodes,self).__init__()

        self.emb = nn.Linear(f_in,f_out,bias= False)
        self.ioux = nn.Linear(f_out, f_out * 2 )
        self.iouh = nn.Linear(f_out, f_out * 2 )
        self.coux = nn.Linear(f_out, f_out )
        self.couh = nn.Linear(f_out, f_out)
        self.fc = nn.Linear(f_out, 2)

    def forward(self,h,adj):
        h_e = self.emb(h)
        #h_e = h
        h_parent = Tree_Parent_AGG(h_e,adj)
        h_e_copy = torch.clone(h_e)
        for i in range(h.size()[1]):
            x = h_e[:,i,:]
            x_parent = h_parent[:,i,:]
            iou = self.ioux(x) + self.iouh(x_parent)
            r, z = torch.split(iou, iou.size(1) // 2, dim=1)  # 取出三个门 i 输入门， o输出门
            r, z = torch.sigmoid(r), torch.sigmoid(z)
            hc = torch.tanh(self.coux(x) + self.couh(x_parent * r))
            h_new = z * x_parent + (1 - z) * hc

            h_e_copy[:,i,:] = h_new
            h_parent = Tree_Parent_AGG(h_e_copy,adj)

        h_leaf = torch.where(torch.sum(adj,dim=-1,keepdim=True)==0,1,0)
        h_leaf.data.masked_fill_(torch.sum(h,dim=-1,keepdim=True)==0,0)
        h_e_copy = torch.mul(h_e_copy,h_leaf)
        h_e_copy = torch.max(h_e_copy,dim=1)[0]
        # h_e_copy = torch.mean(h_e_copy, dim=1)
        y = self.fc(h_e_copy)
        return y,h_e_copy


class RvNN(nn.Module):
    def __init__(self,f_in, f_out):
        super(RvNN,self).__init__()
        self.ioux = nn.Linear(f_in, f_out * 2)
        self.iouh = nn.Linear(f_in, f_out * 2)
        self.coux = nn.Linear(f_in, f_out)
        self.couh = nn.Linear(f_in, f_out)

    def forward(self, h_e, adj):
        h_parent = Tree_Parent_AGG(h_e, adj)
        h_e_copy = torch.clone(h_e)
        for i in range(h_e.size()[1]):
            x = h_e[:, i, :]
            x_parent = h_parent[:, i, :]
            iou = self.ioux(x) + self.iouh(x_parent)
            r, z = torch.split(iou, iou.size(1) // 2, dim=1)  # 取出三个门 i 输入门， o输出门
            r, z = torch.sigmoid(r), torch.sigmoid(z)
            hc = torch.tanh(self.coux(x) + self.couh(x_parent * r))
            h_new = z * x_parent + (1 - z) * hc

            h_e_copy[:, i, :] = h_new
            h_parent = Tree_Parent_AGG(h_e_copy, adj)

        return h_e_copy

class Reconstruction(nn.Module):
    def __init__(self,num_posts, f_in,f_out):
        super(Reconstruction, self).__init__()
        self.num_posts = num_posts
        self.Npost = nn.Parameter(torch.Tensor(num_posts, 1))
        self.recon = RvNN(f_out, f_out)
        self.reemb = nn.Linear(f_out, f_in, bias= False)

        nn.init.kaiming_uniform_(self.Npost)

    def forward(self, x, adj, mask):
        x = x.unsqueeze(dim=1)
        x = torch.repeat_interleave(x, self.num_posts, dim=1)
        mask = torch.repeat_interleave(mask, x.size()[-1], dim=-1)
        x = torch.mul(x, mask)
        x_post = torch.mul(self.Npost, x)
        recx = self.recon(x_post, adj)
        recx = self.reemb(recx)
        return recx


class RvNN_TD_Leaf_Nodes_CCFD(nn.Module):
    def __init__(self,f_in, f_out,num_posts):
        super(RvNN_TD_Leaf_Nodes_CCFD,self).__init__()

        self.emb = nn.Linear(f_in,f_out,bias= False)
        self.rvnn = RvNN(f_out, f_out)
        self.reconstruction = Reconstruction(num_posts, f_in,f_out)

        self.domaingate = nn.Sequential(nn.Linear(f_out, f_out),
                                        nn.Sigmoid())

        self.classifier_all = nn.Linear(f_out, 2)
        self.classifier_com = nn.Linear(f_out, 2)

    def gateFeature(self, x, gate):
        xc = torch.mul(x, gate)
        xs = torch.mul(x, 1 - gate)
        return xc, xs

    def forward(self,h,adj):

        mask = torch.sum(h, dim=-1, keepdim=True)==0
        h_e = self.emb(h)
        h_e_copy = self.rvnn(h_e, adj)

        h_leaf = torch.where(torch.sum(adj,dim=-1,keepdim=True)==0,1,0)
        h_leaf.masked_fill_(torch.sum(h,dim=-1,keepdim=True)==0,0)
        xf = torch.mul(h_e_copy,h_leaf)
        xf_max = torch.max(xf,dim=1)[0]
        # xf_max = torch.mean(xf, dim=1)

        dgate = self.domaingate(xf_max)
        xc, xs = self.gateFeature(xf_max, dgate)

        xall = xc + xs
        preds = self.classifier_all(xall)

        preds_xc = self.classifier_com(xc)
        preds_xs = self.classifier_all(xc.detach() + xs)

        x_rec = self.reconstruction(xc + xs, adj, mask)

        return preds, preds_xc, preds_xs, xc, xs, x_rec, dgate
