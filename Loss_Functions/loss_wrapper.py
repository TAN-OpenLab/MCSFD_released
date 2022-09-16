# -*-coding:utf-8-*-
"""
@Project    : MCSFD
@Time       : 2022/3/27 14:19
@Author     : Danke Wu
@File       : loss_wrapper.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from typing import List
import math

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def bhattacharyya_distance(x,y):
    x = torch.softmax(x, dim=-1)
    y = torch.softmax(y, dim=-1)
    dis = torch.sum(torch.sqrt(torch.mul(x,y)+1e-12),dim=-1)
    dis = torch.mean(dis, dim=0)
    return -torch.log(dis)

def Wasserstein_distance(x,y):
    x = torch.softmax(x, dim=-1)
    y = torch.softmax(y, dim=-1)
    dis = torch.abs(torch.mean(x, dim=-1) - torch.mean(y,dim=-1))
    dis = torch.mean(dis, dim=0)
    return dis


class Center_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # self.distanceloss = nn.L1Loss()
        # self.distanceloss = nn.MSELoss()
        # self.distanceloss = nn.CosineSimilarity()

    def f2_scale(self, X:List[torch.Tensor]):
        output = []
        for x in X:
            if x.dim()> 1 :
                if x.size()[0] >1:
                    x_n2 = torch.norm(x, p=2, dim=-1, keepdim=True)
                    x_n2.data.masked_fill_(x_n2 == 0, 1)
                    x = x / x_n2
                else:
                    if torch.norm(x, p=2, dim=-1) != 0:
                        x = x / (torch.norm(x, p=2, dim=-1))
            output.append(x)

        return output

    def forward(self, positive, mean):
        #L1
        # [positive, mean]= self.f2_scale( [positive, mean])
        # posi_dis = self.distanceloss(positive, torch.repeat_interleave(mean, positive.size()[0],dim=0))
        # ct_loss = posi_dis
        # # #consine
        # [positive, mean] = self.f2_scale([positive, mean])
        # posi_dis = 1 - self.distanceloss(positive, torch.repeat_interleave(mean, positive.size()[0],dim=0))
        # ct_loss = torch.mean(posi_dis,dim=0)

        # #bhattacharyya
        posi_dis = bhattacharyya_distance(positive, torch.repeat_interleave(mean, positive.size()[0], dim=0))
        ct_loss = posi_dis
        return ct_loss


class Reconstruction_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_rec, x):
        x = x.view(x.size()[0], -1)
        x_rec = x_rec.view(x_rec.size()[0], -1)
        nonzero_num = torch.sum(x != 0, dim=-1, keepdim=True)
        mask = torch.where( x!=0 , 1, 0 )
        mask = mask / nonzero_num
        distance = torch.abs(x - x_rec)
        rec_loss = torch.sum(torch.mul(distance, mask),dim=-1)
        return torch.mean(rec_loss,dim=0)

class LossWrapper_center(nn.Module):
    def __init__(self) -> None:
        super(LossWrapper_center, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.l1loss = nn.L1Loss()
        self.centerloss = Center_Loss()
        self.kldloss = nn.KLDivLoss(reduction="batchmean")
        self.distanceloss = Reconstruction_Loss()

    def forward(self, preds: List[torch.Tensor], targets: List[torch.Tensor], *args, **kwargs):
        """
        Calculate the total loss between model prediction and target list

        Args:
            pred (torch.Tensor): a list of model prediction
            targets (List[torch.Tensor]): a list of targets for multi-task / multi loss training

        Returns:
            loss (torch.FloatTensor): a weighted loss tensor
            loss_list (tuple[torch.FloatTensor]): a tuple of loss item
            pred (torch.FloatTensor): model output without grad
        """

        pred, xc_pred, xs_pred, xc, xs, x_rec, fgate = preds
        y, x = targets
        loss = 0

        #pred_xc celoss
        loss_ce = self.celoss(pred, y)
        loss_ce_c = self.kldloss(torch.log_softmax(xc_pred,dim=-1), torch.softmax(pred.detach(),dim=-1))
        # loss_ce_s = self.kldloss(torch.log_softmax(xs_pred, dim=-1), torch.softmax(pred.detach(), dim=-1))

        #center loss
        rumor_index = y
        nonrumor_index = torch.abs(y - 1)
        rumor_true_index = torch.mul(rumor_index, torch.max(xc_pred, dim=-1)[1])
        rumor_feature_scale = rumor_true_index / (torch.sum(rumor_true_index) + 1e-4)
        nonrumor_true_index = torch.mul(nonrumor_index, 1 - torch.max(xc_pred, dim=-1)[1])
        nonrumor_feature_scale = nonrumor_true_index / (torch.sum(nonrumor_true_index) + 1e-4)

        rumor_anchor = torch.matmul(rumor_feature_scale.unsqueeze(0), xc)
        nonrumor_anchor = torch.matmul(nonrumor_feature_scale.unsqueeze(0), xc)

        if torch.sum(nonrumor_index) == 0:
            loss_consis = self.centerloss(xc[rumor_index == 1, :], rumor_anchor.detach()) #+self.ctloss(xc_n[rumor_index == 1, :], rumor_xc_n, nonrumor_anchor)
        elif torch.sum(rumor_index) ==0:
            loss_consis = self.centerloss(xc[nonrumor_index == 1, :], nonrumor_anchor.detach())
        else:
            loss_consis = self.centerloss(xc[rumor_index == 1, :], rumor_anchor.detach()) +\
                          self.centerloss(xc[nonrumor_index == 1, :], nonrumor_anchor.detach())

        specific_weight = F.relu(torch.softmax(xc_pred.detach() + 1e-12, dim=-1) - torch.softmax(xs_pred + 1e-12, dim=-1))
        rumor_true_liklehood = torch.mul(specific_weight[:,1], rumor_index)
        nonrumor_true_likelihood = torch.mul(specific_weight[:,0], nonrumor_index)

        loss_lg = torch.sum( rumor_true_liklehood, dim=0)/torch.sum(rumor_index) + torch.sum(nonrumor_true_likelihood, dim=0)/torch.sum(nonrumor_index)

        fgate = torch.mean(fgate, dim=0, keepdim=True)
        loss_fgate = torch.matmul(1 - fgate, fgate.permute(1, 0))
        if loss_fgate > 0:
            loss_fgate = loss_fgate - 0.01
        else:
            loss_fgate = torch.tensor([0],device=loss_fgate.device)

        loss_rec = self.distanceloss(x_rec, x)

        # #without loss_c
        # loss += loss_ce + loss_ce_c  +loss_lg  + loss_consis+ loss_fgate # + loss_rec
        # return loss, (loss_ce.item(), loss_ce_c.item(), loss_consis.item(), loss_rec.item()) #loss_consis.item() loss_lg.item()

        loss += loss_ce +  loss_ce_c + loss_consis + loss_rec + loss_fgate + loss_lg

        return loss, ( loss_ce.item(), loss_ce_c.item(), loss_consis.item(),  loss_rec.item())
