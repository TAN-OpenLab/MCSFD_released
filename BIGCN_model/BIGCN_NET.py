# -*-coding:utf-8-*-
"""
@Project    : MCSFD
@Time       : 2022/4/14 15:58
@Author     : Danke Wu
@File       : BIGCN_NET.py
"""
# -*-coding:utf-8-*-
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
from dataset.Mydataset_enhance import MyDataset, DataLoaderX
from dataset.Dataloader_enhance import seperate_dataloader, normal_dataloader
from BIGCN_model.BIGCN import BIGCN,BIGCN_CCFD
from Loss_Functions.metrics import evaluationclass
from model.Model_layers import sentence_embedding, LambdaLR

import os, sys
import time
from itertools import cycle as CYCLE


class BIGCN_NET(object):
    def __init__(self, args, device):

        # parameters
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.batch_size = args.batch_size
        self.num_worker = args.num_worker
        self.lr = args.lr
        self.dropout = args.dropout
        self.weight_decay = args.weight_decay
        self.device = device
        self.b1, self.b2 = args.b1, args.b2
        self.checkpoint = args.start_epoch
        self.patience = args.patience
        self.model_path = os.path.join(args.save_dir, args.model_dir, args.dataset)

        #model parameters
        self.f_in = args.f_in
        self.h_GCN = args.h_GCN
        self.num_class = args.num_class
        self.num_worker = args.num_worker
        self.num_posts = args.num_posts


        # =====================================load rumor_detection model================================================

        self.bigcn = BIGCN(self.f_in, self.h_GCN, self.num_posts, self.dropout)

        self.bigcn.to(self.device)
        print(self.bigcn)

        # =====================================load loss function================================================
        self.ce = nn.CrossEntropyLoss()


        self.optimizer = torch.optim.SGD( self.bigcn.parameters(), lr= self.lr, momentum= 0.9, weight_decay=1e-2)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                              lr_lambda=LambdaLR(self.epochs, self.start_epoch, decay_start_epoch=self.weight_decay).step)

        torch.autograd.set_detect_anomaly(True)

    def train_epoch(self, datapath, start_epoch):

        nonrumor_loader, rumor_loader = seperate_dataloader(datapath,'train', self.batch_size,self.num_worker,self.num_posts)
        val_loader = normal_dataloader(datapath, 'val', self.batch_size, self.num_worker, self.num_posts)
        test_loader = normal_dataloader(datapath, 'test', self.batch_size, self.num_worker, self.num_posts)

        acc_check = 0
        loss_check = 100
        start_time = time.clock()


        for epoch in range(start_epoch, self.epochs):

            train_loss, train_acc = self.train_batch(epoch, nonrumor_loader, rumor_loader)
            print(train_loss, train_acc)
            self.lr_scheduler.step()

            with torch.no_grad():
                val_loss, val_acc_dict = self.evaluation(val_loader, epoch)
            end_time = time.clock()

            if acc_check < val_acc_dict['acc'] or (acc_check == val_acc_dict['acc'] and loss_check > val_loss):  # or
                acc_check = val_acc_dict['acc']
                loss_check = val_loss
                self.checkpoint = epoch
                self.save(self.model_path, epoch)
                patience = self.patience

            patience -= 1
            if not patience:
                break

        with torch.no_grad():

            start_epoch = self.load(self.model_path, self.checkpoint)

            test_loss, test_acc_dict = self.evaluation(test_loader, start_epoch)

            with open(os.path.join(self.model_path, 'predict.txt'), 'a') as f:
                f.write('\t'.join(list(test_acc_dict.keys())) + '\n' + '\t'.join(
                    map(str, list(test_acc_dict.values()))) + '\n')


    def train_batch(self, epoch, nonrumor_loader, rumor_loader):

        train_loss_value = 0
        acc_value = 0

        self.bigcn.train()

        # if len(rumor_loader) / len(nonrumor_loader) > 1:
        #     iterloader = zip(CYCLE(nonrumor_loader), rumor_loader)
        # elif len(rumor_loader) / len(nonrumor_loader) >= 0.5:
        #     iterloader = zip(nonrumor_loader, rumor_loader)
        # else:
        #     iterloader = zip(nonrumor_loader, CYCLE(rumor_loader))

        iterloader = zip(nonrumor_loader, rumor_loader)
        for iter, (Nonrumors, Rumors) in enumerate(iterloader):
            xn, yn, An = Nonrumors
            xr, yr, Ar = Rumors
            xn = xn.to(self.device)
            yn = yn.to(self.device)
            xr = xr.to(self.device)
            yr = yr.to(self.device)
            Ar = Ar.to(self.device)
            An = An.to(self.device)
            x = torch.cat((xr,xn),dim=0)
            y = torch.cat((yr,yn),dim=0)
            A = torch.cat((Ar,An),dim=0)

            # ====================================train Model=============================================

            self.optimizer.zero_grad()

            preds,_ = self.bigcn(x, A)
            loss= self.ce(preds +1e-4, y)

            loss.backward()
            self.optimizer.step()


            train_loss_value += loss.item()
            pred = preds.data.max(1)[1]
            acc = (pred == y).sum() / len(y)
            acc_value += acc.item()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d][model loss: %f] [model acc: %f]"
                % (
                    epoch,
                    self.epochs,
                    iter,
                    len(nonrumor_loader),
                    loss.item(),
                    acc.item()
                )
            )

        train_loss_value = round(train_loss_value / (iter + 1), 4)
        acc_value = round(acc_value / (iter + 1), 4)
        return train_loss_value, acc_value

    def evaluation(self, dataloader, epoch):
        mean_loss = 0
        acc_dict = {}
        acc_dict['acc'] = 0
        acc_dict['P1'] = 0
        acc_dict['R1'] = 0
        acc_dict['F11'] = 0
        acc_dict['P2'] = 0
        acc_dict['R2'] = 0
        acc_dict['F12'] = 0

        self.bigcn.eval()

        for iter, sample in enumerate(dataloader):
            x, y, A = sample
            x = x.to(self.device)
            y_ = y.to(self.device)
            A = A.to(self.device)

            preds,_ = self.bigcn(x, A)
            loss = self.ce(preds, y_)
            mean_loss += loss.item()
            preds = preds.data.max(1)[1].cpu()

            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = evaluationclass(preds, y)

            acc_dict['acc'] += Acc_all
            acc_dict['P1'] += Prec1
            acc_dict['R1'] += Recll1
            acc_dict['F11'] += F1
            acc_dict['P2'] += Prec2
            acc_dict['R2'] += Recll2
            acc_dict['F12'] += F2

        for metric in acc_dict.keys():
            acc_dict[metric] = round(acc_dict[metric] / (iter + 1), 4)
        print(acc_dict.items())
        mean_loss = round(mean_loss / (iter + 1), 4)

        return mean_loss, acc_dict

    def test(self, datapath):

        datafile = os.listdir(datapath)
        testdata = MyDataset(datapath, datafile, self.num_posts)

        test_loader = DataLoaderX(testdata,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_worker,
                                  drop_last=True,
                                  pin_memory=True)
        with torch.no_grad():
            start_epoch = self.load(self.model_path, self.checkpoint)
            acc_test_dict = self.stu_content_test(test_loader)
            with open(os.path.join(self.model_path, 'predict.txt'), 'a') as f:
                f.write('target_domain'+'\t' + str(start_epoch) + '\n' +
                        '\t'.join(list(acc_test_dict.keys())) + '\n' + '\t'.join(
                    map(str, list(acc_test_dict.values()))) + '\n')


        return 0

    def stu_content_test(self, dataloader):
        acc_dict = {}
        acc_dict['acc'] = 0
        acc_dict['P1'] = 0
        acc_dict['R1'] = 0
        acc_dict['F11'] = 0
        acc_dict['P2'] = 0
        acc_dict['R2'] = 0
        acc_dict['F12'] = 0
        for iter, sample in enumerate(dataloader):
            x, y, A = sample
            x = x.to(self.device)
            y_ = y.to(self.device)
            A = A.to(self.device)

            preds,_ = self.bigcn(x, A)
            preds = preds.data.max(1)[1].cpu()

            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = evaluationclass(preds, y)

            acc_dict['acc'] += Acc_all
            acc_dict['P1'] += Prec1
            acc_dict['R1'] += Recll1
            acc_dict['F11'] += F1
            acc_dict['P2'] += Prec2
            acc_dict['R2'] += Recll2
            acc_dict['F12'] += F2

        for metric in acc_dict.keys():
            acc_dict[metric] = round(acc_dict[metric] / (iter + 1), 4)
        print(acc_dict.items())
        return acc_dict

    def save(self, model_path, epoch):
        save_states = {'bigcn': self.bigcn.state_dict(),
                       'optimizer': self.optimizer.state_dict(),
                       'checkpoint': epoch}
        torch.save(save_states, os.path.join(model_path, str(epoch) + '_model_states.pkl'))

        print('save classifer : %d epoch' % epoch)

    def load(self, model_path, checkpoint):

        states_dicts = torch.load(os.path.join(model_path, str(checkpoint) + '_model_states.pkl'))

        self.bigcn.load_state_dict(states_dicts['bigcn'])
        self.optimizer.load_state_dict(states_dicts['optimizer'])
        start_epoch = states_dicts['checkpoint']
        print("load epoch {} success!".format(start_epoch))

        return start_epoch


