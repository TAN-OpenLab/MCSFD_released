import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pickle

class BiGraphDataset(Dataset):

    def __init__(self, filepath,file_list,num_nodes):
        # 获得训练数据的总行
        self.filepath = filepath
        self.file_list = file_list
        self.number = len(self.file_list)
        self.num_nodes = num_nodes

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        file = self.file_list[idx]
        x,_, A,_, y = pickle.load(open(os.path.join(self.filepath, file), 'rb'), encoding='utf-8')

        self.number = len(x)

        A = torch.tensor(A, dtype=torch.long)
        A = torch.transpose(A,dim0=1,dim1=0)
        x1 = torch.tensor(x, dtype=torch.float32)
        if(x1.size()[0] >self.num_nodes):
            mask = torch.logical_and(A[0]< self.num_nodes, A[1] < self.num_nodes)
            A1 = A[:,mask]
            x2 =x1[:self.num_nodes,:]
        else:
            x2 = x1.clone()
            A1= A.clone()
        y = torch.tensor(y, dtype=torch.long)


        return Data(x=x2, edge_index=A1, y = y, root=x2[0,:], rootindex=torch.LongTensor([0]))


