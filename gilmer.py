# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 15:09:38 2020

@author: kanishk
"""

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)


import torch
import torch.nn.functional as F

import torch.utils.data


from torch.nn import Sequential, Linear, ReLU, GRU

from torch_geometric.nn import NNConv, Set2Set



class Net(torch.nn.Module):
    def __init__(self, num_layer=5,dim=100,steps=3):
        super(Net, self).__init__()
        self.lin0 = torch.nn.Linear(6, dim)

        self.num_layer=num_layer
        nn = Sequential(Linear(2, 128), ReLU(), Linear(128, dim*dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=steps)
        self.lin1 = torch.nn.Linear(2*dim,dim)
        self.lin2 = Sequential(Linear(dim, 300))

    def forward(self, data):
        
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.num_layer):
            data.edge_attr=data.edge_attr.float()
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        
        out = self.lin1(out)
        
        #out= torch.nn.Sigmoid()(out)
        out = torch.nn.LeakyReLU(0.01)(self.lin2(out))
        return out
    

