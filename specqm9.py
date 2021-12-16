# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:43:14 2020

@author: KANISHK
"""

import os.path as osp
import torch
import networkx as nx
import numpy as np
import argparse
import time
import random
import os,sys
from itertools import repeat, product, chain
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from torch_geometric.data import Dataset

import rdkit
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig


from gmake import graph_make
from transf import nodes,edges



class QM9spec(InMemoryDataset):
    
    def __init__(self,
                  root,
                  #data = None,
                  #slices = None,
                  specarr,num,vertex_transform=nodes,edge_transform=edges,
                  target_transform=None, edge_rep='distance',
                  transform=None,
                  pre_transform=None,
                  pre_filter=None,empty=False
                  ):
        
        self.root = root
        self.specarr = specarr
        self.number=num
        self.vertex_transform=vertex_transform
        self.edge_transform=edge_transform
        self.target_transform=target_transform
        self.edge_rep=edge_rep
        #self.file=f_path
        super(QM9spec, self).__init__(root, transform, pre_transform, pre_filter)

        
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

        
    @property
    def raw_file_names(self):
        
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list
    

    @property
    def processed_file_names(self):
        
        return 'schnetupdate.pt'
    
    def mean(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].mean().item()

    def std(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].std().item()

    
    def process (self) :

        input_path = (self.raw_dir)[:-4]
        print(input_path)
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]

        allids=[int(x[:-4]) for x in files]

        data_list = []

        print("Before loop")
        for i in range(self.number):
            print(i)
            pro=allids[i]
            
            #Generating a graph and target list from graph make by entering a file
            graph,target,sm,molid,cords,a_num= graph_make(os.path.join(input_path, files[i]),self.specarr[pro])
            #print(cords)
            z = torch.tensor(a_num, dtype=torch.long)
            pos = torch.tensor(cords, dtype=torch.float)

            if self.vertex_transform is not None:
                h=self.vertex_transform(graph)
            
            if self.edge_transform is not None :
                graph,gmat,e=self.edge_transform(graph,self.edge_rep)
            
            if self.target_transform is not None :
                target=self.target_transform(target)

            ed,edattr=nx_to_graph_data_obj_simple(graph,e) 
            
            x=torch.tensor(h,dtype=torch.float)
    
            y=torch.tensor(target,dtype=torch.float)
            
            data = Data(x=x, edge_index=ed, edge_attr=edattr,y=target,idx=i,f_id=pro,mol=molid,smiles=sm,z=z,pos=pos)
            data_list.append(data) 
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])
        
        torch.save(self.collate(data_list), self.processed_paths[0])


def nx_to_graph_data_obj_simple(G,e):
    """
    Converts nx graph to pytorch geometric Data object. Assume node indices
    are numbered from 0 to num_nodes - 1. NB: Uses simplified atom and bond
    features, and represent as indices. NB: possible issues with
    recapitulating relative stereochemistry since the edges in the nx
    object are unordered.
    :param G: nx graph obj
    :return: pytorch geometric Data object
    """
    
    # bonds
    num_bond_features = 4  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            
            edge_feature=e[(i, j)]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        edge_attr = torch.tensor(np.array(edge_features_list),
                                  dtype=torch.long)

    return edge_index,edge_attr 

    
             
