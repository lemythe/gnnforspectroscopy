#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:40:15 2020

@author: qhl
"""


from __future__ import print_function

import rdkit
import torch
import networkx as nx
import numpy as np
import shutil
import os
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

#Here we define the edge and vertex transforms to encode the graph as a vector to add them in a list format for dataset class 'QM9'
#nodes,iter() of Networkx graph iterates over the nodes
#Appending node forms

bond_types=[rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC]
hbd_types=[rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,rdkit.Chem.rdchem.HybridizationType.SP,rdkit.Chem.rdchem.HybridizationType.SP2, rdkit.Chem.rdchem.HybridizationType.SP3]                                            


def nodes(g):
  
    h=[]
    for n_val,d_val in list(g.nodes(data=True)) :
        h_temp=[]
        
        h_temp.append(d_val['a_num'])
        h_temp.append(d_val['acceptor'])
        h_temp.append(d_val['donor'])
        h_temp.append(int(d_val['aromatic']))
       
        h_temp.append(hbd_types.index(d_val['hybridization']))
        h_temp.append(d_val['num_h'])
    
        h.append(h_temp)    
    return h

def edges(g,edge_rep='raw_distance'):
    
    remove_edges = []
    e={}
    for n1, n2, d in list(g.edges(data=True)):
        e_t = []
        if edge_rep == 'raw_distance':
                if d['b_type'] is None:
                    remove_edges += [(n1, n2)]
                else:
                   # e_t.append(int(d['inring']))
                    #e_t.append(int(d['conjugated']))
                    e_t.append(d['distance'])
                    
                   
                    e_t.append(bond_types.index(d['b_type']))
                    #print(e_t)
        if e_t:
                e[(n1, n2)] = e_t
    for ed in remove_edges:
        g.remove_edge(*ed)
    return g,nx.to_numpy_matrix(g), e   


