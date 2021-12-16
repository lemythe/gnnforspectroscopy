#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:28:19 2020

@author: qhl
"""


import networkx as nx 
import numpy as np
import random
import argparse

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

import numpy as np
import os
from os import listdir
from os.path import isfile, join


np.random.seed(2)

def transform_qm9(file_path,specfile) :
    
    a=np.load(specfile)
    b=a['spectra']
                
    files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    
    graphs=[]
    labels=[]
    for i in range(0,2):
        
        g,l,sm=graph_make(join(file_path, files[i]),b[i])
        
        graphs +=[g]
        labels.append(l)
    return graphs,labels,sm

#Initialises the graph class type object for QM9 dataset.        
def init_graph(prop,specarr):
    
    prop = prop.split()
    g_tag = prop[0]
    
    E_arr=specarr
    
    return nx.Graph(tag=g_tag),E_arr

     
def graph_make(file,specarr) :
            
    with open(file,'r') as f :
        
        #First line of xyz file tells number of atoms
        n_atoms=int(f.readline())
        
        #Second line has list of all calculated properties
        properties=f.readline()
        pr=properties.split()
        id=pr[0]

        #Encode the properties into graph and labels.
        # g here stores the Graph properties. such as lumo zpve and so on. l saves its labels.
        #Here we store atom coordinates
        atom_properties=[]
        for j in range(n_atoms) :
            atom_prop=f.readline()
            atom_prop=atom_prop.replace('.*^', 'e') 
            atom_prop=atom_prop.replace('*^', 'e')
            atom_prop=atom_prop.split()
            atom_properties.append(atom_prop)

        #Here we store the frequencies mentioned in xyz file
        coords=np.array(atom_properties)
        pos=[]
        for i in range(n_atoms):
            lax=coords[i][1:4]
            pos.append(lax)
        pos=np.array(pos,dtype=float)       
        #We store SMILES strings finally.
        sm=f.readline()
        sm=sm.split()
        sm=sm[0]
        #print(sm)
        mol=Chem.MolFromSmiles(sm)
        mol=Chem.AddHs(mol)
        fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        feats = factory.GetFeaturesForMol(mol)
        g,l= init_graph(properties,specarr)
        atomic_number=[]  
        # We have saved the smile string as a molecule and now we add its features using 
        # properties of the 'mol' class.
        #Now to the graph we had created above, we add nodes as molecules.
        
        for i in range(0, mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
                
            g.add_node(i,a_sym=atom_i.GetSymbol(),a_num=atom_i.GetAtomicNum(),acceptor=0, donor=0,
                        aromatic=atom_i.GetIsAromatic(), hybridization=atom_i.GetHybridization(),
                          num_h=atom_i.GetTotalNumHs(includeNeighbors=True), coord=np.array(atom_properties[i][1:4]).astype(np.float))
            atomic_number.append(atom_i.GetAtomicNum())
            
        #Use RDKit feats to get Donor Acceptor features add to Graph node i 
            
        for i in range(0, len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.nodes[i]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.nodes[i]['acceptor'] 
            
            #Here we make edges for the molecular graph
            
        for i in range(0, mol.GetNumAtoms()):
            for j in range(0, mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j, b_type=e_ij.GetBondType(),conjugated=e_ij.GetIsConjugated(),inring=e_ij.IsInRing(),
                               distance=np.linalg.norm(g.nodes[i]['coord']-g.nodes[j]['coord']))
                else:
                        # Unbonded
                    g.add_edge(i, j, b_type=None,conjugated=None,
                               distance=np.linalg.norm(g.nodes[i]['coord'] - g.nodes[j]['coord']))
        
            
    return g , l,sm,id,pos,atomic_number
        
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Path to the directory where .xyz files are stored')
    # Positional arguments
   
    # I/O
    parser.add_argument('-p', '--path', metavar='dir', type=str, nargs=1,
                        help='path to store the data (default ./)')

    parser.add_argument('-p2', '--path2', metavar='dir', type=str, nargs=1,
                        help='path to store the data (default ./)')



    args = parser.parse_args()
    
    if args.path is None:
        args.path = './'
    else:
        args.path = args.path[0]
    
    args.path2=args.path2[0]
    print(args.path2)
    
    Graphs,Labels=transform_qm9(args.path,args.path2)
    
    print(Graphs[1].nodes('a_sym'))
    #print(Labels[0])
