import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from tqdm import tqdm
import numpy as np
import argparse
import torch.utils.data
import time
import torch
import torch.nn as nn
import random
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter

from LogMetric import AverageMeter
from specqm9 import QM9spec
from GNNnew import GNN
from transf import nodes,edges
from gilmer import Net
from Schnetspec import SchNetspec


torch.backends.cudnn.enabled = False

writer=SummaryWriter()

getattr(tqdm, '_instances', {}).clear()



def train(train_loader, model, criterion, optimizer, epoch,device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    #losses = AverageMeter()
    
    f1=open('train_smiles.txt','ab')
    # switch to train mode
    model.train()
    loss_all=0
    end = time.time()
    for step,batch in enumerate(tqdm(train_loader, desc=" Training Iteration")):

        batch=batch.to(device)
        sm=batch.smiles
        np.savetxt(f1,sm,fmt='%5s')       

    return loss_all

def validate(val_loader, model, criterion, epoch,device):
    model.eval()
    losses_all=0
    f2=open('valid_smiles.txt','ab')
    for step,batch in enumerate(tqdm(val_loader, desc="Validation Iteration")):
       
        batch=batch.to(device)
        # Compute output
        sm=batch.smiles
        np.savetxt(f2,sm,fmt='%5s')
        
            
    return losses_all       

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Path to the directory where .xyz files are stored')
    # Positional arguments
   
    # I/O
    parser.add_argument('-p', '--path', metavar='dir', type=str, nargs=1,
                        help='path to store the data (default ./)')
    
    parser.add_argument('-p2', '--path2', metavar='dir', type=str, nargs=1,
                        help='path to store the spectra file data (default ./)')

    args = parser.parse_args()
    
    if args.path is None:
        args.path = './'
    else:
        args.path = args.path[0]
    
    f_path=args.path
    specfile=args.path2[0]

    num_mols=132519
    
    #idx = np.random.permutation(num_mols)
    #idx = idx.tolist()
    #print(max(idx))
    a=np.load(specfile)
    b=a['spectra']
    seed=90
    remove_indices=[0,1,2,3,4,5,6,7,8,10,11,12]
    b = np.delete(b, remove_indices,axis=0)
    random.seed(seed)
    all_idx = list(range(num_mols))
    print(max(all_idx))
    random.shuffle(all_idx)
    
    frac_train=0.9
    frac_valid=0.05
    
    #train_idx=np.loadtxt('trainindices.txt').astype(int)
    #valid_idx=np.loadtxt('valindices.txt').astype(int)
    #test_idx=np.loadtxt('testindices.txt').astype(int)

   
    train_idx = all_idx[:int(frac_train * num_mols)]
    valid_idx = all_idx[int(frac_train * num_mols):int(frac_valid * num_mols)
                                                   + int(frac_train * num_mols)]
    test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols):]

    data=QM9spec(f_path,b,num_mols,vertex_transform=nodes, edge_transform=edges ,edge_rep='raw_distance')
  
    np.savetxt('upd_train_ids.txt',train_idx,delimiter=',')
    np.savetxt('upd_valid_ids.txt',valid_idx,delimiter=',')
    np.savetxt('upd_test_ids.txt',test_idx,delimiter=',')

 #   train_idx=torch.from_numpy(train_idx)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_dataset = data[torch.tensor(train_idx)]
    val_dataset = data[torch.tensor(valid_idx)]
    test_dataset = data[torch.tensor(test_idx)]
   
    f1=open('train_smiles.txt','ab')
    f2=open('valid_smiles.txt','ab')
   
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers = 12)
    valid_loader = DataLoader(val_dataset, batch_size=200, shuffle=True, num_workers = 12)

    
        
    for step,batch in enumerate(train_loader):

        batch=batch.to(device)
        sm=batch.smiles
        np.savetxt(f1,sm,fmt='%5s')

 
    for step,batch in enumerate(valid_loader):

        batch=batch.to(device)
        sm1=batch.smiles
        np.savetxt(f2,sm1,fmt='%5s')
     
