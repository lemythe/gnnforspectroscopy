import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch.utils.data
import os,sys
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
#import tensorboardX
#from LogMetric import AverageMeter
#from torch_geometric.data import DataLoader
#from specqm9 import QM9spec
#from torch_geometric.data import Data
#from gmake import graph_make
#from transf import nodes,edges 
#from GNNnew import GNN
#from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
#from gilmer import Net

torch.backends.cudnn.enabled = False

#writer=SummaryWriter()

getattr(tqdm, '_instances', {}).clear()
      

def predict (pred_loader,model) :


   preds=[]
   targs=[]
   model.eval()

   #f1=open('preds.txt','ab')
   #f2=open('targs.txt','ab')
   f3=open('train_smiles.txt','ab')
   for step,batch in enumerate(pred_loader):
        batch=batch.to(device)
        # Compute output
        #with torch.no_grad():
        #   output = model(batch.z,batch.pos)
        #output1=output.cpu()
        #target1=batch.y
        smile=batch.smiles
        #target1=target1.cpu()
        #preds.append(output1)
        #targs.append(target1)
        #np.savetxt(f1,output1.detach().numpy())
        #np.savetxt(f2,target1)
        np.savetxt(f3,smile,fmt='%5s')


#   f1.close()
#   f2.close()
   f3.close()
def predict_gnn (pred_loader,model) :


   preds=[]
   targs=[]
   model.eval()

   f1=open('preds.txt','ab')
   f2=open('targs.txt','ab')
   f3=open('smiles.txt','ab')
   for step,batch in enumerate(pred_loader):
        batch=batch.to(device)
        # Compute output
        with torch.no_grad():
           output = model(batch)
        output1=output.cpu()
        target1=batch.y
        smile=batch.smiles
        #target1=target1.cpu()
        preds.append(output1)
        targs.append(target1)
        np.savetxt(f1,output1.detach().numpy())
        np.savetxt(f2,target1)
        np.savetxt(f3,smile,fmt='%5s')


   f1.close()
   f2.close()




if __name__ == '__main__':


    #parser = argparse.ArgumentParser(description='Path to the directory where .xyz files are stored')
    # Positional arguments
   
    # I/O
    #parser.add_argument('-p', '--path', metavar='dir', type=str, nargs=1,
    #                    help='path to store the data (default ./)')
    
    #parser.add_argument('-p2', '--path2', metavar='dir', type=str, nargs=1,
    #                    help='path to store the spectra file data (default ./)')

    #args = parser.parse_args()
    
    #if args.path is None:
    #    args.path = './'
    #else:
    #    args.path = args.path[0]
    
    #f_path=args.path
    #specfile=args.path2[0]
   
    #flen=132531 
    #idx = np.random.permutation(flen)
    #idx = idx.tolist()
    
    #a=np.load(specfile)
    #b=a['spectra']
    #seed=4
    
    #random.seed(seed)
    #all_idx = list(range(flen))
    #random.shuffle(all_idx)
      
    #test_idx=np.loadtxt('/home/ks143235/schnet/newtrain.txt').astype(int)

    #print('loaded')
    #data=QM9spec(f_path,b,flen,vertex_transform=nodes, edge_transform=edges ,edge_rep='raw_distance')
    
    #test_dataset = data[torch.tensor(test_idx)]
    #test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers = 12)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = torch.load('/home/ks143235/schnet/finalmodels/gcnbatch50gemb500cycles5aemb60299')
   
    print('before ttrain')
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    #predict(test_loader,model)
        
