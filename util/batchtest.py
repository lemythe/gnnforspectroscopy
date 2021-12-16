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
import pandas as pd
import tensorboardX
import glob 

from LogMetric import AverageMeter
from torch_geometric.data import DataLoader
from specqm9 import QM9spec
from torch_geometric.data import Data
from gmake import graph_make
from transf import nodes,edges 
from GNNnew import GNN
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
from gilmer import Net

torch.backends.cudnn.enabled = False

writer=SummaryWriter()

getattr(tqdm, '_instances', {}).clear()
      

def predict (pred_loader,model,model_name) :

   model.eval()
   rse=[]

   
   f1=open('/home/ks143235/schnet/fincsv/'+str(model_name)+'_preds.txt','ab')
   f2=open('/home/ks143235/schnet/fincsv/'+str(model_name)+'_targs.txt','ab')
   f3=open('/home/ks143235/schnet/fincsv/'+str(model_name)+'_smiles.txt','ab')

 
   for step,batch in enumerate(pred_loader):
        batch=batch.to(device)
        # Compute output
        with torch.no_grad():
           output = model(batch.z,batch.pos)
        output1=output.cpu()
        output1=output1.detach().numpy()
        output1=np.array(output1)
        target1=batch.y
        target1=np.array(target1)
        smile=batch.smiles
        
        rse_diff=relative_difference(output1,target1)
        rse.append(rse_diff)

        np.savetxt(f1,output1)
        np.savetxt(f2,target1)
        np.savetxt(f3,smile,fmt='%5s')


   return rse

def predict_gnn (pred_loader,model,model_name) :

   rse=[]
   model.eval()

   
   f1=open('/home/ks143235/schnet/fincsv/'+str(model_name)+'_preds.txt','ab')
   f2=open('/home/ks143235/schnet/fincsv/'+str(model_name)+'_targs.txt','ab')
   f3=open('/home/ks143235/schnet/fincsv/'+str(model_name)+'_smiles.txt','ab')


   for step,batch in enumerate(pred_loader):
        batch=batch.to(device)
        # Compute output
        with torch.no_grad():
           output = model(batch)
        output1=output.cpu().detach()
        output1=np.array(output1)
        target1=batch.y
        target1=np.array(target1)
        smile=batch.smiles
       
        rse_diff=relative_difference(output1,target1)
        rse.append(rse_diff)

        np.savetxt(f1,output1)
        np.savetxt(f2,target1)
        np.savetxt(f3,smile,fmt='%5s')

        
   return rse


def relative_difference(prediction, target):
    #print(len(prediction))
    dE = 0.1 #how many eV's one dE is
    #print(dE)
    numerator = np.sum(dE*np.power((target-prediction),2))
    denominator = np.sum(dE*target)

    return np.sqrt(numerator)/denominator



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Path to the directory where .xyz files are stored')
    # Positional arguments
   
    # I/O
    parser.add_argument('-p', '--path', metavar='dir', type=str, nargs=1,
                        help='path to store the data (default ./)')
    
    parser.add_argument('-p2', '--path2', metavar='dir', type=str, nargs=1,
                        help='path to store the spectra file data (default ./)')
    
    
    
    parser.add_argument('-folder','--folder',metavar='dir',type=str,nargs=1,help='folder where all model details are')
  
        

    args = parser.parse_args()
    
    if args.path is None:
        args.path = './'
    else:
        args.path = args.path[0]
    
    f_path=args.path
    specfile=args.path2[0]
  
    csv_path=args.folder[0]

    a=np.load(specfile)
    b=a['spectra']
    print(b.shape)
    remove_indices=[0,1,2,3,4,5,6,7,8,10,11,12]
    b = np.delete(b, remove_indices,axis=0)
    seed=4
    print(b.shape)
    flen=np.shape(b)[0]
    idx = np.random.permutation(flen)
    idx = idx.tolist()
    print(flen)


    random.seed(seed)
    all_idx = list(range(flen))
    random.shuffle(all_idx)
      
    test_idx=np.loadtxt('/home/ks143235/schnet/newtest.txt').astype(int)


    data=QM9spec(f_path,b,flen,vertex_transform=nodes, edge_transform=edges ,edge_rep='raw_distance')
    
    test_dataset = data[torch.tensor(test_idx)]
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers = 12)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.chdir(csv_path)
    
    file_list=glob.glob("*")
  #  print(file_list)
    csv_list=glob.glob( "*.csv")
    py_list=glob.glob("*.py")
    txt_list=glob.glob("*.txt")
 #   print(csv_list)
  
    mod_list=[x for x in file_list if x not in csv_list]
    mod_list=[x for x in mod_list if x not in py_list]
    mod_list=[x for x in mod_list if x not in txt_list]
    #mod_list.remove('runs')
    #mod_list.remove('__pycache__')
    
    #csv_name=csv_list[0]
    #df = pd.read_csv(csv_name)
 #   print(mod_list)
    model_array=[]
    rse_array=[]
    for model_load in mod_list:
                 
        if (model_load.find('schnet')!= -1 ):
           model_type= 'schnet'
        
        else:
           model_type='gnn'        
        print(model_load,model_type) 
        model = torch.load(model_load)
   
        if model_type == 'schnet':
           error_rse=predict(test_loader,model,model_load)
        else:
           error_rse=predict_gnn(test_loader,model,model_load)
        
        
        #print(res)
        error_rse=np.array(error_rse)  
        ave=np.average(error_rse)
        rse_array.append(ave)
        model_array.append(model_load)
        
    

        #df.loc[(df['Epochs'] == int(res)) , ['RSE']] =  np.average(error_rse)    

    df = pd.DataFrame(list(zip(model_array, rse_array)),columns =['Model', 'RSE'])    
    new_csv='TestRSE.csv'
    df.to_csv('/home/ks143235/schnet/fincsv/'+new_csv)    
