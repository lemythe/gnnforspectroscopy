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
import itertools,csv
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter

from specqm9 import QM9spec
from GNNnew import GNN
from transf import nodes,edges
from gilmer import Net
from Schnetspec import SchNetspec
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


torch.backends.cudnn.enabled = False

writer=SummaryWriter()

getattr(tqdm, '_instances', {}).clear()



def train(train_loader,model_type, model, criterion, optimizer, epoch,device):

    model.train()
    loss_all=0

    for step,batch in enumerate(tqdm(train_loader, desc=" Training Iteration")):

        batch=batch.to(device)

        optimizer.zero_grad()

        if model_type=='schnet':
        	output=model(batch.z,batch.pos,batch.batch)
        else:
        	output = model(batch)              
        target = torch.Tensor(batch.y).to(device)

        #if epoch==500 :
        #  __import__("pdb").set_trace()
        train_loss = criterion(output.double(), target.double())

        loss_all += train_loss.data * batch.num_graphs

        train_loss.backward()
        optimizer.step()
     
    return (loss_all)

def validate(val_loader,model_type, model,criterion,validator, epoch,device):


    losses_all=0   
    rse_all=0
    # switch to evaluate mode
    model.eval()

    for step,batch in enumerate(tqdm(val_loader, desc="Validation Iteration")):
        
        batch=batch.to(device)
        # Compute output
        if model_type=='schnet':
        	output=model(batch.z,batch.pos,batch.batch)
        else:
        	output = model(batch)
        target = torch.Tensor(batch.y).to(device)
        
        val_loss = criterion(output.double(), target.double())
        rse_val=validator(output,target)
      
        rse_all+=rse_val.data
        losses_all += val_loss.data * batch.num_graphs

            
    return losses_all,rse_all       


def predict (pred_loader,model_type,model) :


   preds=[]
   targs=[]
   rse=[]
   model.eval()

#   f1=open('preds.txt','ab')
#   f2=open('targs.txt','ab')
#   f3=open('smiles.txt','ab')
   for step,batch in enumerate(pred_loader):
        batch=batch.to(device)
        # Compute output
        with torch.no_grad():
           if model_type=='schnet':
           	output = model(batch.z,batch.pos)
           else :
           	output=model(batch)   
        output1=output.cpu()
        target1=np.array(batch.y)
        #print(type(output1))
        #print(type(target1)) 
        preds.append(output1.detach().numpy())
        #print(type(preds))
       
        targs.append(target1)
 
   return preds,targs

def relative_difference(prediction, target):
    print(len(prediction))
    dE = 30/len(prediction) #how many eV's one dE is
    #print(dE)
    numerator = np.sum(dE*np.power((target-prediction),2))
    denominator = np.sum(dE*target)
    
    return np.sqrt(numerator)/denominator

def torch_rel_diff(prediction,target):
    dE=0.01
    #print(target-prediction)
    numerator=dE*torch.sum(torch.pow((target-prediction),2),1)
    denominator=dE*torch.sum(target,1)
    rse=torch.sqrt(numerator)/denominator
    #print(torch.sum(rse))
    return torch.sum(rse)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Path to the directory where .xyz files are stored')
    # Positional arguments
   
    # I/O
    parser.add_argument('-p', '--path', metavar='dir', type=str, nargs=1,
                        help='path to store the data (default ./)')
    
    parser.add_argument('-p2', '--path2', metavar='dir', type=str, nargs=1,
                        help='path to store the spectra file data (default ./)')
    
    parser.add_argument('-gnn','--gnn', type=str,nargs=1, help='gnn type')

    parser.add_argument('-cycles','--cycles',type=int,nargs=1,help='number of mp cycles')

    parser.add_argument('-gsize','--gsize',type=int,nargs='?',default=0,help='graph embedding')

    parser.add_argument('-asize','--asize',type=int,nargs=1,help='atom embedding')

    parser.add_argument('-bsize','--bsize',type=int,nargs=1,help='batch size')

    parser.add_argument('-setsteps','--setsteps',type=int,nargs='?', default=0,help='set2setsteps')
    args = parser.parse_args()
    
    if args.path is None:
        args.path = './'
    else:
        args.path = args.path[0]
    
    f_path=args.path
    specfile=args.path2[0]
    type_model = args.gnn[0]
    cycles=args.cycles[0]
    graph_size=args.gsize 
    asize=args.asize[0]
    bsize=args.bsize[0]
    set_steps=args.setsteps
    #flen=132519
     
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
    print(max(all_idx))
    random.shuffle(all_idx)
    
    frac_train=0.9
    frac_valid=0.05
    
    train_idx=np.loadtxt('/home/ks143235/schnet/newtrain.txt').astype(int)
    valid_idx=np.loadtxt('/home/ks143235/schnet/newvalid.txt').astype(int)
    test_idx=np.loadtxt('/home/ks143235/schnet/newtest.txt').astype(int)

    num_mols=flen
    #train_idx = all_idx[:int(frac_train * num_mols)]
    #valid_idx = all_idx[int(frac_train * num_mols):int(frac_valid * num_mols)
    #                                               + int(frac_train * num_mols)]
    #test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols):]

    data=QM9spec(f_path,b,flen,vertex_transform=nodes, edge_transform=edges ,edge_rep='raw_distance')
   

   # train_idx=torch.from_numpy(train_idx)

    train_dataset = data[torch.tensor(train_idx)]
    val_dataset = data[torch.tensor(valid_idx)]
    test_dataset = data[torch.tensor(test_idx)]
    
    
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, num_workers = 12)
    valid_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=True, num_workers = 12)

    #choose models
    print(type_model)
    #conv = GNN(300,5,,'gcn','True','False',0.0,'last','sum')  
    #conv1=Net()
    #conv2=SchNetspec(hidden_channels=300, num_filters=200, num_interactions=6,
    #                num_gaussians=50, cutoff=10.0)
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    if type_model=='gcn' or type_model=='gin':
	    print("here")
	    conv = GNN(300,cycles,graph_size,asize,type_model,'True','False',0.0,'last','sum')
	    model = conv.to(device)
    elif type_model=='mpnn':
	    conv1=Net(cycles,asize,set_steps)   
	    model=conv1.to(device)
    else :
            conv2=SchNetspec(hidden_channels=graph_size,num_filters=200,num_interactions=cycles,a_emb=asize,num_gaussians=50,cutoff=10.0)
            model=conv2.to(device)

       
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.9, patience=5,
                                                       min_lr=0.00001)
    print('The length of train data is :',len(train_dataset))
    print('The length of validation data is:', len(val_dataset))
    criterion=nn.MSELoss()
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    loss_train=[]
    epoch_rse=[]   
    rse_sav=[]
    epoch_val=[]
    valid_loss=[]      
   
    if type_model=='mpnn': 
    	model_save=type_model+'batch'+str(bsize)+'aemb'+str(asize)+'cycles'+str(cycles)+'setsteps'+str(set_steps)     
    else :
        model_save=type_model+'batch'+str(bsize)+'gemb'+str(graph_size)+'cycles'+str(cycles)+'aemb'+str(asize)


    for epoch in range(300):
        lr = scheduler.optimizer.param_groups[0]['lr']
        #print(lr)
        train_loss = (train(train_loader,type_model,model,criterion,optimizer,epoch,device)/(len(train_dataset))) 
        #print(train_loss)
        writer.add_scalar('train_loss', train_loss,epoch)
        

        val_loss,rse_div=validate(valid_loader,type_model,model,criterion,torch_rel_diff,epoch,device)
        val_loss=(val_loss/len(val_dataset))
        rse_div=(rse_div/len(val_dataset))
        #print(val_loss,rse_div)
        scheduler.step(val_loss)
        writer.add_scalar('val_loss', val_loss,epoch)
        print("Epoch number:",epoch)
        
        if ((epoch+1)%50==0):
          #   print("in loop")
             rse_sav.append(np.round(rse_div.item(),5))
             epoch_val.append(epoch)
             valid_loss.append(np.round(val_loss.item(),5))
             t = time.localtime()
             timestamp = time.strftime('%b-%d-%Y_%H%M', t)
             torch.save(model,model_save+str(epoch))

    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t) 
            	        
    Outfile=open(model_save+'.csv','w',newline="")
    Outfile.write("Epochs,Val_loss,RSE\n")
    rows=itertools.zip_longest(epoch_val, valid_loss, rse_sav)
    c=csv.writer(Outfile,delimiter=',')
    c.writerows(rows)
            
    torch.save(model, model_save+str(epoch))
    torch.cuda.empty_cache()
    
         
