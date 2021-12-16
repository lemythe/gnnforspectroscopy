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
    

    # switch to train mode
    model.train()
    loss_all=0
    end = time.time()
    for step,batch in enumerate(tqdm(train_loader, desc=" Training Iteration")):

        batch=batch.to(device)
        # Measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        # Compute output
       # print((batch.z).shape)
        if model==conv2:
        	output=model(batch.z,batch.pos,batch.batch)
        else:
        	output = model(batch)              
        target = torch.Tensor(batch.y).to(device)
#        print(target.shape,output.shape)
        #target=(batch.y)
        #if epoch==500 :
        #  __import__("pdb").set_trace()
        train_loss = criterion(output.double(), target.double())

        loss_all += train_loss.data * batch.num_graphs
        # compute gradient and do SGD step
        train_loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
     
    return (loss_all)

def validate(val_loader, model, criterion, epoch,device):
    batch_time = AverageMeter()

    losses_all=0   

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for step,batch in enumerate(tqdm(val_loader, desc="Validation Iteration")):
        
        batch=batch.to(device)
        # Compute output
        if model==conv2:
        	output=model(batch.z,batch.pos,batch.batch)
        else:
        	output = model(batch)
        target = torch.Tensor(batch.y).to(device)
        
        val_loss = criterion(output.double(), target.double())

        losses_all += val_loss.data * batch.num_graphs
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
            
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
    seed=45678
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

    train_dataset = data[torch.tensor(train_idx)]
    val_dataset = data[torch.tensor(valid_idx)]
    test_dataset = data[torch.tensor(test_idx)]
    
    
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers = 12)
    valid_loader = DataLoader(val_dataset, batch_size=50, shuffle=True, num_workers = 12)

    #choose models
    conv = GNN(300,5,300,'gin','True','False',0.0,'last','sum')  
    conv1=Net()
    conv2=SchNetspec(hidden_channels=300, num_filters=200, num_interactions=6,
                     num_gaussians=50, cutoff=10.0)
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = conv.to(device)
    #model=conv1.to(device)
    #model=conv2.to(device)

       
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.9, patience=5,
                                                       min_lr=0.00001)
    print('The length of train data is :',len(train_dataset))
    print('The length of validation data is:', len(val_dataset))
    criterion=nn.MSELoss()
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)


    loss_train=[]
    
    for epoch in range(300):
        lr = scheduler.optimizer.param_groups[0]['lr']
        #print(lr)
        train_loss = (train(train_loader,model,criterion,optimizer,epoch,device)/(len(train_dataset))) 
        print(train_loss)
        writer.add_scalar('train_loss', train_loss,epoch)
        

        val_loss=(validate(valid_loader,model,criterion,epoch,device)/len(val_dataset))
        print(val_loss)
        scheduler.step(val_loss)
        writer.add_scalar('val_loss', val_loss,epoch)
 
 
    torch.save(model, 'trygin')
    torch.cuda.empty_cache()
    
         
