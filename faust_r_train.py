# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:40:56 2021

@author: Michael
"""
import os
import sys
from itertools import permutations
import os.path as osp
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import scipy.io as sio

sys.path.append(osp.join(os.getcwd(),'src'))
import diffusion_net

from matching_dataset import MatchingDataset


# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default = 'hks')
args = parser.parse_args()


# system things
device = torch.device('cuda:0')
dtype = torch.float32


# model 
# input_features = args.input_features # one of ['xyz', 'hks']
input_features = 'hks'
k_eig = 128

# training settings
train = not args.evaluate
n_epoch = 2
lr = 1e-3

# Important paths
base_path = osp.dirname(__file__)
dataset_path = osp.join(base_path, 'data','faust_5k')
pretrain_path = osp.join(dataset_path, "pretrained_models/faust_{}_4x128.pth".format(input_features))
# model_save_path = os.path.join(dataset_path, 'saved_models','t_hk1104_faust_{}_4x128.pth'.format(input_features))


# Load the train dataset
if train:
    train_dataset = MatchingDataset(dataset_path, train=True, k_eig=k_eig, use_cache=True)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    now = datetime.now()
    folder_str = now.strftime("%Y_%m_%d__%H_%M_%S")
    model_save_dir=osp.join(dataset_path,'save_models',folder_str)
    diffusion_net.utils.ensure_dir_exists(model_save_dir)

# === Create the model

C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features



model = diffusion_net.layers.RFMNet(C_in=C_in,C_out=256)

model = model.to(device)

if not train:
    # load the pretrained model
    print("Loading pretrained model from: " + str(pretrain_path))
    model.load_state_dict(torch.load(pretrain_path))

# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train_epoch(epoch):
    # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0.0
    total_num = 0
    for data in tqdm(train_loader):

        # Get data
        descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y=data
        
        # Move to device
        descs_x=descs_x.to(device)
        massvec_x=massvec_x.to(device)
        evals_x=evals_x.to(device)
        evecs_x=evecs_x.to(device)
        gs_x=gs_x.to(device)
        gradX_x=gradX_x.to(device) 
        gradY_x=gradY_x.to(device) #[N,N]

        descs_y=descs_y.to(device)
        massvec_y=massvec_y.to(device)
        evals_y=evals_y.to(device)
        evecs_y=evecs_y.to(device)
        gs_y=gs_y.to(device)
        gradX_y=gradX_y.to(device)
        gradY_y=gradY_y.to(device)

        # Apply the model
        loss, C = model(descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y)

        # Evaluate loss
        loss.backward()
        
        # track accuracy
        total_loss+=loss.item()
        total_num += 1

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

        if total_num%100==0:
            print('Iterations: {:02d}, train loss: {:.4f}'.format(total_num, total_loss / total_num))
            total_loss=0.0
            total_num=0

if train:
    print("Training...")

    for epoch in range(n_epoch):
        train_acc = train_epoch(epoch)
        
        model_save_path=osp.join(model_save_dir,'ckpt_ep{}.pth'.format(n_epoch))
        torch.save(model.state_dict(), model_save_path)

    print(" ==> saving last model to " + model_save_path)
    torch.save(model.state_dict(), model_save_path)
    

# Test
# Load the test dataset
test_dataset = MatchingDataset(dataset_path, train=False, k_eig=k_eig, use_cache=True)
test_loader = DataLoader(test_dataset, batch_size=None)

results_dir=osp.join(model_save_dir,'hks_results')
diffusion_net.utils.ensure_dir_exists(results_dir)

        
file=osp.join(dataset_path,'files_test.txt')
with open(file, 'r') as f:
    names = [line.rstrip() for line in f]

combinations = list(permutations(range(len(names)), 2))

model.eval()
with torch.no_grad():
    count=0
    for data in test_loader:
        descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y=data
        
        # Move to device
        descs_x=descs_x.to(device)
        massvec_x=massvec_x.to(device)
        evals_x=evals_x.to(device)
        evecs_x=evecs_x.to(device)
        gs_x=gs_x.to(device)
        gradX_x=gradX_x.to(device)
        gradY_x=gradY_x.to(device)

        descs_y=descs_y.to(device)
        massvec_y=massvec_y.to(device)
        evals_y=evals_y.to(device)
        evecs_y=evecs_y.to(device)
        gs_y=gs_y.to(device)
        gradX_y=gradX_y.to(device)
        gradY_y=gradY_y.to(device)

    
        # Apply the model
        C, T = model.model_test_opt(descs_x,massvec_x,evals_x,evecs_x,gs_x,gradX_x,gradY_x,\
            descs_y,massvec_y,evals_y,evecs_y,gs_y,gradX_y,gradY_y)
        
        idx1,idx2=combinations[count]
        count+=1

        results_path=osp.join(results_dir,names[idx1]+'_'+names[idx2]+'.mat')
        sio.savemat(results_path, {'C':C.to('cpu').numpy().astype(np.float32),
                                   'T':T.to('cpu').numpy().astype(np.int64)+1}) # T: convert to matlab index
