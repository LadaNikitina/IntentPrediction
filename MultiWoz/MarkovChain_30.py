#!/usr/bin/env python
# coding: utf-8

# # Dialogue Graph Auto Construction based on data with a regular structure
# 

# Goal: Extract regular structures from the data by building a dialogue graph
#     
# Tasks: 
# * Cluster dialog data using embeddings of pre-trained models (BERT, ConveRT, S-BERT…)
# * Evaluate the quality of clustering using intent’s labeling of Multi-WoZ dataset 
# * Linking clusters of dialogs using naive approaches (Estimation of Probabilities by Frequency Models)
# * Try other approaches (Deep Neural Networks) for linking clusters and improve the naive approach
# 

# In[1]:


from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from torch import nn
from torch_geometric.data import Data
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from dgl.dataloading import GraphDataLoader
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import networkx as nx
import sys
import os
import torch
import math
import tensorflow as tf
import random
import dgl
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import torch.nn as nn


# In[2]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(torch.cuda.device_count())


# In[4]:

# In[5]:


first_num_clusters = 200
second_num_clusters = 30


# In[6]:


import sys
sys.path.insert(1, '/cephfs/home/ledneva/multiwoz/common_utils/')


# In[19]:


from preprocess import Clusters, get_accuracy_k, get_all_accuracy_k

file = open("MarkovChain_distilroberta.txt", "w")

def create_graph(num_nodes, train_user_data, train_system_data, 
              test_user_data, test_system_data, 
              train_dataset, test_dataset):
    w_user_sys = np.zeros((num_nodes + 1, num_nodes))
    w_sys_user = np.zeros((num_nodes + 1, num_nodes))
    
    ind_user = 0
    ind_system = 0

    for obj in train_dataset:
        pred_cluster = num_nodes

        for j in range(len(obj["utterance"])):
            if obj['speaker'][j] == 0:
                cur_cluster = train_user_data["cluster"][ind_user]
                
                ind_user += 1

                w_sys_user[pred_cluster][cur_cluster] += 1
                pred_cluster = cur_cluster
            else:
                cur_cluster = train_system_data["cluster"][ind_system]

                ind_system += 1

                w_user_sys[pred_cluster][cur_cluster] += 1
                pred_cluster = cur_cluster
            
    for i in range(num_nodes + 1):
        sum_i_user_sys = sum(w_user_sys[i])
        sum_i_sys_user = sum(w_sys_user[i])
        
        if sum_i_user_sys != 0:
            w_user_sys[i] /= sum_i_user_sys
                
        if sum_i_sys_user != 0:    
            w_sys_user[i] /= sum_i_sys_user 
    
    sys_test = []
    user_test = []
        
    ind_user = 0
    ind_system = 0
    
    for obj in test_dataset:
        pred_cluster = num_nodes

        for j in range(len(obj["utterance"])):
            if obj['speaker'][j] == 0:
                cur_cluster = test_user_data["cluster"][ind_user]
                user_test.append(w_sys_user[pred_cluster])
                ind_user += 1
                pred_cluster = cur_cluster
            else:
                cur_cluster = test_system_data["cluster"][ind_system]
                sys_test.append(w_user_sys[pred_cluster])
                ind_system += 1
                pred_cluster = cur_cluster
                
    file.write("USER metric\n")

    file.write(f"Acc@1: {get_accuracy_k(1, clusters.test_user_df, user_test, clusters.test_dataset, 0)}\n")
    file.write(f"Acc@3: {get_accuracy_k(3, clusters.test_user_df, user_test, clusters.test_dataset, 0)}\n")
    file.write(f"Acc@5: {get_accuracy_k(5, clusters.test_user_df, user_test, clusters.test_dataset, 0)}\n")
    file.write(f"Acc@10: {get_accuracy_k(10, clusters.test_user_df, user_test, clusters.test_dataset, 0)}\n")

    file.write("SYSTEM metric\n")

    file.write(f"Acc@1: {get_accuracy_k(1, clusters.test_system_df, sys_test, clusters.test_dataset, 1)}\n")
    file.write(f"Acc@3: {get_accuracy_k(3, clusters.test_system_df, sys_test, clusters.test_dataset, 1)}\n")
    file.write(f"Acc@5: {get_accuracy_k(5, clusters.test_system_df, sys_test, clusters.test_dataset, 1)}\n")
    file.write(f"Acc@10: {get_accuracy_k(10, clusters.test_system_df, sys_test, clusters.test_dataset, 1)}\n")


        # In[33]:


    file.write("ALL metric\n")
    file.write(f"Acc@1: {get_all_accuracy_k(1, clusters.test_user_df, clusters.test_system_df, user_test, sys_test, clusters.test_dataset)}\n")
    file.write(f"Acc@3: {get_all_accuracy_k(3, clusters.test_user_df, clusters.test_system_df, user_test, sys_test, clusters.test_dataset)}\n")
    file.write(f"Acc@5: {get_all_accuracy_k(5, clusters.test_user_df, clusters.test_system_df, user_test, sys_test, clusters.test_dataset)}\n")
    file.write(f"Acc@10: {get_all_accuracy_k(10, clusters.test_user_df, clusters.test_system_df, user_test, sys_test, clusters.test_dataset)}\n")

    
num_iters = 3
for i in range(num_iters):
    # In[7]:


    path = "/cephfs/home/ledneva/multiwoz/distilroberta_embeddings.csv"
    clusters = Clusters(first_num_clusters, second_num_clusters, path)
    clusters.form_clusters()


    # In[35]:


    # In[36]:


    create_graph(second_num_clusters, 
              clusters.train_user_df, 
              clusters.train_system_df, 
              clusters.test_user_df, 
              clusters.test_system_df,
              clusters.train_dataset,
              clusters.test_dataset)


# In[ ]:




