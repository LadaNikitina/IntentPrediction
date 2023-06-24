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
sys.path.insert(1, '/cephfs/home/ledneva/personachat/utils/')


# In[19]:


from preprocess import Clusters, get_accuracy_k

file = open("MarkovChain_distilroberta_30.txt", "w")

def create_graph(num_nodes, train_data, test_data, train_dataset, test_dataset):
    probs = np.zeros((num_nodes + 1, num_nodes))
    
    index = 0

    for obj in train_dataset:
        pred_cluster = num_nodes

        for j in range(len(obj)):
            cur_cluster = train_data["cluster"][index]
            index += 1

            probs[pred_cluster][cur_cluster] += 1
            pred_cluster = cur_cluster
          
    for i in range(num_nodes + 1):
        sum_i_probs = sum(probs[i])
        
        if sum_i_probs != 0:
            probs[i] /= sum_i_probs
 
    test = []
            
    index = 0
    
    for obj in test_dataset:
        pred_cluster = num_nodes

        for j in range(len(obj)):
            cur_cluster = test_data["cluster"][index]
            test.append(probs[pred_cluster])
            index += 1
            pred_cluster = cur_cluster
                
    file.write("Accuracy metric\n")

    file.write(f"Acc@1: {get_accuracy_k(1, test_data, test, test_dataset)}\n")
    file.write(f"Acc@3: {get_accuracy_k(3, test_data, test, test_dataset)}\n")
    file.write(f"Acc@5: {get_accuracy_k(5, test_data, test, test_dataset)}\n")
    file.write(f"Acc@10: {get_accuracy_k(10, test_data, test, test_dataset)}\n")

    
num_iters = 3
for i in range(num_iters):

    clusters = Clusters(first_num_clusters, second_num_clusters)
    clusters.form_clusters()

    create_graph(second_num_clusters, 
                 clusters.cluster_train_df, 
                 clusters.cluster_test_df, 
                 clusters.train_dataset,
                 clusters.test_dataset)


# In[ ]:




