#!/usr/bin/env python
# coding: utf-8

# In[7]:


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


# In[8]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(torch.cuda.device_count())


# In[9]:


first_num_clusters = 200
second_num_clusters = 30


# In[10]:


import sys
sys.path.insert(1, '/cephfs/home/ledneva/focus/utils/')


# In[11]:


from preprocess import Clusters, get_accuracy_k, get_all_accuracy_k


num_iterations = 3

file = open("encoder_distilroberta_30.txt", "w")

# In[3]:

for iteration in range(num_iterations):
# In[12]:


    clusters = Clusters(first_num_clusters, second_num_clusters)
    clusters.form_clusters()


    # In[13]:


    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    model = model.to('cuda')


    # In[14]:


    train_user_utterances = clusters.train_user_df['utterance'].tolist()
    train_system_utterances = clusters.train_system_df['utterance'].tolist()
    train_user_clusters = clusters.train_user_df['cluster'].tolist()
    train_system_clusters = clusters.train_system_df['cluster'].tolist()


    # In[15]:


    batches_train_user_utterances = np.array_split(train_user_utterances, 5000)
    batches_train_system_utterances = np.array_split(train_system_utterances, 5000)


    # In[16]:


    from tqdm import tqdm

    train_user_embeddings = np.concatenate([model.encode(train_user_utterances)
                                for train_user_utterances in tqdm(batches_train_user_utterances)])


    train_system_embeddings = np.concatenate([model.encode(train_system_utterances) for train_system_utterances in tqdm(batches_train_system_utterances)])



    # In[22]:


    user_metric = {1 : [], 3 : [], 5 : [], 10 : []}
    system_metric = {1 : [], 3 : [], 5 : [], 10 : []}
    num = 0
    all_num = 0
    ind_user = 0
    ind_system = 0

    for obj in tqdm(clusters.test_dataset):
        user_utterence_metric = {1 : [], 3 : [], 5 : [], 10 : []}
        system_utterence_metric = {1 : [], 3 : [], 5 : [], 10 : []}

        for j in range(len(obj["utterance"])):
            all_num += 1
            utterance_history = " "

            if j > 0:
                utterance_history = obj["utterance"][j - 1]


            context_encoding = model.encode(utterance_history)

            if obj['speaker'][j] == 0:
                cur_cluster = clusters.test_user_df["cluster"][ind_user]
                probs = context_encoding.dot(train_user_embeddings.T)

                scores = list(zip(probs, train_user_clusters))
                sorted_scores = list(map(lambda x: x[1], sorted(scores, reverse = True)))
                
                result_clusters = []
            
                for cluster in sorted_scores:
                    if cluster not in result_clusters:
                        result_clusters.append(cluster)
                    if len(result_clusters) == 10:
                        break

                for k in [1, 3, 5, 10]:
                    if cur_cluster in result_clusters[:k]:
                        user_utterence_metric[k].append(1) 
                    else:
                        user_utterence_metric[k].append(0) 
                ind_user += 1
            else:
                cur_cluster = clusters.test_system_df["cluster"][ind_system]

                probs = context_encoding.dot(train_system_embeddings.T).tolist()
                scores = list(zip(probs,
                            train_system_clusters))
                sorted_scores = list(map(lambda x: x[1], sorted(scores, reverse = True)))
                
                result_clusters = []
            
                for cluster in sorted_scores:
                    if cluster not in result_clusters:
                        result_clusters.append(cluster)
                    if len(result_clusters) == 10:
                        break

                for k in [1, 3, 5, 10]:
                    if cur_cluster in result_clusters[:k]:
                        system_utterence_metric[k].append(1) 
                    else:
                        system_utterence_metric[k].append(0) 
                ind_system += 1

            if cur_cluster in sorted_scores[:10]:
                num += 1
        for k in [1, 3, 5, 10]:
            user_metric[k].append(np.mean(user_utterence_metric[k])) 
            system_metric[k].append(np.mean(system_utterence_metric[k])) 


    # In[23]:


    file.write("USER METRIC\n")

    for k in [1, 3, 5, 10]:
        file.write(f"Acc@{k}: {np.mean(user_metric[k])}\n")

    file.write("SYSTEM METRIC\n")

    for k in [1, 3, 5, 10]:
        file.write(f"Acc@{k}: {np.mean(system_metric[k])}\n")

    file.write("ALL METRIC\n")

    for k in [1, 3, 5, 10]:
        file.write(f"Acc@{k}: {(np.mean(system_metric[k]) + np.mean(user_metric[k])) / 2}\n")


    # In[ ]:




