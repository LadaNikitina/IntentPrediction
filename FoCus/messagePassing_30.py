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
from tqdm import tqdm

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

sys.path.insert(1, '/cephfs/home/ledneva/focus/utils/')
from preprocess import Clusters, get_accuracy_k, get_all_accuracy_k

num_iterations = 3

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
print(torch.cuda.device_count())

from data_function import get_data
from GAT_functions import get_data_dgl_no_cycles
from early_stopping_tools import LRScheduler, EarlyStopping

first_num_clusters = 200
second_num_clusters = 30

clusters = Clusters(first_num_clusters, second_num_clusters)
clusters.form_clusters()

file = open("MP_noise_30.txt", "w")

for iteration in range(num_iterations):
    print(f"Iteration number {iteration}")
    file.write(f"Iteration number {iteration}\n\n")

    device = torch.device('cuda:0')

    top_k = 5
    batch_size = 128

    user_train_x, user_train_y, sys_train_x, sys_train_y = get_data(clusters.train_dataset, 
                                                                    top_k, second_num_clusters, 
                                                                    clusters.train_user_df, 
                                                                    clusters.train_system_df,
                                                                    np.array(clusters.train_user_embs),
                                                                    np.array(clusters.train_system_embs))
    user_test_x, user_test_y, sys_test_x, sys_test_y = get_data(clusters.test_dataset, 
                                                                top_k, second_num_clusters,
                                                                clusters.test_user_df,
                                                                clusters.test_system_df,
                                                                np.array(clusters.test_user_embs),
                                                                np.array(clusters.test_system_embs))
    user_valid_x, user_valid_y, sys_valid_x, sys_valid_y = get_data(clusters.valid_dataset, 
                                                                    top_k, second_num_clusters,
                                                                    clusters.valid_user_df, 
                                                                    clusters.valid_system_df,
                                                                    np.array(clusters.valid_user_embs),
                                                                    np.array(clusters.valid_system_embs))

    user_train_data = get_data_dgl_no_cycles(user_train_x, user_train_y, 1, top_k, batch_size)

    sys_train_data = get_data_dgl_no_cycles(sys_train_x, sys_train_y, 1, top_k, batch_size)


    user_test_data = get_data_dgl_no_cycles(user_test_x, user_test_y, 0, top_k, batch_size)

    sys_test_data = get_data_dgl_no_cycles(sys_test_x, sys_test_y, 0, top_k, batch_size)

    user_valid_data = get_data_dgl_no_cycles(user_valid_x, user_valid_y, 1, top_k, batch_size)

    sys_valid_data = get_data_dgl_no_cycles(sys_valid_x, sys_valid_y, 1, top_k, batch_size)

    linear_weights = np.zeros(top_k)
    linear_weights[...] = 1 / top_k
    linear_weights = torch.tensor(linear_weights).view(1, -1)
    linear_weights = linear_weights.to(device)

    centre_embs_dim = len(clusters.user_cluster_embs[0])

    num_comps = 512

    learn_embs_dim = num_comps
    learn_emb = nn.Parameter(
                torch.Tensor(2 * second_num_clusters + 1, learn_embs_dim), requires_grad=False
    )
    learn_emb = torch.Tensor(nn.init.xavier_uniform_(learn_emb))

    null_cluster_centre_emb = np.zeros(centre_embs_dim)

    centre_mass = torch.Tensor(np.concatenate([clusters.user_cluster_embs, 
                                               clusters.system_cluster_embs, 
                                               [null_cluster_centre_emb]])).to(device)

    # clusters.user_cluster_embs, clusters.system_cluster_embs - center of mass


    hidden_dim = 512
    embs_dim = 768
    num_heads = 2


    # In[28]:


    from dgl import nn as dgl_nn
    from torch import nn

    class GAT_user(nn.Module):
        def __init__(self, hidden_dim, num_heads):
            super(GAT_user, self).__init__()

            self.embs = nn.Embedding.from_pretrained(learn_emb).requires_grad_(True)
            self.layer1 = dgl_nn.GATv2Conv(embs_dim, hidden_dim, num_heads)
            self.layer2 = dgl_nn.GATv2Conv(hidden_dim * num_heads, hidden_dim, num_heads)

            self.do1 = nn.Dropout(0.2)
            self.do2 = nn.Dropout(0.2)

            self.linear_weights = nn.Embedding.from_pretrained(linear_weights.float()).requires_grad_(True)  

            self.classify = nn.Linear(hidden_dim * num_heads, second_num_clusters)

        def forward(self, bg):
            x = bg.ndata['attr']
            x_emb = bg.ndata['emb']
            embeddings = self.embs.weight
#             all_embs = centre_mass

#             get_embs = lambda i: all_embs[i]
#             node_embs = get_embs(x)

            result_embs = x_emb
#             result_embs = x_emb
            h = result_embs.to(torch.float32)

            h = self.layer1(bg, h)
            h = self.do1(h)
            h = torch.reshape(h, (len(h), num_heads * hidden_dim))      
            h = self.layer2(bg, h)
            h = self.do2(h)


            bg.ndata['h'] = h
            h = torch.reshape(h, (result_embs.shape[0] // top_k, top_k, num_heads * hidden_dim))        
            linear_weights_1dim = torch.reshape(self.linear_weights.weight, (top_k, ))
            get_sum = lambda e: torch.matmul(linear_weights_1dim, e)
            h = list(map(get_sum, h))
            hg = torch.stack(h)
            return self.classify(hg)   

    user_model = GAT_user(hidden_dim, num_heads).to(device)
    user_train_epoch_losses = []
    user_valid_epoch_losses = []

    for param in user_model.parameters():
        param.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(user_model.parameters(), lr = 0.0001)
    lr_scheduler = LRScheduler(optimizer)
    early_stopping = EarlyStopping(5)

    user_num_epochs = 100

    for epoch in range(user_num_epochs):
        train_epoch_loss = 0

        for iter, (batched_graph, labels) in tqdm(enumerate(user_train_data)):
    #         print(f"{iter} / {len(user_train_data)}")
            out = user_model(batched_graph.to(device))
            loss = criterion(out, labels.to(device))
            optimizer.zero_grad()

            loss.backward() 
            optimizer.step() 
            train_epoch_loss += loss.detach().item()

        train_epoch_loss /= (iter + 1)
        user_train_epoch_losses.append(train_epoch_loss)

        valid_epoch_loss = 0
        with torch.no_grad():
            for iter, (batched_graph, labels) in enumerate(user_valid_data):
                out = user_model(batched_graph.to(device))
                loss = criterion(out, labels.to(device))
                valid_epoch_loss += loss.detach().item()

            valid_epoch_loss /= (iter + 1)
            user_valid_epoch_losses.append(valid_epoch_loss)
        print(f'Epoch {epoch}, train loss {train_epoch_loss:.4f}, valid loss {valid_epoch_loss:.4f}')  

        lr_scheduler(valid_epoch_loss)
        early_stopping(valid_epoch_loss)

        if early_stopping.early_stop:
            break


    # In[31]:


    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.title('user model cross entropy averaged over minibatches')
    plt.plot(user_train_epoch_losses, label = "train")
    plt.plot(user_valid_epoch_losses, label = "valid")
    plt.legend()


    # In[32]:


    user_model.eval()
    user_test_X, user_test_Y = map(list, zip(*user_test_data))

    user_probs = []
    user_test = []

    for i in range(len(user_test_Y)):
        g = user_test_X[i].to(device)
        labels = user_test_Y[i]
        labels = labels.tolist()
        user_test += labels
        user_probs_Y = torch.softmax(user_model(g), 1).tolist()
        user_probs += user_probs_Y


    # In[33]:


    file.write("USER metric\n")

    file.write(f"Acc@1: {get_accuracy_k(1, clusters.test_user_df, user_probs, clusters.test_dataset, 0)}\n")
    file.write(f"Acc@3: {get_accuracy_k(3, clusters.test_user_df, user_probs, clusters.test_dataset, 0)}\n")
    file.write(f"Acc@5: {get_accuracy_k(5, clusters.test_user_df, user_probs, clusters.test_dataset, 0)}\n")
    file.write(f"Acc@10: {get_accuracy_k(10, clusters.test_user_df, user_probs, clusters.test_dataset, 0)}\n")


    # ## 4.6 Prediction of system clusters

    # In[36]:

    from dgl import nn as dgl_nn
    from torch import nn

    class GAT_system(nn.Module):
        def __init__(self, hidden_dim, num_heads):
            super(GAT_system, self).__init__()

            self.embs = nn.Embedding.from_pretrained(learn_emb).requires_grad_(True)
            self.layer1 = dgl_nn.GATv2Conv(embs_dim, hidden_dim, num_heads)
            self.layer2 = dgl_nn.GATv2Conv(hidden_dim * num_heads, hidden_dim, num_heads)

            self.do1 = nn.Dropout(0.2)
            self.do2 = nn.Dropout(0.2)

            self.linear_weights = nn.Embedding.from_pretrained(linear_weights.float()).requires_grad_(True)  

            self.classify = nn.Linear(hidden_dim * num_heads, second_num_clusters)

        def forward(self, bg):
            x = bg.ndata['attr']
            x_emb = bg.ndata['emb']
            embeddings = self.embs.weight
#             all_embs = torch.concat((embeddings, centre_mass), dim = 1)

#             get_embs = lambda i: all_embs[i]
#             node_embs = get_embs(x)

#             result_embs = torch.concat((node_embs, x_emb), dim = 1)
            result_embs = x_emb
            h = result_embs.to(torch.float32)

            h = self.layer1(bg, h)
            h = self.do1(h)
            h = torch.reshape(h, (len(h), num_heads * hidden_dim))      
            h = self.layer2(bg, h)
            h = self.do2(h)


            bg.ndata['h'] = h
            h = torch.reshape(h, (result_embs.shape[0] // top_k, top_k, num_heads * hidden_dim))        
            linear_weights_1dim = torch.reshape(self.linear_weights.weight, (top_k, ))
            get_sum = lambda e: torch.matmul(linear_weights_1dim, e)
            h = list(map(get_sum, h))
            hg = torch.stack(h)
            return self.classify(hg)   


    # In[37]:


    system_model = GAT_system(hidden_dim, num_heads).to(device)

    for param in system_model.parameters():
        param.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(system_model.parameters(), lr = 0.0001)
    lr_scheduler = LRScheduler(optimizer)
    early_stopping = EarlyStopping(5)

    sys_num_epochs = 100

    for epoch in range(sys_num_epochs):
        train_epoch_loss = 0

        for iter, (batched_graph, labels) in tqdm(enumerate(sys_train_data)):
    #         print(f"{iter}/{len(sys_train_data)}")
            out = system_model(batched_graph.to(device))
            loss = criterion(out, labels.to(device))
            optimizer.zero_grad()

            loss.backward() 
            optimizer.step() 
            train_epoch_loss += loss.detach().item()

        train_epoch_loss /= (iter + 1)

        valid_epoch_loss = 0
        with torch.no_grad():
            for iter, (batched_graph, labels) in enumerate(sys_valid_data):
                out = system_model(batched_graph.to(device))
                loss = criterion(out, labels.to(device))

                valid_epoch_loss += loss.detach().item()

            valid_epoch_loss /= (iter + 1)

        print(f'Epoch {epoch}, train loss {train_epoch_loss:.4f}, valid loss {valid_epoch_loss:.4f}')  

        lr_scheduler(valid_epoch_loss)
        early_stopping(valid_epoch_loss)

        if early_stopping.early_stop:
            break


    # In[38]:


    system_model.eval()
    system_test_X, system_test_Y = map(list, zip(*sys_test_data))

    system_probs = []
    system_test = []

    for i in range(len(system_test_Y)):
        g = system_test_X[i].to(device)
        labels = system_test_Y[i]
        labels = labels.tolist()
        system_test += labels
        system_probs_Y = torch.softmax(system_model(g), 1).tolist()
        system_probs += system_probs_Y


    # In[39]:


    file.write("SYSTEM metric\n")

    file.write(f"Acc@1: {get_accuracy_k(1, clusters.test_system_df, system_probs, clusters.test_dataset, 1)}\n")
    file.write(f"Acc@3: {get_accuracy_k(3, clusters.test_system_df, system_probs, clusters.test_dataset, 1)}\n")
    file.write(f"Acc@5: {get_accuracy_k(5, clusters.test_system_df, system_probs, clusters.test_dataset, 1)}\n")
    file.write(f"Acc@10: {get_accuracy_k(10, clusters.test_system_df, system_probs, clusters.test_dataset, 1)}\n")


    # In[33]:


    file.write("ALL metric\n")
    file.write(f"Acc@1: {get_all_accuracy_k(1, clusters.test_user_df, clusters.test_system_df, user_probs, system_probs, clusters.test_dataset)}\n")
    file.write(f"Acc@3: {get_all_accuracy_k(3, clusters.test_user_df, clusters.test_system_df, user_probs, system_probs, clusters.test_dataset)}\n")
    file.write(f"Acc@5: {get_all_accuracy_k(5, clusters.test_user_df, clusters.test_system_df, user_probs, system_probs, clusters.test_dataset)}\n")
    file.write(f"Acc@10: {get_all_accuracy_k(10, clusters.test_user_df, clusters.test_system_df, user_probs, system_probs, clusters.test_dataset)}\n")


    # In[ ]:
#     del clusters
    
    del user_train_x, user_train_y, sys_train_x, sys_train_y
    del user_test_x, user_test_y, sys_test_x, sys_test_y
    del user_valid_x, user_valid_y, sys_valid_x, sys_valid_y
    del user_train_data, sys_train_data
    del user_test_data, sys_test_data
    del user_valid_data , sys_valid_data 
    del user_model, system_model
    
    torch.cuda.empty_cache()

