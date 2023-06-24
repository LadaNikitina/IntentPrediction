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

sys.path.insert(1, '/personachat/utils/') # set the correct path to the utils dir
from preprocess import Clusters, get_accuracy_k

num_iterations = 3

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7,8"
print(torch.cuda.device_count())

from data_function import get_data
from GAT_functions import get_data_dgl_no_cycles
from early_stopping_tools import LRScheduler, EarlyStopping


first_num_clusters = 200
second_num_clusters = 30

clusters = Clusters(first_num_clusters, second_num_clusters)
clusters.form_clusters()

file = open("MP.txt", "w")

for iteration in range(num_iterations):
    print(f"Iteration number {iteration}")
    file.write(f"Iteration number {iteration}\n\n")


    device = torch.device('cuda')

    top_k = 5
    batch_size = 512


    train_x, train_y = get_data(clusters.train_dataset, 
                                top_k, second_num_clusters, 
                                clusters.cluster_train_df, 
                                clusters.train_embs)
    test_x, test_y = get_data(clusters.test_dataset, 
                              top_k, second_num_clusters,
                              clusters.cluster_test_df,
                              clusters.test_embs)
    valid_x, valid_y = get_data(clusters.valid_dataset, 
                       top_k, second_num_clusters,
                       clusters.cluster_valid_df,
                       clusters.valid_embs)
    
    train_data = get_data_dgl_no_cycles(train_x, train_y, 1, top_k, batch_size)
    test_data = get_data_dgl_no_cycles(test_x, test_y, 0, top_k, batch_size)
    valid_data = get_data_dgl_no_cycles(valid_x, valid_y, 1, top_k, batch_size)

    linear_weights = np.zeros(top_k)
    linear_weights[...] = 1 / top_k
    linear_weights = torch.tensor(linear_weights).view(1, -1)
    linear_weights = linear_weights.to(device)

    centre_embs_dim = len(clusters.cluster_embs[0])

    num_comps = 512

    learn_embs_dim = num_comps
    learn_emb = nn.Parameter(
                torch.Tensor(second_num_clusters + 1, learn_embs_dim), requires_grad=False
    )
    learn_emb = torch.Tensor(nn.init.xavier_uniform_(learn_emb))

    null_cluster_centre_emb = np.zeros(centre_embs_dim)
    centre_mass = torch.Tensor(np.concatenate([clusters.cluster_embs, 
                                               [null_cluster_centre_emb]])).to(device)
    
    hidden_dim = 2048
    embs_dim = 768
    num_heads = 2

    from dgl import nn as dgl_nn
    from torch import nn

    class GAT(nn.Module):
        def __init__(self, hidden_dim, num_heads):
            super(GAT, self).__init__()

            self.embs = nn.Embedding.from_pretrained(learn_emb).requires_grad_(True)
            self.layer1 = dgl_nn.GATv2Conv(embs_dim, hidden_dim, num_heads)
            self.layer2 = dgl_nn.GATv2Conv(hidden_dim * num_heads, hidden_dim, num_heads)
            self.layer3 = dgl_nn.GATv2Conv(hidden_dim * num_heads, hidden_dim, num_heads)

            self.do1 = nn.Dropout(0.2)
            self.do2 = nn.Dropout(0.2)
            self.do3 = nn.Dropout(0.2)

            self.linear_weights = nn.Embedding.from_pretrained(linear_weights.float()).requires_grad_(True)  

            self.classify = nn.Linear(hidden_dim * num_heads, second_num_clusters)

        def forward(self, bg):
            x = bg.ndata['attr']
            x_emb = bg.ndata['emb']
            embeddings = self.embs.weight
            all_embs = centre_mass

            get_embs = lambda i: all_embs[i]
            node_embs = get_embs(x)

            result_embs = x_emb
#             result_embs = x_emb
            h = result_embs.to(torch.float32)

            h = self.layer1(bg, h)
            h = self.do1(h)
            h = torch.reshape(h, (len(h), num_heads * hidden_dim))      
            h = self.layer2(bg, h)
            h = self.do2(h)
            h = torch.reshape(h, (len(h), num_heads * hidden_dim))      
            h = self.layer3(bg, h)
            h = self.do3(h)


            bg.ndata['h'] = h
            h = torch.reshape(h, (len(node_embs) // top_k, top_k, num_heads * hidden_dim))        
            linear_weights_1dim = torch.reshape(self.linear_weights.weight, (top_k, ))
            get_sum = lambda e: torch.matmul(linear_weights_1dim, e)
            h = list(map(get_sum, h))
            hg = torch.stack(h)
            return self.classify(hg)   

    model = GAT(hidden_dim, num_heads).to(device)
    train_epoch_losses = []
    valid_epoch_losses = []

    for param in model.parameters():
        param.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    lr_scheduler = LRScheduler(optimizer)
    early_stopping = EarlyStopping(10)

    num_epochs = 100

    for epoch in range(num_epochs):
        train_epoch_loss = 0

        for iter, (batched_graph, labels) in tqdm(enumerate(train_data)):
            out = model(batched_graph.to(device))
            loss = criterion(out, labels.to(device).type(torch.int64))
            optimizer.zero_grad()

            loss.backward() 
            optimizer.step() 
            train_epoch_loss += loss.detach().item()

        train_epoch_loss /= (iter + 1)
        train_epoch_losses.append(train_epoch_loss)

        valid_epoch_loss = 0
        with torch.no_grad():
            for iter, (batched_graph, labels) in enumerate(valid_data):
                out = model(batched_graph.to(device))
                loss = criterion(out, labels.to(device).type(torch.int64))
                valid_epoch_loss += loss.detach().item()

            valid_epoch_loss /= (iter + 1)
            valid_epoch_losses.append(valid_epoch_loss)
        print(f'Epoch {epoch}, train loss {train_epoch_loss:.4f}, valid loss {valid_epoch_loss:.4f}')  

        lr_scheduler(valid_epoch_loss)
        early_stopping(valid_epoch_loss)

        if early_stopping.early_stop:
            break

    model.eval()
    test_X, test_Y = map(list, zip(*test_data))

    probs = []
    test = []

    for i in range(len(test_Y)):
        g = test_X[i].to(device)
        labels = test_Y[i]
        labels = labels.tolist()
        test += labels
        probs_Y = torch.softmax(model(g), 1).tolist()
        probs += probs_Y

    file.write("Accuracy metric\n")

    file.write(f"Acc@1: {get_accuracy_k(1, clusters.cluster_test_df, probs, clusters.test_dataset)}\n")
    file.write(f"Acc@3: {get_accuracy_k(3, clusters.cluster_test_df, probs, clusters.test_dataset)}\n")
    file.write(f"Acc@5: {get_accuracy_k(5, clusters.cluster_test_df, probs, clusters.test_dataset)}\n")
    file.write(f"Acc@10: {get_accuracy_k(10, clusters.cluster_test_df, probs, clusters.test_dataset)}\n")

    del train_x, train_y
    del test_x, test_y
    del valid_x, valid_y
    del train_data
    del test_data
    del valid_data
    del model
    
    torch.cuda.empty_cache()

