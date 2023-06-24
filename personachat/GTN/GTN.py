from collections import Counter
from datasets import load_dataset
from dgl.dataloading import GraphDataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
import dgl
import dgl.nn.pytorch as dglnn
import math
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import sys
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7,8"
print(torch.cuda.device_count())

sys.path.insert(1, '/personachat/utils/') # set the correct path to the utils dir

from data_function import get_data
from functions_GTN import preprocessing
from early_stopping_tools import LRScheduler, EarlyStopping
from preprocess import Clusters, get_accuracy_k
from model_fastgtn import FastGTNs
from tqdm import tqdm

num_iterations = 3
file = open("GTN.txt", "w")

first_num_clusters = 200
second_num_clusters = 30

for iteration in range(num_iterations):
    print(f"Iteration number {iteration}")
    file.write(f"Iteration number {iteration}\n\n")
    
    clusters = Clusters(first_num_clusters, second_num_clusters)
    clusters.form_clusters()

    device = torch.device('cuda')

    top_k = 5
    batch_size = 512
    embs_dim = len(clusters.cluster_embs[0])


    null_cluster_emb = np.zeros(embs_dim)
    fake_cluster_emb = np.zeros(embs_dim)

    embs = np.concatenate([clusters.cluster_embs, [null_cluster_emb, fake_cluster_emb]])

    train_x, train_y, train_embs = get_data(clusters.train_dataset, top_k, 
                                            second_num_clusters, 
                                            clusters.cluster_train_df, 
                                            clusters.train_embs)
    
    test_x, test_y, test_embs = get_data(clusters.test_dataset, top_k,
                                         second_num_clusters, 
                                         clusters.cluster_test_df,
                                         clusters.test_embs)
    
    valid_x, valid_y, valid_embs = get_data(clusters.valid_dataset, 
                                            top_k, second_num_clusters, 
                                            clusters.cluster_valid_df,
                                            clusters.valid_embs)

    train_matrices, train_node_embs, train_labels = preprocessing(train_x, 
                                                                  train_y, 
                                                                  batch_size,
                                                                  top_k, embs,
                                                                  train_embs, 
                                                                  second_num_clusters, 1)

    test_matrices, test_node_embs, test_labels = preprocessing(test_x, 
                                                               test_y, 
                                                               batch_size,
                                                               top_k, embs,
                                                               test_embs, 
                                                               second_num_clusters, 0)


    valid_matrices, valid_node_embs, valid_labels = preprocessing(valid_x, 
                                                                  valid_y, 
                                                                  batch_size,
                                                                  top_k, embs,
                                                                  valid_embs,
                                                                  second_num_clusters, 1)


    class GTN_arguments():
        epoch = 100
        model = 'FastGTN'
        node_dim = 512
        num_channels = 2
        lr = 0.0001
        weight_decay = 0.0005
        num_layers = 2
        channel_agg = 'concat'
        remove_self_loops = False
        beta = 1
        non_local = False
        non_local_weight = 0
        num_FastGTN_layers = 1
        top_k = 5

    args = GTN_arguments()
    args.num_nodes = train_node_embs[0].shape[0]

    model = FastGTNs(num_edge_type = 3,
                     w_in = train_node_embs[0].shape[1],
                     num_class = second_num_clusters,
                     num_nodes = train_node_embs[0].shape[0],
                     args = args)

    model.to(device)
    loss = nn.CrossEntropyLoss()

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    lr_scheduler = LRScheduler(optimizer)
    early_stopping = EarlyStopping()

    train_num_batches = len(train_matrices)
    valid_num_batches = len(valid_matrices)
    old_valid_loss = np.inf

    for epoch in range(args.epoch):
        train_epoch_loss = 0

        for num_iter in tqdm(range(train_num_batches)):
            A = train_matrices[num_iter]
            node_features = train_node_embs[num_iter]
            y_true = torch.from_numpy(train_labels[num_iter]).to(device)

            model.zero_grad()
            model.train()

            y_train = model(A, node_features, epoch=epoch)
            if -1 in y_true:
                train_loss = loss(y_train[y_true != -1], y_true[y_true != -1])
            else:
                train_loss = loss(y_train, y_true)

            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.detach().item()

        train_epoch_loss /= train_num_batches

        valid_epoch_loss = 0
        with torch.no_grad():
            for num_iter in range(valid_num_batches):
                A = valid_matrices[num_iter]
                node_features = valid_node_embs[num_iter]
                y_true = torch.from_numpy(valid_labels[num_iter]).to(device)

                y_valid = model.forward(A, node_features, epoch=epoch)
                if -1 in y_true:
                    valid_loss = loss(y_valid[y_true != -1], y_true[y_true != -1])
                else:
                    valid_loss = loss(y_valid, y_true)

                valid_epoch_loss += valid_loss.detach().item()

            valid_epoch_loss /= valid_num_batches

            old_valid_loss = valid_epoch_loss

        print(f'Epoch {epoch}, train loss {train_epoch_loss:.4f}, valid loss {valid_epoch_loss:.4f}')  

        lr_scheduler(valid_epoch_loss)
        early_stopping(valid_epoch_loss)

        if early_stopping.early_stop:
            break

    model.eval()
    test_num_batches = len(test_matrices)
    true = []
    test = []

    with torch.no_grad():
        for num_iter in range(test_num_batches):
            A = test_matrices[num_iter]
            node_features = test_node_embs[num_iter]
            y_true = torch.from_numpy(test_labels[num_iter])
            y_test = torch.softmax(model.forward(A, node_features), 1)

            if -1 in y_true:
                test += y_test[y_true != -1].tolist()
                true += y_true[y_true != -1].tolist()
            else:
                test += y_test.tolist()
                true += y_true.tolist()

    file.write("Accuracy metric\n")

    file.write(f"Acc@1: {get_accuracy_k(1, clusters.cluster_test_df, test, clusters.test_dataset)}\n")
    file.write(f"Acc@3: {get_accuracy_k(3, clusters.cluster_test_df, test, clusters.test_dataset)}\n")
    file.write(f"Acc@5: {get_accuracy_k(5, clusters.cluster_test_df, test, clusters.test_dataset)}\n")
    file.write(f"Acc@10: {get_accuracy_k(10, clusters.cluster_test_df, test, clusters.test_dataset)}\n")
    
    del model
    del train_x, train_y, train_embs
    del test_x, test_y, test_embs
    del valid_x, valid_y, valid_embs
    del train_matrices, train_node_embs, train_labels
    del test_matrices, test_node_embs, test_labels
    del valid_matrices, valid_node_embs, valid_labels
    
    torch.cuda.empty_cache()
