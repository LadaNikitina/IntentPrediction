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
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
print(torch.cuda.device_count())

sys.path.insert(1, '/focus/utils/') # set the correct path to utils dir

from data_function import get_data
from functions_GTN import preprocessing
from early_stopping_tools import LRScheduler, EarlyStopping
from preprocess import Clusters, get_accuracy_k, get_all_accuracy_k

from model_fastgtn import FastGTNs
from tqdm import tqdm

num_iterations = 3

file = open("GTN.txt", "w")

first_num_clusters = 200 # set the number of clusters
second_num_clusters = 30 # set the number of clusters

for iteration in range(num_iterations):
    print(f"Iteration number {iteration}")
    file.write(f"Iteration number {iteration}\n\n")
    
    clusters = Clusters(first_num_clusters, second_num_clusters)
    clusters.form_clusters()

    device = torch.device('cuda')
    
    top_k = 5
    batch_size = 128
    embs_dim = len(clusters.user_cluster_embs[0])

    null_cluster_emb = np.zeros(embs_dim)
    fake_cluster_emb = np.zeros(embs_dim)

    embs = np.concatenate([clusters.user_cluster_embs, clusters.system_cluster_embs, [null_cluster_emb, fake_cluster_emb]])

    user_train_x, user_train_y, user_train_embs, sys_train_x, sys_train_y, sys_train_embs = get_data(clusters.train_dataset, top_k, 
                                                        second_num_clusters, 
                                                        clusters.train_user_df, 
                                                        clusters.train_system_df,
                                                        clusters.train_user_embs,
                                                        clusters.train_system_embs)
    user_test_x, user_test_y, user_test_embs, sys_test_x, sys_test_y, sys_test_embs = get_data(clusters.test_dataset, top_k,
                                                     second_num_clusters, 
                                                     clusters.test_user_df, 
                                                     clusters.test_system_df,
                                                     clusters.test_user_embs,
                                                     clusters.test_system_embs)
    user_valid_x, user_valid_y, user_valid_embs, sys_valid_x, sys_valid_y, sys_valid_embs = get_data(clusters.valid_dataset, 
                                                        top_k, second_num_clusters, 
                                                        clusters.valid_user_df, 
                                                        clusters.valid_system_df,
                                                        clusters.valid_user_embs,
                                                        clusters.valid_system_embs)

    user_train_matrices, user_train_node_embs, user_train_labels = preprocessing(user_train_x, 
                                                                                 user_train_y, 
                                                                                 batch_size,
                                                                                 top_k, embs,
                                                                                 user_train_embs, 
                                                                                 second_num_clusters, 1)

    sys_train_matrices, sys_train_node_embs, sys_train_labels = preprocessing(sys_train_x, 
                                                                              sys_train_y, 
                                                                              batch_size,
                                                                              top_k, embs,
                                                                              sys_train_embs,
                                                                              second_num_clusters, 1)

    user_test_matrices, user_test_node_embs, user_test_labels = preprocessing(user_test_x, 
                                                                              user_test_y, 
                                                                              batch_size,
                                                                              top_k, embs,
                                                                              user_test_embs, 
                                                                              second_num_clusters, 0)
    sys_test_matrices, sys_test_node_embs, sys_test_labels = preprocessing(sys_test_x,
                                                                           sys_test_y, 
                                                                           batch_size,
                                                                           top_k, embs,
                                                                           sys_test_embs,
                                                                           second_num_clusters, 0)

    user_valid_matrices, user_valid_node_embs, user_valid_labels = preprocessing(user_valid_x, 
                                                                                user_valid_y, 
                                                                                batch_size,
                                                                                top_k, embs,
                                                                                user_valid_embs,
                                                                                second_num_clusters, 1)
    sys_valid_matrices, sys_valid_node_embs, sys_valid_labels = preprocessing(sys_valid_x,
                                                                             sys_valid_y, 
                                                                             batch_size,
                                                                             top_k, embs,
                                                                             sys_valid_embs,
                                                                             second_num_clusters, 1)

    class user_GTN_arguments():
        epoch = 100
        model = 'FastGTN'
        node_dim = 512
        num_channels = 5
        lr = 0.0005
        weight_decay = 0.0005
        num_layers = 3
        channel_agg = 'mean'
        remove_self_loops = False
        beta = 1
        non_local = False
        non_local_weight = 0
        num_FastGTN_layers = 2
        top_k = 5

    user_args = user_GTN_arguments()
    user_args.num_nodes = user_train_node_embs[0].shape[0]

    user_model = FastGTNs(num_edge_type = 4,
                    w_in = user_train_node_embs[0].shape[1],
                    num_class=second_num_clusters,
                    num_nodes = user_train_node_embs[0].shape[0],
                    args = user_args)

    user_model.to(device)
    user_loss = nn.CrossEntropyLoss()

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    user_optimizer = torch.optim.Adam(user_model.parameters(), lr = user_args.lr)
    user_lr_scheduler = LRScheduler(user_optimizer)
    user_early_stopping = EarlyStopping()

    train_num_batches = len(user_train_matrices)
    valid_num_batches = len(user_valid_matrices)
    old_valid_loss = np.inf

    for epoch in range(user_args.epoch):
        train_epoch_loss = 0

        for num_iter in tqdm(range(train_num_batches)):
            A = user_train_matrices[num_iter]
            node_features = user_train_node_embs[num_iter]
            y_true = torch.from_numpy(user_train_labels[num_iter]).to(device)

            user_model.zero_grad()
            user_model.train()

            y_train = user_model(A, node_features, epoch=epoch)
            if -1 in y_true:
                train_loss = user_loss(y_train[y_true != -1], y_true[y_true != -1])
            else:
                train_loss = user_loss(y_train, y_true)
            # тут считать лосс, выкинуть фейки

            train_loss.backward()
            user_optimizer.step()
            train_epoch_loss += train_loss.detach().item()

        train_epoch_loss /= train_num_batches

        valid_epoch_loss = 0
        with torch.no_grad():
            for num_iter in range(valid_num_batches):
                A = user_valid_matrices[num_iter]
                node_features = user_valid_node_embs[num_iter]
                y_true = torch.from_numpy(user_valid_labels[num_iter]).to(device)

                y_valid = user_model.forward(A, node_features, epoch=epoch)
                if -1 in y_true:
                    valid_loss = user_loss(y_valid[y_true != -1], y_true[y_true != -1])
                else:
                    valid_loss = user_loss(y_valid, y_true)

                # тут считать лосс, выкинуть фейки
                valid_epoch_loss += valid_loss.detach().item()

            valid_epoch_loss /= valid_num_batches

            if abs(valid_epoch_loss - old_valid_loss) < 1e-4 or old_valid_loss < valid_epoch_loss:
                break
            old_valid_loss = valid_epoch_loss

        print(f'Epoch {epoch}, train loss {train_epoch_loss:.4f}, valid loss {valid_epoch_loss:.4f}')  

        user_lr_scheduler(valid_epoch_loss)
        user_early_stopping(valid_epoch_loss)

        if user_early_stopping.early_stop:
            break
            
    user_model.eval()
    test_num_batches = len(user_test_matrices)
    user_true = []
    user_test = []

    with torch.no_grad():
        for num_iter in range(test_num_batches):
            A = user_test_matrices[num_iter]
            node_features = user_test_node_embs[num_iter]
            y_true = torch.from_numpy(user_test_labels[num_iter])
            y_test = torch.softmax(user_model.forward(A, node_features), 1)

            if -1 in y_true:
                user_test += y_test[y_true != -1].tolist()
                user_true += y_true[y_true != -1].tolist()
            else:
                user_test += y_test.tolist()
                user_true += y_true.tolist()

    file.write("USER metric\n")

    file.write(f"Acc@1: {get_accuracy_k(1, clusters.test_user_df, user_test, clusters.test_dataset, 0)}\n")
    file.write(f"Acc@3: {get_accuracy_k(3, clusters.test_user_df, user_test, clusters.test_dataset, 0)}\n")
    file.write(f"Acc@5: {get_accuracy_k(5, clusters.test_user_df, user_test, clusters.test_dataset, 0)}\n")
    file.write(f"Acc@10: {get_accuracy_k(10, clusters.test_user_df, user_test, clusters.test_dataset, 0)}\n")

    class sys_GTN_arguments():
        epoch = 100
        model = 'FastGTN'
        node_dim = 512
        num_channels = 5
        lr = 0.0005
        weight_decay = 0.0005
        num_layers = 3
        channel_agg = 'mean'
        remove_self_loops = False
        beta = 1
        non_local = False
        non_local_weight = 0
        num_FastGTN_layers = 2
        top_k = 5

    sys_args = sys_GTN_arguments()
    sys_args.num_nodes = sys_train_node_embs[0].shape[0]

    sys_model = FastGTNs(num_edge_type = 4,
                    w_in = sys_train_node_embs[0].shape[1],
                    num_class=second_num_clusters, # разобраться что с фейками
                    num_nodes = sys_train_node_embs[0].shape[0],
                    args = sys_args)

    sys_optimizer = torch.optim.Adam(sys_model.parameters(), lr=sys_args.lr)
    sys_lr_scheduler = LRScheduler(sys_optimizer)
    sys_early_stopping = EarlyStopping()

    sys_model.cuda()
    sys_loss = nn.CrossEntropyLoss()

    train_num_batches = len(sys_train_matrices)
    valid_num_batches = len(sys_valid_matrices)
    old_valid_loss = np.inf

    for epoch in range(sys_args.epoch):
        train_epoch_loss = 0

        for num_iter in tqdm(range(train_num_batches)):
            A = sys_train_matrices[num_iter]
            node_features = sys_train_node_embs[num_iter]
            y_true = torch.from_numpy(sys_train_labels[num_iter]).to(device)

            sys_model.zero_grad()
            sys_model.train()

            y_train = sys_model(A, node_features, epoch=epoch)
            if -1 in y_true:
                train_loss = sys_loss(y_train[y_true != -1], y_true[y_true != -1])
            else:
                train_loss = sys_loss(y_train, y_true)

            train_loss.backward()
            sys_optimizer.step()
            train_epoch_loss += train_loss.detach().item()

        train_epoch_loss /= train_num_batches

        valid_epoch_loss = 0
        with torch.no_grad():
            for num_iter in range(valid_num_batches):
                A = sys_valid_matrices[num_iter]
                node_features = sys_valid_node_embs[num_iter]
                y_true = torch.from_numpy(sys_valid_labels[num_iter]).to(device)

                y_valid = sys_model.forward(A, node_features, epoch=epoch)
                if -1 in y_true:
                    valid_loss = sys_loss(y_valid[y_true != -1], y_true[y_true != -1])
                else:
                    valid_loss = sys_loss(y_valid, y_true)

                valid_epoch_loss += valid_loss.detach().item()

            valid_epoch_loss /= valid_num_batches

            if abs(valid_epoch_loss - old_valid_loss) < 1e-4 or old_valid_loss < valid_epoch_loss:
                break
            old_valid_loss = valid_epoch_loss

        print(f'Epoch {epoch}, train loss {train_epoch_loss:.4f}, valid loss {valid_epoch_loss:.4f}')  

        sys_lr_scheduler(valid_epoch_loss)
        sys_early_stopping(valid_epoch_loss)

        if sys_early_stopping.early_stop:
            break

    sys_model.eval()
    test_num_batches = len(sys_test_matrices)
    sys_true = []
    sys_test = []

    with torch.no_grad():
        for num_iter in range(test_num_batches):
            A = sys_test_matrices[num_iter]
            node_features = sys_test_node_embs[num_iter]
            y_true = torch.from_numpy(sys_test_labels[num_iter])
            y_test = torch.softmax(sys_model.forward(A, node_features), 1)

            if -1 in y_true:
                sys_test += y_test[y_true != -1].tolist()
                sys_true += y_true[y_true != -1].tolist()
            else:
                sys_test += y_test.tolist()
                sys_true += y_true.tolist()

    file.write("SYSTEM metric\n")

    file.write(f"Acc@1: {get_accuracy_k(1, clusters.test_system_df, sys_test, clusters.test_dataset, 1)}\n")
    file.write(f"Acc@3: {get_accuracy_k(3, clusters.test_system_df, sys_test, clusters.test_dataset, 1)}\n")
    file.write(f"Acc@5: {get_accuracy_k(5, clusters.test_system_df, sys_test, clusters.test_dataset, 1)}\n")
    file.write(f"Acc@10: {get_accuracy_k(10, clusters.test_system_df, sys_test, clusters.test_dataset, 1)}\n")

    file.write("ALL metric\n")
    file.write(f"Acc@1: {get_all_accuracy_k(1, clusters.test_user_df, clusters.test_system_df, user_test, sys_test, clusters.test_dataset)}\n")
    file.write(f"Acc@3: {get_all_accuracy_k(3, clusters.test_user_df, clusters.test_system_df, user_test, sys_test, clusters.test_dataset)}\n")
    file.write(f"Acc@5: {get_all_accuracy_k(5, clusters.test_user_df, clusters.test_system_df, user_test, sys_test, clusters.test_dataset)}\n")
    file.write(f"Acc@10: {get_all_accuracy_k(10, clusters.test_user_df, clusters.test_system_df, user_test, sys_test, clusters.test_dataset)}\n")
    
    del sys_model, user_model
    del user_train_x, user_train_y, user_train_embs, sys_train_x, sys_train_y, sys_train_embs
    del user_test_x, user_test_y, user_test_embs, sys_test_x, sys_test_y, sys_test_embs
    del user_valid_x, user_valid_y, user_valid_embs, sys_valid_x, sys_valid_y, sys_valid_embs
    del user_train_matrices, user_train_node_embs, user_train_labels, \
        sys_train_matrices, sys_train_node_embs, sys_train_labels
    del user_test_matrices, user_test_node_embs, user_test_labels, \
        sys_test_matrices, sys_test_node_embs, sys_test_labels
    del user_valid_matrices, user_valid_node_embs, user_valid_labels, \
        sys_valid_matrices, sys_valid_node_embs, sys_valid_labels

    torch.cuda.empty_cache()
