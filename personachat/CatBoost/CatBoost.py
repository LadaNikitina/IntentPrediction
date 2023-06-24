from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from catboost import CatBoostClassifier

import pandas as pd
import numpy as np
import networkx as nx
import torch
import sys
import os

from wooden_functions import get_features, get_data
sys.path.insert(1, '/personachat/utils/') # set the correct path to the utils dir
from preprocess import Clusters, get_accuracy_k

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5"
print(torch.cuda.device_count())

num_iterations = 3

file = open("Wooden.txt", "w")

for iteration in range(num_iterations):
    print(f"Iteration number {iteration}")
    file.write(f"Iteration number {iteration}\n\n")

    first_num_clusters = 200
    second_num_clusters = 30

    clusters = Clusters(first_num_clusters, second_num_clusters)
    clusters.form_clusters()

    top_k = 5
    features, null_features = get_features(clusters.cluster_embs, second_num_clusters)

    train_X, train_y = get_data(features, null_features,
                                          clusters.cluster_train_df, 
                                          clusters.train_dataset, top_k, 
                                          second_num_clusters,
                                          np.array(clusters.train_embs))


    test_X, test_y = get_data(features, null_features,
                              clusters.cluster_test_df, 
                              clusters.test_dataset, top_k, 
                              second_num_clusters,
                              np.array(clusters.test_embs))


    valid_X, valid_y = get_data(features, null_features,
                               clusters.cluster_valid_df, 
                               clusters.valid_dataset, top_k, 
                               second_num_clusters,
                               np.array(clusters.valid_embs))

    classif = CatBoostClassifier(iterations = 500, learning_rate = 0.1, random_seed = 43, loss_function = 'MultiClass', task_type = 'GPU')
    classif.fit(train_X, train_y, eval_set = [(valid_X, valid_y)], verbose = 10)

    test_pred = classif.predict_proba(test_X)
    test_true = test_y['target'].tolist()


    file.write("Accuracy metric\n")

    file.write(f"Acc@1: {get_accuracy_k(1, clusters.cluster_test_df, test_pred, clusters.test_dataset)}\n")
    file.write(f"Acc@3: {get_accuracy_k(3, clusters.cluster_test_df, test_pred, clusters.test_dataset)}\n")
    file.write(f"Acc@5: {get_accuracy_k(5, clusters.cluster_test_df, test_pred, clusters.test_dataset)}\n")
    file.write(f"Acc@10: {get_accuracy_k(10, clusters.cluster_test_df, test_pred, clusters.test_dataset)}\n")
