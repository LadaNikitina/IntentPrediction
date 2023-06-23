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

from wooden_functions_all_uttr_embs import get_features, get_data
sys.path.insert(1, '/multiwoz/utils/') # set the correct path to the utils dir
from preprocess import Clusters, get_accuracy_k, get_all_accuracy_k

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"
print(torch.cuda.device_count())

num_iterations = 3

file = open("Wooden.txt", "w")

for iteration in range(num_iterations):
    print(f"Iteration number {iteration}")
    file.write(f"Iteration number {iteration}\n\n")

    first_num_clusters = 200 # set the correct number of clusters
    second_num_clusters = 30

    embs_path = "/multiwoz/distilroberta_embeddings.csv" # set the correct path to the embeddings
    clusters = Clusters(first_num_clusters, second_num_clusters, embs_path)
    clusters.form_clusters()

    near_number = 3
    top_k = 15
    num_coords = len(clusters.user_cluster_embs[0])

    intents = []

    for obj in clusters.train_dataset:
        for x in obj['dialogue_acts']:
            intents += x['dialog_act']['act_type'] 

    unique_intent = list(set(intents))
    num_intents = len(unique_intent)

    user_features, system_features, null_features = get_features(clusters.train_user_df, 
                                                    clusters.train_system_df, 
                                                    clusters.train_dataset, 
                                                    unique_intent, 
                                                    clusters.user_cluster_embs, 
                                                    clusters.system_cluster_embs, 
                                                    near_number, 
                                                    num_coords,
                                                    second_num_clusters)

    user_train_X, user_train_y, system_train_X, system_train_y = get_data(user_features, 
                                                                          system_features, 
                                                                          null_features,
                                                                          clusters.train_user_df, 
                                                                          clusters.train_system_df, 
                                                                          clusters.train_dataset, 
                                                                          top_k, 
                                                                          second_num_clusters,
                                                                          np.array(clusters.train_user_embs),
                                                                          np.array(clusters.train_system_embs))

    user_test_X, user_test_y, system_test_X, system_test_y = get_data(user_features, 
                                                                      system_features, 
                                                                      null_features,
                                                                      clusters.test_user_df, 
                                                                      clusters.test_system_df, 
                                                                      clusters.test_dataset, 
                                                                      top_k, 
                                                                      second_num_clusters,
                                                                      np.array(clusters.test_user_embs),
                                                                      np.array(clusters.test_system_embs))

    user_valid_X, user_valid_y, system_valid_X, system_valid_y = get_data(user_features, 
                                                                          system_features, 
                                                                          null_features,
                                                                          clusters.valid_user_df, 
                                                                          clusters.valid_system_df, 
                                                                          clusters.validation_dataset, 
                                                                          top_k, 
                                                                          second_num_clusters,
                                                                          np.array(clusters.valid_user_embs),
                                                                          np.array(clusters.valid_system_embs))
    
    user_classif = CatBoostClassifier(iterations = 500, learning_rate = 0.1, random_seed = 43, loss_function = 'MultiClass', task_type = 'GPU')
    user_classif.fit(user_train_X, user_train_y, eval_set = [(user_valid_X, user_valid_y)], verbose = 10)

    system_classif = CatBoostClassifier(iterations = 500, learning_rate = 0.1, random_seed = 43, loss_function = 'MultiClass', task_type = 'GPU')
    system_classif.fit(system_train_X, system_train_y, eval_set = [(system_test_X, system_test_y)], verbose = 10)

    test_user_pred = user_classif.predict_proba(user_test_X)
    test_sys_pred = system_classif.predict_proba(system_test_X)

    test_user_true = user_test_y['target'].tolist()
    test_sys_true = system_test_y['target'].tolist()

    file.write("USER metric\n")

    file.write(f"Acc@1: {get_accuracy_k(1, clusters.test_user_df, test_user_pred, clusters.test_dataset, 0)}\n")
    file.write(f"Acc@3: {get_accuracy_k(3, clusters.test_user_df, test_user_pred, clusters.test_dataset, 0)}\n")
    file.write(f"Acc@5: {get_accuracy_k(5, clusters.test_user_df, test_user_pred, clusters.test_dataset, 0)}\n")
    file.write(f"Acc@10: {get_accuracy_k(10, clusters.test_user_df, test_user_pred, clusters.test_dataset, 0)}\n")

    file.write("SYSTEM metric\n")

    file.write(f"Acc@1: {get_accuracy_k(1, clusters.test_system_df, test_sys_pred, clusters.test_dataset, 1)}\n")
    file.write(f"Acc@3: {get_accuracy_k(3, clusters.test_system_df, test_sys_pred, clusters.test_dataset, 1)}\n")
    file.write(f"Acc@5: {get_accuracy_k(5, clusters.test_system_df, test_sys_pred, clusters.test_dataset, 1)}\n")
    file.write(f"Acc@10: {get_accuracy_k(10, clusters.test_system_df, test_sys_pred, clusters.test_dataset, 1)}\n")

    file.write("ALL metric\n")

    file.write(f"Acc@1: {get_all_accuracy_k(1, clusters.test_user_df, clusters.test_system_df, test_user_pred, test_sys_pred, clusters.test_dataset)}\n")
    file.write(f"Acc@3: {get_all_accuracy_k(3, clusters.test_user_df, clusters.test_system_df, test_user_pred, test_sys_pred, clusters.test_dataset)}\n")
    file.write(f"Acc@5: {get_all_accuracy_k(5, clusters.test_user_df, clusters.test_system_df, test_user_pred, test_sys_pred, clusters.test_dataset)}\n")
    file.write(f"Acc@10: {get_all_accuracy_k(10, clusters.test_user_df, clusters.test_system_df, test_user_pred, test_sys_pred, clusters.test_dataset)}\n")
