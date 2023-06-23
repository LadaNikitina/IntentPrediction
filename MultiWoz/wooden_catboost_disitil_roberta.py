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

# In[ ]:


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
sys.path.insert(1, '/cephfs/home/ledneva/multiwoz/common_utils/')
from preprocess import Clusters, get_accuracy_k, get_all_accuracy_k


# ## 1. Data loading and processing

# In[ ]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"
print(torch.cuda.device_count())

num_iterations = 3
# In[ ]:

file = open("Wooden_distilroberta.txt", "w")

for iteration in range(num_iterations):
    print(f"Iteration number {iteration}")
    file.write(f"Iteration number {iteration}\n\n")

    first_num_clusters = 200
    second_num_clusters = 30


    # In[ ]:


    embs_path = "/cephfs/home/ledneva/multiwoz/distilroberta_embeddings.csv"
    clusters = Clusters(first_num_clusters, second_num_clusters, embs_path)
    clusters.form_clusters()


    # In[ ]:


    near_number = 3
    top_k = 15
    num_coords = len(clusters.user_cluster_embs[0])


    # In[ ]:


    intents = []

    for obj in clusters.train_dataset:
        for x in obj['dialogue_acts']:
            intents += x['dialog_act']['act_type'] 

    unique_intent = list(set(intents))
    num_intents = len(unique_intent)


    # In[ ]:


    user_features, system_features, null_features = get_features(clusters.train_user_df, 
                                                    clusters.train_system_df, 
                                                    clusters.train_dataset, 
                                                    unique_intent, 
                                                    clusters.user_cluster_embs, 
                                                    clusters.system_cluster_embs, 
                                                    near_number, 
                                                    num_coords,
                                                    second_num_clusters)


    # In[ ]:


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


    # In[ ]:


    user_train_X.shape


    # In[ ]:


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


    # In[ ]:


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


    # In[ ]:


    user_classif = CatBoostClassifier(iterations = 500, learning_rate = 0.1, random_seed = 43, loss_function = 'MultiClass', task_type = 'GPU')
    user_classif.fit(user_train_X, user_train_y, eval_set = [(user_valid_X, user_valid_y)], verbose = 10)


    # In[ ]:


    system_classif = CatBoostClassifier(iterations = 500, learning_rate = 0.1, random_seed = 43, loss_function = 'MultiClass', task_type = 'GPU')
    system_classif.fit(system_train_X, system_train_y, eval_set = [(system_test_X, system_test_y)], verbose = 10)


    # In[ ]:


    test_user_pred = user_classif.predict_proba(user_test_X)
    test_sys_pred = system_classif.predict_proba(system_test_X)


    # In[ ]:


    test_user_true = user_test_y['target'].tolist()
    test_sys_true = system_test_y['target'].tolist()


    # In[ ]:


    # id_one_hot + major_intent_id_one_hot, 10, catboost
    file.write("USER metric\n")

    file.write(f"Acc@1: {get_accuracy_k(1, clusters.test_user_df, test_user_pred, clusters.test_dataset, 0)}\n")
    file.write(f"Acc@3: {get_accuracy_k(3, clusters.test_user_df, test_user_pred, clusters.test_dataset, 0)}\n")
    file.write(f"Acc@5: {get_accuracy_k(5, clusters.test_user_df, test_user_pred, clusters.test_dataset, 0)}\n")
    file.write(f"Acc@10: {get_accuracy_k(10, clusters.test_user_df, test_user_pred, clusters.test_dataset, 0)}\n")


    # In[23]:


    # id_one_hot + major_intent_id_one_hot, 10, catboost
    file.write("SYSTEM metric\n")

    file.write(f"Acc@1: {get_accuracy_k(1, clusters.test_system_df, test_sys_pred, clusters.test_dataset, 1)}\n")
    file.write(f"Acc@3: {get_accuracy_k(3, clusters.test_system_df, test_sys_pred, clusters.test_dataset, 1)}\n")
    file.write(f"Acc@5: {get_accuracy_k(5, clusters.test_system_df, test_sys_pred, clusters.test_dataset, 1)}\n")
    file.write(f"Acc@10: {get_accuracy_k(10, clusters.test_system_df, test_sys_pred, clusters.test_dataset, 1)}\n")


    # In[27]:

    file.write("ALL metric\n")

    file.write(f"Acc@1: {get_all_accuracy_k(1, clusters.test_user_df, clusters.test_system_df, test_user_pred, test_sys_pred, clusters.test_dataset)}\n")
    file.write(f"Acc@3: {get_all_accuracy_k(3, clusters.test_user_df, clusters.test_system_df, test_user_pred, test_sys_pred, clusters.test_dataset)}\n")
    file.write(f"Acc@5: {get_all_accuracy_k(5, clusters.test_user_df, clusters.test_system_df, test_user_pred, test_sys_pred, clusters.test_dataset)}\n")
    file.write(f"Acc@10: {get_all_accuracy_k(10, clusters.test_user_df, clusters.test_system_df, test_user_pred, test_sys_pred, clusters.test_dataset)}\n")

