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

from conversational_sentence_encoder.vectorizers import SentenceEncoder

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3,4"
print(torch.cuda.device_count())

first_num_clusters = 200
second_num_clusters = 30

import sys
sys.path.insert(1, '/personachat/utils/') # set the correct path to the utils dir

from preprocess import Clusters, get_accuracy_k


num_iterations = 3

file = open("ConveRT_distilroberta_30.txt", "w")
sentence_encoder = SentenceEncoder(multiple_contexts=True)


for iteration in range(num_iterations):
    clusters = Clusters(first_num_clusters, second_num_clusters)
    clusters.form_clusters()

    train_utterances = clusters.cluster_train_df['sentences'].tolist()
    train_clusters = clusters.cluster_train_df['cluster'].tolist()
    
    test_utterances = clusters.cluster_test_df['sentences'].tolist()
    test_clusters = clusters.cluster_test_df['cluster'].tolist()

    batches_train_utterances = np.array_split(train_utterances, 2500)
    batches_test_utterances = np.array_split(test_utterances, 2500)


    from tqdm import tqdm

    train_embeddings = torch.from_numpy(np.concatenate([sentence_encoder.encode_responses(train_utterances)
                       for train_utterances in tqdm(batches_train_utterances)])).to("cuda")

    top_k = 5

    metric = {1 : [], 3 : [], 5 : [], 10 : []}
    num = 0
    all_num = 0
    index = 0

    for obj in tqdm(clusters.test_dataset):
        utterance_metric = {1 : [], 3 : [], 5 : [], 10 : []}

        for j in range(len(obj)):
            all_num += 1
            utterances_histories = [""]

            if j > 0:
                for k in range(max(0, j - top_k), j):
                    utterances_histories.append(obj[k])


            context_encoding = torch.from_numpy(np.array((sentence_encoder.encode_multicontext(utterances_histories))[0])).to("cuda")
            cur_cluster = clusters.cluster_test_df["cluster"][index]
            probs = torch.matmul(context_encoding, train_embeddings.T).tolist()

            scores = list(zip(probs, train_clusters))
            sorted_scores = list(map(lambda x: x[1], sorted(scores, reverse = True)))

            result_clusters = []

            for cluster in sorted_scores:
                if cluster not in result_clusters:
                    result_clusters.append(cluster)
                if len(result_clusters) == 10:
                    break

            for k in [1, 3, 5, 10]:
                if cur_cluster in result_clusters[:k]:
                    utterance_metric[k].append(1) 
                else:
                    utterance_metric[k].append(0) 
            index += 1

        for k in [1, 3, 5, 10]:
            metric[k].append(np.mean(utterance_metric[k])) 

    file.write("ACCURACY METRIC\n")

    for k in [1, 3, 5, 10]:
        file.write(f"Acc@{k}: {np.mean(metric[k])}\n")

        
        
