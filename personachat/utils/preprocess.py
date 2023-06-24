from collections import Counter
from datasets import load_dataset
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

import itertools
import numpy as np
import pandas as pd

def get_first_clusters(embs, n_clusters):
    '''
        first-stage clustering
    '''
    model = KMeans(n_clusters = n_clusters)
    model.fit(embs)
    return model.labels_

class Clusters:
    ''' 
        the class that forms clusters
    '''
    def __init__(self, first_num_clusters, second_num_clusters):
        self.first_num_clusters = first_num_clusters
        self.second_num_clusters = second_num_clusters
        
        if self.first_num_clusters == -1:
            self.first_num_clusters = self.second_num_clusters
            self.second_num_clusters = -1
            
    def train_test_valid(self):
        '''
            data loading
        '''
        dataset = load_dataset("bavard/personachat_truecased")
        
        # train data

        train_data = dataset['train'] 
        train_history = train_data['history']
        train_conv_id = train_data['conv_id']
        self.train_dataset = [train_history[i] for i, conv_id in enumerate(train_conv_id) \
                              if i + 1 != len(train_conv_id) and conv_id != train_conv_id[i + 1]] 

        # validation data
        validation_data = dataset['validation'] 
        validation_history = validation_data['history']
        validation_conv_id = validation_data['conv_id']
        self.valid_dataset = [validation_history[i] for i, conv_id in enumerate(validation_conv_id) \
                                   if i + 1 != len(validation_conv_id) and conv_id != validation_conv_id[i + 1]] 


        # test data
        self.test_dataset = self.train_dataset[-len(self.valid_dataset) : ]
        self.train_dataset = self.train_dataset[ : -len(self.valid_dataset)]
        
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.valid_df = pd.DataFrame()
        
        self.train_df['sentences'] = list(itertools.chain.from_iterable(self.train_dataset))
        self.test_df['sentences'] = list(itertools.chain.from_iterable(self.test_dataset))
        self.valid_df['sentences'] = list(itertools.chain.from_iterable(self.valid_dataset))
        
        self.train_index = self.train_df.index
        self.test_index = self.test_df.index + len(self.train_df)
        self.valid_index = self.valid_df.index + len(self.train_df) + len(self.test_df)

    def get_embeddings(self):
        '''
            loading pre-calculated embeddings
        '''
        embeddings = np.loadtxt("/cephfs/home/ledneva/personachat/utils/distil_roberta_embeddings.txt")
        self.train_embs = embeddings[self.train_index]
        self.test_embs = embeddings[self.test_index]
        self.valid_embs = embeddings[self.valid_index]

        self.embs_dim = self.train_embs.shape[1]
    
    def first_stage(self):
        '''
            creating first-stage clusters
        '''
        self.train_df_first_stage = self.train_df.copy()

        self.train_df_first_stage['cluster'] = get_first_clusters(self.train_embs, self.first_num_clusters)

        # counting center of mass of the cluster
        self.mean_emb = np.zeros((self.first_num_clusters, self.embs_dim))

        for i in range(self.first_num_clusters):
            index_cluster = self.train_df_first_stage[self.train_df_first_stage['cluster'] == i].index
            self.mean_emb[i] = np.mean(self.train_embs[index_cluster], axis = 0)

        # counting word2vec embeddings of the cluster
        index = 0
        array_for_word2vec = []

        for obj in self.train_dataset:
            utterance_clusters = []

            for j in range(len(obj)):
                utterance_clusters.append(str(self.train_df_first_stage["cluster"][index]))
                index += 1

            array_for_word2vec.append(utterance_clusters)       

        model_first_stage = Word2Vec(sentences = array_for_word2vec, sg = 0, min_count = 1, workers = 4, window = 10, epochs = 20)

        # counting final embeddings of the clusters
        self.cluster_embs_first_stage = []

        for i in range(self.first_num_clusters):
            self.cluster_embs_first_stage.append(list(model_first_stage.wv[str(i)]))
    
    def get_test_and_valid(self, num_clusters):
        '''
            cluster searching for test and validation
        '''
        self.cluster_test_df = self.test_df.copy()
        self.cluster_valid_df = self.valid_df.copy()
        
        # searching the nearest cluster for each test user utterance
        test_clusters = []
        for i in range(len(self.cluster_test_df)):
            distances = []
            emb = np.array(self.test_embs[i])

            for j in range(num_clusters):
                distances.append((np.sqrt(np.sum(np.square(emb - self.mean_emb[j]))), j))

            distances = sorted(distances)
            test_clusters.append(distances[0][1])
            
        self.cluster_test_df['cluster'] = test_clusters   

        # searching the nearest cluster for each validation user utterance
        valid_clusters = []

        for i in range(len(self.cluster_valid_df)):
            distances = []
            emb = np.array(self.valid_embs[i])

            for j in range(num_clusters):
                distances.append((np.sqrt(np.sum(np.square(emb - self.mean_emb[j]))), j))

            distances = sorted(distances)
            valid_clusters.append(distances[0][1])

        self.cluster_valid_df['cluster'] = valid_clusters      

            
    def one_stage_clustering(self):
        '''
            one stage clustering
        '''
        self.get_test_and_valid(self.first_num_clusters)
        self.cluster_embs = self.cluster_embs_first_stage
        self.cluster_train_df = self.train_df_first_stage
          
    def second_stage(self):
        '''
            creating second-stage clusters
        '''
        # creating user second-stage clusters
        self.train_df_sec_stage = self.train_df.copy()

        model_kmeans = KMeans(n_clusters = self.second_num_clusters, algorithm = "elkan")
        model_kmeans.fit(self.cluster_embs_first_stage)
        first_stage_clusters = model_kmeans.labels_ 

        new_clusters = []

        for i in range(len(self.train_df_first_stage)):
            cur_cluster = self.train_df_first_stage['cluster'][i]
            new_clusters.append(first_stage_clusters[cur_cluster])

        self.train_df_sec_stage['cluster'] = new_clusters

        # counting center of mass of the cluster        
        self.mean_emb = np.zeros((self.second_num_clusters, self.embs_dim))

        for i in range(self.second_num_clusters):
            index_cluster = self.train_df_sec_stage[self.train_df_sec_stage['cluster'] == i].index
            self.mean_emb[i] = np.mean(self.train_embs[index_cluster], axis = 0)

        # counting word2vec embeddings of the cluster
        index = 0
        array_for_word2vec = []

        for obj in self.train_dataset:
            utterance_clusters = []

            for j in range(len(obj)):
                utterance_clusters.append(str(self.train_df_sec_stage["cluster"][index]))
                index += 1

            array_for_word2vec.append(utterance_clusters)       

        model_sec_stage = Word2Vec(sentences = array_for_word2vec, sg = 0, min_count = 1, workers = 4, window = 10, epochs = 20)

        # counting final embeddings of the clusters
        self.cluster_embs_sec_stage = []

        for i in range(self.second_num_clusters):
            self.cluster_embs_sec_stage.append(list(model_sec_stage.wv[str(i)]) + list(self.mean_emb[i]))
    
    def two_stage_clustering(self):
        '''
            two_stage_clustering
        '''

        self.get_test_and_valid(self.second_num_clusters)
        self.cluster_embs = self.cluster_embs_sec_stage
        self.cluster_train_df = self.train_df_sec_stage
        
    def form_clusters(self):
        '''
            formation of clusters
        '''
        print("The data is loading...")
        self.train_test_valid()
        print("The embeddings are loading...")
        self.get_embeddings()
        print("The first stage of clustering has begun...")
        self.first_stage()
        
        if self.second_num_clusters == -1:
            print("The searching clusters for test and validation has begun...")
            self.one_stage_clustering()
        else:
            print("The second stage of clustering has begun...")
            self.second_stage()
            print("The searching clusters for test and validation has begun...")
            self.two_stage_clustering()
            
            
def get_accuracy_k(k, test_df, probabilities, data):
    '''
        metric function
    '''
    index = 0
    metric = []
    
    for obj in data:
        utterence_metric = []

        for i in range(len(obj)):
            cur_cluster = test_df["cluster"][index]
                
            top = []
                    
            for j in range(len(probabilities[index][:])):
                top.append((probabilities[index][j], j))
                        
            top.sort(reverse=True)
            top = top[:k]

            if (probabilities[index][cur_cluster], cur_cluster) in top:
                utterence_metric.append(1)
            else:
                utterence_metric.append(0)
            index += 1
                
        metric.append(np.array(utterence_metric).mean()) 
    return np.array(metric).mean()
