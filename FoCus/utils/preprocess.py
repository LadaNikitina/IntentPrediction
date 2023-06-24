from collections import Counter
from datasets import load_dataset
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

import numpy as np
import json
import itertools
import pandas as pd

def get_pd_utterances_speaker(data):
    '''
        parsing data
    '''
    utterances = []

    for obj in data:
        utterances += obj['utterance']

    speakers = []

    for obj in data:
        speakers += obj['speaker']    
    
    df = pd.DataFrame()
    
    df['utterance'] = utterances
    df['speaker'] = speakers
    
    return df

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
        with open("/focus/utils/train_focus.json", "r") as read_file: # set correct path to the train_focus.json
            train_data = json.load(read_file)
        with open("/focus/utils/valid_focus.json", "r") as read_file: # set correct path to the valid_focus.json
            test_data = json.load(read_file)

        self.train_dataset = []
        for obj in train_data['data']:
            len_dialog = len(obj['utterance'][-1]['dialogue' + str(len(obj['utterance']))])
            self.train_dataset.append({"utterance" : obj['utterance'][-1]['dialogue' + str(len(obj['utterance']))],
                                       "speaker" :  [i % 2 for i in range(len_dialog)]})                                                     

        self.test_dataset = []
        self.test_speaker = []
        for obj in test_data['data']:
            len_dialog = len(obj['utterance'][-1]['dialogue' + str(len(obj['utterance']))])
            self.test_dataset.append({"utterance" : obj['utterance'][-1]['dialogue' + str(len(obj['utterance']))],
                           "speaker" :  [i % 2 for i in range(len_dialog)]})        

        self.valid_dataset = self.train_dataset[-len(self.test_dataset) : ]
        self.train_dataset = self.train_dataset[ : -len(self.test_dataset)]
        
        # get utterances from data
        train_df = get_pd_utterances_speaker(self.train_dataset)
        test_df = get_pd_utterances_speaker(self.test_dataset)
        valid_df = get_pd_utterances_speaker(self.valid_dataset)

        # split data on user/system train/test/valid
        self.user_train_df = train_df[train_df["speaker"] == 0].reset_index(drop=True)
        self.user_test_df = test_df[test_df["speaker"] == 0].reset_index(drop=True)
        self.user_valid_df = valid_df[valid_df["speaker"] == 0].reset_index(drop=True)
        self.system_train_df = train_df[train_df["speaker"] == 1].reset_index(drop=True)
        self.system_test_df = test_df[test_df["speaker"] == 1].reset_index(drop=True)
        self.system_valid_df = valid_df[valid_df["speaker"] == 1].reset_index(drop=True)
        
        # get user/system train/test/valid indexes for getting embeddings
        self.user_train_index = train_df[train_df["speaker"] == 0].index
        self.user_valid_index = valid_df[valid_df["speaker"] == 0].index + len(train_df)
        self.user_test_index = test_df[test_df["speaker"] == 0].index \
                                + len(train_df) + len(valid_df)
        
        self.system_train_index = train_df[train_df["speaker"] == 1].index
        self.system_valid_index = valid_df[valid_df["speaker"] == 1].index + len(train_df)
        self.system_test_index = test_df[test_df["speaker"] == 1].index \
                                  + len(train_df) + len(valid_df)

    def get_embeddings(self):
        '''
            loading pre-calculated embeddings
        '''
        embeddings = np.loadtxt("/focus/utils/distil_roberta_embeddings.txt") # set correct path to the embeddings

        self.embs_dim = embeddings.shape[1]
        # train user/system embeddings
        self.train_user_embs = embeddings[self.user_train_index]
        self.train_system_embs = embeddings[self.system_train_index]

        # test user/system embeddings
        self.test_user_embs = embeddings[self.user_test_index]
        self.test_system_embs = embeddings[self.system_test_index]

        # validation user/system embeddings
        self.valid_user_embs = embeddings[self.user_valid_index]
        self.valid_system_embs = embeddings[self.system_valid_index]
    
    def first_stage(self):
        '''
            creating first-stage clusters
        '''
        self.train_user_df_first_stage = self.user_train_df.copy()
        self.train_system_df_first_stage = self.system_train_df.copy()

        self.train_user_df_first_stage['cluster'] = get_first_clusters(self.train_user_embs, self.first_num_clusters)
        self.train_system_df_first_stage['cluster'] = get_first_clusters(self.train_system_embs, self.first_num_clusters)

        # counting center of mass of the cluster
        self.user_mean_emb = np.zeros((self.first_num_clusters, self.embs_dim))
        self.system_mean_emb = np.zeros((self.first_num_clusters, self.embs_dim))

        for i in range(self.first_num_clusters):
            index_cluster = self.train_user_df_first_stage[self.train_user_df_first_stage['cluster'] == i].index
            self.user_mean_emb[i] = np.mean(self.train_user_embs[index_cluster], axis = 0)

            index_cluster = self.train_system_df_first_stage[self.train_system_df_first_stage['cluster'] == i].index
            self.system_mean_emb[i] = np.mean(self.train_system_embs[index_cluster], axis = 0)

        # counting word2vec embeddings of the cluster
        ind_user = 0
        ind_system = 0
        array_for_word2vec = []

        for obj in self.train_dataset:
            utterance_clusters = []

            for j in range(len(obj["utterance"])):
                if obj['speaker'][j] == 0:
                    utterance_clusters.append(str(self.train_user_df_first_stage["cluster"][ind_user]) + "-user")
                    ind_user += 1
                else:
                    utterance_clusters.append(str(self.train_system_df_first_stage["cluster"][ind_system]) + "-system")
                    ind_system += 1

            array_for_word2vec.append(utterance_clusters)       

        model_first_stage = Word2Vec(sentences = array_for_word2vec, sg = 0, min_count = 1, workers = 4, window = 10, epochs = 20)

        # counting final embeddings of the clusters
        self.user_cluster_embs_first_stage = []
        self.system_cluster_embs_first_stage = []

        for i in range(self.first_num_clusters):
            self.user_cluster_embs_first_stage.append(list(model_first_stage.wv[str(i)  + "-user"]))
            self.system_cluster_embs_first_stage.append(list(model_first_stage.wv[str(i)  + "-system"]))
    
    def get_test_and_valid(self, num_clusters):
        '''
            cluster searching for test and validation
        '''
        self.test_user_df = self.user_test_df.copy()
        self.test_system_df = self.system_test_df.copy()

        self.valid_user_df = self.user_valid_df.copy()
        self.valid_system_df = self.system_valid_df.copy()
        
        # searching the nearest cluster for each test user utterance
        test_user_clusters = []
        for i in range(len(self.test_user_df)):
            distances = []
            emb = np.array(self.test_user_embs[i])

            for j in range(num_clusters):
                distances.append((np.sqrt(np.sum(np.square(emb - self.user_mean_emb[j]))), j))

            distances = sorted(distances)
            test_user_clusters.append(distances[0][1])
            
        self.test_user_df['cluster'] = test_user_clusters   

        # searching the nearest cluster for each test system utterance
        test_system_clusters = []
        
        for i in range(len(self.test_system_df)):
            distances = []
            emb = np.array(self.test_system_embs[i])

            for j in range(num_clusters):
                distances.append((np.sqrt(np.sum(np.square(emb - self.system_mean_emb[j]))), j))

            distances = sorted(distances)
            test_system_clusters.append(distances[0][1])

        self.test_system_df['cluster'] = test_system_clusters

        # searching the nearest cluster for each validation user utterance
        valid_user_clusters = []

        for i in range(len(self.valid_user_df)):
            distances = []
            emb = np.array(self.valid_user_embs[i])

            for j in range(num_clusters):
                distances.append((np.sqrt(np.sum(np.square(emb - self.user_mean_emb[j]))), j))

            distances = sorted(distances)
            valid_user_clusters.append(distances[0][1])

        self.valid_user_df['cluster'] = valid_user_clusters      

        # searching the nearest cluster for each validation system utterance
        valid_system_clusters = []

        for i in range(len(self.valid_system_df)):
            distances = []
            vec = np.array(self.valid_system_embs[i])

            for j in range(num_clusters):
                distances.append((np.sqrt(np.sum(np.square(vec - self.system_mean_emb[j]))), j))

            distances = sorted(distances)
            valid_system_clusters.append(distances[0][1])

        self.valid_system_df['cluster'] = valid_system_clusters
            
    def one_stage_clustering(self):
        '''
            one stage clustering
        '''
 
        self.get_test_and_valid(self.first_num_clusters)
        self.user_cluster_embs = self.user_cluster_embs_first_stage
        self.system_cluster_embs = self.system_cluster_embs_first_stage
        self.train_user_df = self.train_user_df_first_stage
        self.train_system_df = self.train_system_df_first_stage
          
    def second_stage(self):
        '''
            creating second-stage clusters
        '''
        # creating user second-stage clusters
        self.train_user_df_sec_stage = self.user_train_df.copy()

        model_kmeans = KMeans(n_clusters = self.second_num_clusters, algorithm = "elkan")
        model_kmeans.fit(self.user_cluster_embs_first_stage)
        user_new_clusters = model_kmeans.labels_ 

        new_user_clusters = []

        for i in range(len(self.train_user_df_first_stage)):
            cur_cluster = self.train_user_df_first_stage['cluster'][i]
            new_user_clusters.append(user_new_clusters[cur_cluster])

        self.train_user_df_sec_stage['cluster'] = new_user_clusters
        
        # creating system second-stage clusters
        self.train_system_df_sec_stage = self.system_train_df.copy()

        model_kmeans = KMeans(n_clusters = self.second_num_clusters, algorithm = "elkan")
        model_kmeans.fit(self.system_cluster_embs_first_stage)
        system_new_clusters = model_kmeans.labels_ 

        new_sys_clusters = []

        for i in range(len(self.train_system_df_first_stage)):
            cur_cluster = self.train_system_df_first_stage['cluster'][i]
            new_sys_clusters.append(system_new_clusters[cur_cluster])

        self.train_system_df_sec_stage['cluster'] = new_sys_clusters

        # counting center of mass of the cluster        
        self.user_mean_emb = np.zeros((self.second_num_clusters, self.embs_dim))
        self.system_mean_emb = np.zeros((self.second_num_clusters, self.embs_dim))

        for i in range(self.second_num_clusters):
            index_cluster = self.train_user_df_sec_stage[self.train_user_df_sec_stage['cluster'] == i].index
            self.user_mean_emb[i] = np.mean(self.train_user_embs[index_cluster], axis = 0)

            index_cluster = self.train_system_df_sec_stage[self.train_system_df_sec_stage['cluster'] == i].index
            self.system_mean_emb[i] = np.mean(self.train_system_embs[index_cluster], axis = 0)

        # counting word2vec embeddings of the cluster
        ind_user = 0
        ind_system = 0
        array_for_word2vec = []

        for obj in self.train_dataset:
            utterance_clusters = []

            for j in range(len(obj["utterance"])):
                if obj['speaker'][j] == 0:
                    utterance_clusters.append(str(self.train_user_df_sec_stage["cluster"][ind_user]) + "-user")
                    ind_user += 1
                else:
                    utterance_clusters.append(str(self.train_system_df_sec_stage["cluster"][ind_system]) + "-system")
                    ind_system += 1

            array_for_word2vec.append(utterance_clusters)       

        model_sec_stage = Word2Vec(sentences = array_for_word2vec, sg = 0, min_count = 1, workers = 4, window = 10, epochs = 20)

        # counting final embeddings of the clusters
        self.user_cluster_embs_sec_stage = []
        self.system_cluster_embs_sec_stage = []

        for i in range(self.second_num_clusters):
            self.user_cluster_embs_sec_stage.append(list(model_sec_stage.wv[str(i)  + "-user"]) + list(self.user_mean_emb[i]))
            self.system_cluster_embs_sec_stage.append(list(model_sec_stage.wv[str(i)  + "-user"]) + list(self.system_mean_emb[i]))
    
    def two_stage_clustering(self):
        '''
            two_stage_clustering
        '''

        self.get_test_and_valid(self.second_num_clusters)
        self.user_cluster_embs = self.user_cluster_embs_sec_stage
        self.system_cluster_embs = self.system_cluster_embs_sec_stage
        self.train_user_df = self.train_user_df_sec_stage
        self.train_system_df = self.train_system_df_sec_stage
        
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
            self.one_stage_clustering()
        else:
            print("The second stage of clustering has begun...")
            self.second_stage()
            print("The searching clusters for test and validation has begun...")
            self.two_stage_clustering()
            
            
            

def get_accuracy_k(k, test_df, probabilities, data, flag):
    '''
        metric function, flag: user - speaker 0, system - speaker 1
    '''
    index = 0
    metric = []
    
    for obj in data:
        utterence_metric = []

        for i in range(len(obj["utterance"])):
            if obj['speaker'][i] == flag:
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




def get_all_accuracy_k(k, test_user_data, test_system_data, probs_sys_user, probs_user_sys, data):
    '''
        metric function for both speakers
    '''
    ind_user = 0
    ind_system = 0
    metric = []
    
    for obj in data:
        utterence_metric = []
        pred_cluster = -1

        for i in range(len(obj["utterance"])):
            if obj['speaker'][i] == 0:
                cur_cluster = test_user_data["cluster"][ind_user]
                
                top = []
                    
                for j in range(len(probs_sys_user[ind_user][:])):
                    top.append((probs_sys_user[ind_user][j], j))
                        
                top.sort(reverse=True)
                top = top[:k]

                if (probs_sys_user[ind_user][cur_cluster], cur_cluster) in top:
                    utterence_metric.append(1)
                else:
                    utterence_metric.append(0)
                pred_cluster = cur_cluster   
                ind_user += 1
            else:
                cur_cluster = test_system_data["cluster"][ind_system]
                
                top = []
                    
                for kk in range(len(probs_user_sys[ind_system][:])):
                    top.append((probs_user_sys[ind_system][kk], kk))
                        
                top.sort(reverse=True)
                top = top[:k]

                if (probs_user_sys[ind_system][cur_cluster],cur_cluster) in top:
                    utterence_metric.append(1)
                else:
                    utterence_metric.append(0)
                pred_cluster = cur_cluster  
                ind_system += 1
         
                
        metric.append(np.array(utterence_metric).mean()) 
    return np.array(metric).mean()
