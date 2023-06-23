import numpy as np
import pandas as pd
import networkx as nx
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from tqdm import tqdm


def get_features(train_user_clusters, train_system_clusters, 
                 train_dataset, unique_intent, 
                 train_user_embs, train_system_embs, 
                 near_number, num_coords, num_clusters):    
    '''
        getting features for each cluster
    '''
    user_array = train_user_embs
    sys_array = train_system_embs
    
    null_array = [0 for i in range(len(sys_array[0]))]

    # intent distribution of each cluster

    user_intents = []
    system_intents = []

    for i in range(num_clusters):
        cluster = train_user_clusters[train_user_clusters['cluster'] == i]

        intents = []
        for intent_arr in cluster['intent']:
            intents += intent_arr

        intent_count = []
        for intent in unique_intent:
            intent_count.append(intents.count(intent))
        user_intents.append(list(np.array(intent_count) / sum(intent_count)))

    for i in range(num_clusters):
        cluster = train_system_clusters[train_system_clusters['cluster'] == i]

        intents = []
        for intent_arr in cluster['intent']:
            intents += intent_arr

        intent_count = []
        for intent in unique_intent:
            intent_count.append(intents.count(intent))
        system_intents.append(list(np.array(intent_count) / sum(intent_count)))

    for i in range(len(unique_intent)):
        null_array.append(0)
        
    sys_array = np.hstack((sys_array, system_intents))
    user_array = np.hstack((user_array, user_intents)) 
    
    return (user_array.tolist(), sys_array.tolist(), null_array)


def get_data(user_features, system_features, null_features,
             user_clusters, system_clusters, data, 
             top_k, num_clusters, user_embs, sys_embs):    
    '''
        getting data and targets
    '''
    user_target = []
    system_target = []
    embs_dim = len(user_embs[0])

    ind_user = 0
    ind_system = 0
    data_user = []
    data_system = []
    num = 0

    for obj in tqdm(data):
        num += 1    
        
        pred_clusters = [(num_clusters, -1) for _ in range(top_k)]
        pred_embs = np.zeros((top_k, embs_dim)).tolist()

        for j in range(len(obj["utterance"])):        
            if obj['speaker'][j] == 0:
                cur_cluster = (user_clusters["cluster"][ind_user], 0)

                data_obj = []
                user_target.append(cur_cluster[0])

                for k in range(top_k):
                    cluster_k = pred_clusters[k]

                    if cluster_k[1] == 0:
                        data_obj += user_features[cluster_k[0]]
                    elif cluster_k[1] == 1:
                        data_obj += system_features[cluster_k[0]]
                    else:
                        data_obj += null_features
                        
                    data_obj += pred_embs[k]

                data_user.append(data_obj)
                cur_embs = user_embs[ind_user].tolist()

                ind_user += 1
            else:
                cur_cluster = (system_clusters["cluster"][ind_system], 1)

                data_obj = []
                system_target.append(cur_cluster[0]) 

                for k in range(top_k):
                    cluster_k = pred_clusters[k]

                    if cluster_k[1] == 0:
                        data_obj += user_features[cluster_k[0]]
                    elif cluster_k[1] == 1:
                        data_obj += system_features[cluster_k[0]]
                    else:
                        data_obj += null_features

                    data_obj += pred_embs[k]
                    

                data_system.append(data_obj)
                cur_embs = sys_embs[ind_system].tolist()

                ind_system += 1


            for k in range(1, top_k):
                pred_clusters[k - 1] = pred_clusters[k]
                pred_embs[k - 1] = pred_embs[k]
                
            pred_clusters[top_k - 1] = cur_cluster
            pred_embs[top_k - 1] = cur_embs

    data_user = np.array(data_user)
    data_user_target = pd.DataFrame({"target" : user_target})     

    data_system = np.array(data_system)
    data_system_target = pd.DataFrame({"target" : system_target})

    return (data_user, data_user_target, data_system, data_system_target)

