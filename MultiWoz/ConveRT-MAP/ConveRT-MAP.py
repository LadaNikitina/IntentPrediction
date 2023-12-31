from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from torch import nn
from torch.nn import Linear

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
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
print(torch.cuda.device_count())

first_num_clusters = 200 # set the correct number of clusters
second_num_clusters = 30

num_iterations = 3

import sys
sys.path.insert(1, '/multiwoz/utils/') # set the correct path to the utils dir
from preprocess import Clusters, get_accuracy_k, get_all_accuracy_k

file = open("ConveRT_MAP_30.txt", "w")

for i in range(num_iterations):
    print(f"Iteration number {i}")

    path = "/multiwoz/distilroberta_embeddings.csv" # set the correct path to the preprocessed embeddings
    clusters = Clusters(first_num_clusters, second_num_clusters, path)
    clusters.form_clusters()

    train_user_utterances = clusters.train_user_df['utterance'].tolist()
    train_system_utterances = clusters.train_system_df['utterance'].tolist()
    train_user_clusters = clusters.train_user_df['cluster'].tolist()
    train_system_clusters = clusters.train_system_df['cluster'].tolist()

    # create pairs context-response
    from tqdm import tqdm 
    from torch.utils.data import DataLoader

    top_k = 15

    import tensorflow_hub as tfhub
    import tensorflow_text  # NOQA: it is used when importing the model
    import tensorflow as tf
    import numpy as np
    from tqdm import tqdm

    nocontext_model = "https://github.com/davidalami/ConveRT/releases/download/1.0/nocontext_tf_model.tar.gz"
    multicontext_model = "https://github.com/davidalami/ConveRT/releases/download/1.0/multicontext_tf_model.tar"

    # The following setting allows the TF1 model to run in TF2
    tf.compat.v1.disable_eager_execution()

    # setting the logging verbosity level to errors-only
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    class SentenceEncoder:
        """A client for running inference with a ConveRT encoder model.

            This wraps tensorflow hub, and gives an interface to input text, and
            get numpy encoding vectors in return. It includes a few optimizations to
            make encoding faster: deduplication of inputs, caching, and internal
            batching.
        """

        def __init__(self,
                     multiple_contexts=True,
                     batch_size=32,
                     ):

            self.multiple_contexts = multiple_contexts
            self.batch_size = batch_size

            self.sess = tf.compat.v1.Session()

            self.text_placeholder = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])


            if self.multiple_contexts:
                self.module = tfhub.Module(multicontext_model)
                self.extra_text_placeholder = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
                self.context_encoding_tensor = self.module(
                    {
                        'context': self.text_placeholder,
                        'extra_context': self.extra_text_placeholder,
                    },
                    signature="encode_context"
                )

            else:
                self.module = tfhub.Module(nocontext_model)
                self.context_encoding_tensor = self.module(self.text_placeholder, signature="encode_context")
                self.encoding_tensor = self.module(self.text_placeholder)

            self.response_encoding_tensor = self.module(self.text_placeholder, signature="encode_response")
            self.sess.run(tf.compat.v1.tables_initializer())
            self.sess.run(tf.compat.v1.global_variables_initializer())

        def encode_multicontext(self, dialogue_history):
            """Encode the whole dialogue to the encoding space to 512-dimensional vectors"""
            if not self.multiple_contexts:
                raise NotImplementedError("Can't encode multiple contexts using a noncontext model")

            context = dialogue_history[-1]
            extra_context = list(dialogue_history[:-1])
            extra_context.reverse()
            extra_context_feature = " ".join(extra_context)

            return self.sess.run(
                self.context_encoding_tensor,
                feed_dict={
                    self.text_placeholder: [context],
                    self.extra_text_placeholder: [extra_context_feature],
                }
            )

        def encode_sentences(self, sentences):
            """Encode the given texts to the encoding space to 1024-dimensional vectors"""
            return self.batch_process(lambda x: self.sess.run(
                self.encoding_tensor, feed_dict={self.text_placeholder: x}
            ), sentences)

        def encode_contexts(self, sentences):
            """Encode the given context texts to the encoding space to 512-dimensional vectors"""
            return self.batch_process(lambda x: self.sess.run(
                self.context_encoding_tensor, feed_dict={self.text_placeholder: x}
                  ), sentences)

        def encode_responses(self, sentences):
            """Encode the given response texts to the encoding space to 512-dimensional vectors"""
            return self.batch_process(
                lambda x: self.sess.run(
                    self.response_encoding_tensor, feed_dict={self.text_placeholder: x}
                ),
                sentences)

        def batch_process(self, func, sentences):
            encodings = []
            for i in range(0, len(sentences), self.batch_size):
                encodings.append(func(sentences[i:i + self.batch_size]))
            return SentenceEncoder.l2_normalize(np.vstack(encodings))

        @staticmethod
        def l2_normalize(encodings):
            """L2 normalizes the given matrix of encodings."""
            norms = np.linalg.norm(encodings, ord=2, axis=-1, keepdims=True)
            return encodings / norms

    sentence_encoder = SentenceEncoder(multiple_contexts=True)

    train_user_utterances = clusters.train_user_df['utterance'].tolist()
    train_system_utterances = clusters.train_system_df['utterance'].tolist()
    valid_user_utterances = clusters.valid_user_df['utterance'].tolist()
    valid_system_utterances = clusters.valid_system_df['utterance'].tolist()

    batches_train_user_utterances = np.array_split(train_user_utterances, 5000)
    batches_train_system_utterances = np.array_split(train_system_utterances, 5000)
    batches_valid_user_utterances = np.array_split(valid_user_utterances, 5000)
    batches_valid_system_utterances = np.array_split(valid_system_utterances, 5000)

    train_user_embeddings = np.concatenate([sentence_encoder.encode_responses(train_user_utterances)
                                for train_user_utterances in tqdm(batches_train_user_utterances)])


    train_system_embeddings = np.concatenate([sentence_encoder.encode_responses(train_system_utterances)
                                              for train_system_utterances in tqdm(batches_train_system_utterances)])
    valid_user_embeddings = np.concatenate([sentence_encoder.encode_responses(valid_user_utterances)
                                for valid_user_utterances in tqdm(batches_valid_user_utterances)])


    valid_system_embeddings = np.concatenate([sentence_encoder.encode_responses(valid_system_utterances)
                                              for valid_system_utterances in tqdm(batches_valid_system_utterances)])

    def get_data(dataset, user_embs, system_embs):
        ''' create pairs context-response '''
        num_negative = 5
        data = []

        ind_user = 0
        ind_system = 0
        for obj in tqdm(dataset):
            for j in range(len(obj["utterance"])):
                utterances_histories = [""]

                if j > 0:
                    for k in range(max(0, j - top_k), j):
                        utterances_histories.append(obj["utterance"][k])

                utterance_emb = sentence_encoder.encode_multicontext(utterances_histories)

                if obj['speaker'][j] == 0:
                    data.append((utterance_emb, user_embs[ind_user]))
                    ind_user += 1
                else:
                    data.append((utterance_emb, system_embs[ind_system]))
                    ind_system += 1

        data_loader = DataLoader(data, batch_size=32, shuffle=True)

        return data_loader


    train_loader = get_data(clusters.train_dataset, train_user_embeddings, train_system_embeddings)
    valid_loader = get_data(clusters.validation_dataset, valid_user_embeddings, valid_system_embeddings)
    
    import random
    def generate_negative_samples(context_emb, response_emb, num_samples):
        batch_size = context_emb.shape[0]
        negative_context_samples = []
        negative_response_samples = []

        for i in range(batch_size):
            indexes = list(range(batch_size))
            indexes.remove(i)
            random_responses = response_emb[random.sample(indexes, num_samples)]
            negative_context_samples.extend([context_emb[i]] * num_samples)
            negative_response_samples.extend(random_responses)

        negative_context_samples = torch.stack(negative_context_samples)
        negative_response_samples = torch.stack(negative_response_samples)
        return negative_context_samples, negative_response_samples

    def get_negative_data(pos_loader):
        neg_batches = []
        for batch in tqdm(train_loader):
            context_emb = batch[0]
            response_emb = batch[1]

            num_negative_samples = 5
            neg_batches.append(generate_negative_samples(context_emb, response_emb, num_negative_samples))
        return neg_batches

    train_negative_samples = get_negative_data(train_loader)
    valid_negative_samples = get_negative_data(valid_loader)

    from torch import nn
    from torch import linalg as LA

    class FeedForward2(nn.Module):
        """Fully-Connected 2-layer Linear Model"""

        def __init__(self, feed_forward2_hidden, num_embed_hidden):
            super().__init__()
            self.linear_1 = nn.Linear(feed_forward2_hidden, feed_forward2_hidden)
            self.linear_2 = nn.Linear(feed_forward2_hidden, feed_forward2_hidden)
            self.norm1 = nn.LayerNorm(feed_forward2_hidden)
            self.norm2 = nn.LayerNorm(feed_forward2_hidden)
            self.final = nn.Linear(feed_forward2_hidden, num_embed_hidden)
            self.orthogonal_initialization()

        def orthogonal_initialization(self):
            for l in [self.linear_1, self.linear_2]:
                torch.nn.init.xavier_uniform_(l.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + F.gelu(self.linear_1(self.norm1(x)))
            x = x + F.gelu(self.linear_2(self.norm2(x)))

            return F.normalize(self.final(x), dim=1, p=2)
        
    import logging
    from collections import OrderedDict

    import numpy as np
    import torch

    from torch.nn.modules.normalization import LayerNorm

    class ConveRT_MAP(nn.Module):
        def __init__(self, feed_forward2_hidden, num_embed_hidden):
            super().__init__()
            self.ff2_context = FeedForward2(feed_forward2_hidden, num_embed_hidden)
            self.ff2_reply = FeedForward2(feed_forward2_hidden, num_embed_hidden)

        def forward(self, context_emb, response_emb):
            context = self.ff2_context(context_emb.reshape(context_emb.shape[0], context_emb.shape[2]))
            response = self.ff2_reply(response_emb)
            cosine_sim = F.cosine_similarity(context, response, dim=1)
            return cosine_sim

        def calculate_loss(self, cos_sim_pos, cos_sim_neg):
    #         pos_loss = torch.mean(pos_cos_sim)
    #         neg_loss = torch.mean(neg_cos_sim)
            total_loss = torch.mean(torch.pow(cos_sim_neg, 2)) + torch.mean(torch.pow(torch.clamp(1.0 - cos_sim_pos, min=0.0), 2))
    #         total_loss = torch.max(torch.tensor(0.0), 1.0 - pos_loss + neg_loss)
            return total_loss

        def encode_context(self, x):
            return self.ff2_context(x)

        def encode_reply(self, x):
            return self.ff2_reply(x)

    model = ConveRT_MAP(512, 512)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


    max_epochs = 20
    best_valid_loss = 100.0
    for epoch in range(max_epochs):
        model.train()
        train_losses = []

        for i, batch in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            context_emb = batch[0]
            response_emb = batch[1]
            pos_cos_sim = model(context_emb, response_emb)

    #         # Generate negative samples
            negative_context_samples, negative_response_samples = train_negative_samples[i][0], train_negative_samples[i][1]
            neg_cos_sim = model(negative_context_samples, negative_response_samples)

            loss = model.calculate_loss(pos_cos_sim, neg_cos_sim)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        avg_train_loss = np.mean(train_losses)

        model.eval()
        with torch.no_grad():
            val_losses = []
            for i, batch in tqdm(enumerate(valid_loader)):
                context_emb = batch[0]
                response_emb = batch[1]
                pos_cos_sim = model(context_emb, response_emb)

                # Generate negative samples
                negative_context_samples, negative_response_samples = valid_negative_samples[i][0], valid_negative_samples[i][1]
                neg_cos_sim = model(negative_context_samples, negative_response_samples)

                val_loss = model.calculate_loss(pos_cos_sim, neg_cos_sim)
                val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses)
            if avg_val_loss > best_valid_loss:
                break

            best_valid_loss = avg_val_loss
            print(f"Epoch {epoch} | Train Loss: {avg_train_loss} | Validation Loss: {avg_val_loss}")

    with torch.no_grad():
        convert_train_user_embeddings = model.encode_reply(torch.from_numpy(train_user_embeddings)).cpu().numpy()
        convert_train_system_embeddings = model.encode_reply(torch.from_numpy(train_system_embeddings)).cpu().numpy()

    with torch.no_grad():
        user_metric = {1 : [], 3 : [], 5 : [], 10 : []}
        system_metric = {1 : [], 3 : [], 5 : [], 10 : []}
        num = 0
        all_num = 0
        ind_user = 0
        ind_system = 0

        for obj in tqdm(clusters.test_dataset):
            user_utterence_metric = {1 : [], 3 : [], 5 : [], 10 : []}
            system_utterence_metric = {1 : [], 3 : [], 5 : [], 10 : []}

            for j in range(len(obj["utterance"])):
                all_num += 1
                utterances_histories = [""]

                if j > 0:
                    for k in range(max(0, j - top_k), j):
                        utterances_histories.append(obj["utterance"][k])

                convert_encoding = sentence_encoder.encode_multicontext(utterances_histories)
                context_encoding = model.encode_context(torch.from_numpy((convert_encoding)))[0].cpu().numpy()

                if obj['speaker'][j] == 0:
                    cur_cluster = clusters.test_user_df["cluster"][ind_user]
                    probs = context_encoding.dot(convert_train_user_embeddings.T)

                    scores = list(zip(probs, train_user_clusters))
                    sorted_scores = list(map(lambda x: x[1], sorted(scores, reverse = True)))
                    result_clusters = []

                    for cluster in sorted_scores:
                        if cluster not in result_clusters:
                            result_clusters.append(cluster)
                        if len(result_clusters) == 10:
                            break

                    for k in [1, 3, 5, 10]:
                        if cur_cluster in result_clusters[:k]:
                            user_utterence_metric[k].append(1) 
                        else:
                            user_utterence_metric[k].append(0) 
                    ind_user += 1
                else:
                    cur_cluster = clusters.test_system_df["cluster"][ind_system]

                    probs = context_encoding.dot(convert_train_system_embeddings.T).tolist()
                    scores = list(zip(probs,
                                train_system_clusters))
                    sorted_scores = list(map(lambda x: x[1], sorted(scores, reverse = True)))

                    result_clusters = []

                    for cluster in sorted_scores:
                        if cluster not in result_clusters:
                            result_clusters.append(cluster)
                        if len(result_clusters) == 10:
                            break

                    for k in [1, 3, 5, 10]:
                        if cur_cluster in result_clusters[:k]:
                            system_utterence_metric[k].append(1) 
                        else:
                            system_utterence_metric[k].append(0) 
                    ind_system += 1

            for k in [1, 3, 5, 10]:
                user_metric[k].append(np.mean(user_utterence_metric[k])) 
                system_metric[k].append(np.mean(system_utterence_metric[k])) 

        file.write("USER METRIC\n")

        for k in [1, 3, 5, 10]:
            file.write(f"Acc@{k}: {np.mean(user_metric[k])}\n")

        file.write("SYSTEM METRIC\n")

        for k in [1, 3, 5, 10]:
            file.write(f"Acc@{k}: {np.mean(system_metric[k])}\n")

        file.write("ALL METRIC\n")

        for k in [1, 3, 5, 10]:
            file.write(f"Acc@{k}: {(np.mean(system_metric[k]) + np.mean(user_metric[k])) / 2}\n")
