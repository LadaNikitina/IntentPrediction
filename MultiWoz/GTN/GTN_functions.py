import dgl
import dgl.function as fn
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm
# from ..utils import transform_relation_graph_list

def get_accuracy(k, data, prediction):
    metric = []
    target = data
    
    for i in range(len(target)):
        cur_cluster = target[i]
        top_k = []
        predictions = list(prediction[i])
                    
        for kk in range(len(predictions)):
            top_k.append((predictions[kk], kk))
                        
        top_k.sort(reverse=True)
        top_k = top_k[:k]

        if (predictions[cur_cluster], cur_cluster) in top_k:
            metric.append(1)
        else:
            metric.append(0)
                
    return np.array(metric).mean()

def get_data_dgl(x, y, batch_size, top_k, num_clusters, embs):
    null_cluster = num_clusters * 2
    data_len = len(x)
    indexes = np.random.shuffle(np.arange(data_len))
    x_ = np.concatenate((x[indexes][0], np.full((batch_size - data_len % batch_size, top_k), null_cluster + 1)), axis = 0)
    y_ = np.concatenate((y[indexes][0], np.full((batch_size - data_len % batch_size), num_clusters)), axis = 0)
    batches_x = np.reshape(x_, (len(x_) // batch_size, batch_size, top_k))
    batches_y = np.reshape(y_, (len(y_) // batch_size, batch_size))
    num_batches = len(batches_x)
    matrices = []
    all_node_features = []
    all_labels = []
    all_batches = []
    
    for ind_batch in range(num_batches):
        batch_x = batches_x[ind_batch]
        labels = batches_y[ind_batch]
        
        len_batch = len(batch_x)
    
        data = {}
        node_types_dict = {}
        node_types_dict = node_types_dict.fromkeys(np.arange(2 * num_clusters + 2), 0)
        get_features = lambda i: embs[i]
        node_features = get_features(np.reshape(batch_x, (len_batch * top_k)))

        for ind_graph in range(len_batch):
            graph = batch_x[ind_graph]
            
            for i in range(top_k - 1):
                
                if graph[i] == null_cluster or graph[i] == null_cluster + 1:
                    type_edge = 'null'
                elif graph[i] > num_clusters:
                    type_edge = 'user'
                else:
                    type_edge = 'system'
                
                triplet = (graph[i], type_edge, graph[i + 1])
                if triplet not in data:
                    data[triplet] = []
                j1 = node_types_dict[graph[i]]
                node_types_dict[graph[i]] += 1
                j2 = node_types_dict[graph[i + 1]]
                data[triplet].append(torch.tensor([j1, j2]))
            node_types_dict[graph[top_k - 1]] += 1
        
#         node_features = {}
#         for i in node_types_dict.keys():
#             if node_types_dict[i] != 0:
#                 node_features[i] = torch.tensor(np.full((node_types_dict[i], embs_dim), embs[i])).float()
            
        g = dgl.heterograph(data)
#         g.ndata['h'] = node_features

        all_batches.append((g, node_features, labels))
    return all_batches

class BaseModel(nn.Module, metaclass=ABCMeta):
    @classmethod
    def build_model_from_args(cls, args, hg):
        r"""
        Build the model instance from args and hg.

        So every subclass inheriting it should override the method.
        """
        raise NotImplementedError("Models must implement the build_model_from_args method")

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *args):
        r"""
        The model plays a role of encoder. So the forward will encoder original features into new features.

        Parameters
        -----------
        hg : dgl.DGlHeteroGraph
            the heterogeneous graph
        h_dict : dict[str, th.Tensor]
            the dict of heterogeneous feature

        Return
        -------
        out_dic : dict[str, th.Tensor]
            A dict of encoded feature. In general, it should ouput all nodes embedding.
            It is allowed that just output the embedding of target nodes which are participated in loss calculation.
        """
        raise NotImplementedError

    def extra_loss(self):
        r"""
        Some model want to use L2Norm which is not applied all parameters.

        Returns
        -------
        th.Tensor
        """
        raise NotImplementedError

    def h2dict(self, h, hdict):
        pre = 0
        out_dict = {}
        for i, value in hdict.items():
            out_dict[i] = h[pre:value.shape[0]+pre]
            pre += value.shape[0]
        return out_dict

    def get_emb(self):
        r"""
        Return the embedding of a model for further analysis.

        Returns
        -------
        numpy.array
        """
        raise NotImplementedError

def transform_relation_graph_list(hg, identity=True):
    r"""
        extract subgraph :math:`G_i` from :math:`G` in which
        only edges whose type :math:`R_i` belongs to :math:`\mathcal{R}`
        Parameters
        ----------
            hg : dgl.heterograph
                Input heterogeneous graph
            category : string
                Type of predicted nodes.
            identity : bool
                If True, the identity matrix will be added to relation matrix set.
    """

    # get target category id
    etype = np.array(hg.etypes)
    etype[etype == 'user'] = 0
    etype[etype == 'system'] = 1
    etype[etype == 'null'] = 2

    etype = etype.astype(int)
    etype = torch.tensor(etype)
    
    g = dgl.to_homogeneous(hg)
    # find out the target node ids in g

    edges = g.edges()
    ctx = g.device
    #g.edata['w'] = th.ones(g.num_edges(), device=ctx)
    num_edge_type = th.max(etype).item()

    # norm = EdgeWeightNorm(norm='right')
    # edata = norm(g.add_self_loop(), th.ones(g.num_edges() + g.num_nodes(), device=ctx))
    graph_list = []
    for i in range(num_edge_type + 1):
        e_ids = th.nonzero(etype == i).squeeze(-1)
        sg = dgl.graph((edges[0][e_ids], edges[1][e_ids]), num_nodes=g.num_nodes())
        # sg.edata['w'] = edata[e_ids]
        sg.edata['w'] = th.ones(sg.num_edges(), device=ctx)
        graph_list.append(sg)
    if identity == True:
        x = th.arange(0, g.num_nodes(), device=ctx)
        sg = dgl.graph((x, x))
        # sg.edata['w'] = edata[g.num_edges():]
        sg.edata['w'] = th.ones(g.num_nodes(), device=ctx)
        graph_list.append(sg)
        
    return graph_list, th.arange(g.num_nodes())

class fastGTN(BaseModel):
    r"""
        fastGTN from paper `Graph Transformer Networks: Learning Meta-path Graphs to Improve GNNs
        <https://arxiv.org/abs/2106.06218>`__.
        It is the extension paper  of GTN.
        `Code from author <https://github.com/seongjunyun/Graph_Transformer_Networks>`__.

        Given a heterogeneous graph :math:`G` and its edge relation type set :math:`\mathcal{R}`.Then we extract
        the single relation adjacency matrix list. In that, we can generate combination adjacency matrix by conv
        the single relation adjacency matrix list. We can generate :math:'l-length' meta-path adjacency matrix
        by multiplying combination adjacency matrix. Then we can generate node representation using a GCN layer.

        Parameters
        ----------
        num_edge_type : int
            Number of relations.
        num_channels : int
            Number of conv channels.
        in_dim : int
            The dimension of input feature.
        hidden_dim : int
            The dimension of hidden layer.
        num_class : int
            Number of classification type.
        num_layers : int
            Length of hybrid metapath.
        category : string
            Type of predicted nodes.
        norm : bool
            If True, the adjacency matrix will be normalized.
        identity : bool
            If True, the identity matrix will be added to relation matrix set.

    """

    def __init__(self, num_edge_type, num_channels, in_dim, hidden_dim, num_class, num_layers, norm,
                 identity, top_k):
        super(fastGTN, self).__init__()
        self.num_edge_type = num_edge_type
        self.num_channels = num_channels
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        self.identity = identity
        self.top_k = top_k

        layers = []
        for i in range(num_layers):
            layers.append(GTConv(num_edge_type, num_channels))
        self.params = nn.ParameterList()
        for i in range(num_channels):
            self.params.append(nn.Parameter(th.Tensor(in_dim, hidden_dim)))
        self.layers = nn.ModuleList(layers)
        self.gat = GATConv_Grad(hidden_dim, hidden_dim, 1)
        self.norm = EdgeWeightNorm(norm='right')
#         self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.num_class)
        linear_weights = np.zeros(self.top_k)
        linear_weights[...] = 1 / self.top_k
        linear_weights = torch.tensor(linear_weights).view(1, -1)
        self.linear_weights = nn.Embedding.from_pretrained(linear_weights.float()).requires_grad_(True) 
        self.category_idx = None
        self.A = None
        self.h = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.params is not None:
            for para in self.params:
                nn.init.xavier_uniform_(para)

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            g = H[i]
            g = dgl.remove_self_loop(g)
            g.edata['w_sum'] = self.norm(g, g.edata['w_sum'])
            norm_H.append(g)
        return norm_H

    def forward(self, hg, h):
        # * =============== Extract edges in original graph ================
        self.A, self.category_idx = transform_relation_graph_list(hg,identity=self.identity)
        # выколупать h из словарика и передавать в лоб!!!
#         print(h[0][0])
        # X_ = self.gcn(g, self.h)
        A = self.A
        # * =============== Get new graph structure ================
        H = []
        for n_c in range(self.num_channels):
            H.append(th.matmul(h, self.params[n_c]))
        res_H = None
        for i in range(self.num_layers):
            hat_A = self.layers[i](A)

            for n_c in range(self.num_channels):
#                     print("^", len(H[n_c]), hat_A[n_c])
                edge_weight = self.norm(hat_A[n_c], hat_A[n_c].edata['w_sum'])
                H[n_c] = self.gat(hat_A[n_c], H[n_c], edge_weights=edge_weight)
                H[n_c] = H[n_c].view(-1, self.hidden_dim)
                
        for n_c in range(self.num_channels):
            if n_c == 0:
                res_H = F.relu(H[n_c])
            else:
                res_H = res_H + F.relu(H[n_c])
        res_H /= self.num_channels
            
#         X_ = self.linear1(res_H)
#         X_ = F.relu(X_)
        
        feat = torch.reshape(res_H, (len(res_H) // self.top_k, self.top_k, self.hidden_dim))        
        linear_weights_1dim = torch.reshape(self.linear_weights.weight, (self.top_k, ))
        get_sum = lambda e: torch.matmul(linear_weights_1dim, e)
        feat = list(map(get_sum, feat))
        hg = torch.stack(feat)

        y = self.linear2(hg)
        return y


import dgl.function as fn
from dgl import DGLError
from dgl.nn.pytorch.conv.gatconv import GATConv
from dgl.ops import edge_softmax


class GATConv_Grad(GATConv):

    def forward(self, graph, feat, edge_weights, get_attention=False):
        
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
           
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

            # @author haoran
            # --- compute grad start
            graph.edata['edge_weights'] = edge_weights
            graph.update_all(fn.u_mul_e('ft', 'edge_weights', 'mul'), fn.sum('mul', 'ft'))
            # --- compute grad end
            
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
                
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class GTConv(nn.Module):
    r"""
        We conv each sub adjacency matrix :math:`A_{R_{i}}` to a combination adjacency matrix :math:`A_{1}`:

        .. math::
            A_{1} = conv\left(A ; W_{c}\right)=\sum_{R_{i} \in R} w_{R_{i}} A_{R_{i}}

        where :math:`R_i \subseteq \mathcal{R}` and :math:`W_{c}` is the weight of each relation matrix
    """

    def __init__(self, in_channels, out_channels, softmax_flag=True):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(th.Tensor(out_channels, in_channels))
        self.softmax_flag = softmax_flag
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, A):
        if self.softmax_flag:
            Filter = F.softmax(self.weight, dim=1)
        else:
            Filter = self.weight
        
        num_channels = Filter.shape[0]
        results = []
#         print(Filter.shape, len(A))
        for i in range(num_channels):
            for j, g in enumerate(A):
                A[j].edata['w_sum'] = g.edata['w'] * Filter[i][j]
            sum_g = dgl.adj_sum_graph(A, 'w_sum')
            results.append(sum_g)
        return results
