import datasets
import os
import torch
from torch.utils.data import Subset, DataLoader

from tqdm import tqdm
import csv
import wandb
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html')
# get_ipython().system('pip install torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html')
# get_ipython().system('pip install torch-geometric')


# In[2]:


# get_ipython().system('pip freeze > env.txt')


# In[3]:


import random
import json
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torch.optim import Adam

from transformers import AutoTokenizer
from transformers.activations import ACT2FN

import datasets
from tqdm import tqdm
import wandb

from torch_geometric.nn import GCNConv, GATConv
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertOnlyMLMHead
from transformers.modeling_outputs import MaskedLMOutput
from torch.nn import CrossEntropyLoss, MSELoss
import math
from sklearn.metrics import f1_score
# 


# In[4]:


# /kaggle/input/amalrec-test1/amalrec_dev.json


# In[5]:


class CrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states=None,
            past_key_value=None,
            attention_mask=None,
            output_attentions: bool = False,
            extra_attn=None,
            only_attn=False,
    ):
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if extra_attn is not None:
            attn_weights += extra_attn

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions or only_attn:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        if only_attn:
            return attn_weights_reshaped

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
                .transpose(1, 2)
                .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


# In[6]:


class GraphAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states=None,
            past_key_value=None,
            attention_mask=None,
            output_attentions: bool = False,
            extra_attn=None,
            only_attn=False,
    ):
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if extra_attn is not None:
            attn_weights += extra_attn

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        if only_attn:
            return attn_weights_reshaped

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
                .transpose(1, 2)
                .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class GraphLayer(nn.Module):
    def __init__(self, config, graph_type):
        super(GraphLayer, self).__init__()
        self.config = config

        self.graph_type = graph_type
        if self.graph_type == 'graphormer':
            self.graph = GraphAttention(config.hidden_size, config.num_attention_heads,
                                        config.attention_probs_dropout_prob)
        elif self.graph_type == 'GCN':
            self.graph = GCNConv(config.hidden_size, config.hidden_size)
        elif self.graph_type == 'GAT':
            self.graph = GATConv(config.hidden_size, config.hidden_size, 1)

        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.dropout = config.attention_probs_dropout_prob
        self.activation_fn = ACT2FN[config.hidden_act]
        self.activation_dropout = config.hidden_dropout_prob
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, label_emb, extra_attn):
        residual = label_emb
        if self.graph_type == 'graphormer':
            label_emb, attn_weights, _ = self.graph(
                hidden_states=label_emb, attention_mask=None, output_attentions=False,
                extra_attn=extra_attn,
            )
            label_emb = nn.functional.dropout(label_emb, p=self.dropout, training=self.training)
            label_emb = residual + label_emb
            label_emb = self.layer_norm(label_emb)

            residual = label_emb
            label_emb = self.activation_fn(self.fc1(label_emb))
            label_emb = nn.functional.dropout(label_emb, p=self.activation_dropout, training=self.training)
            label_emb = self.fc2(label_emb)
            label_emb = nn.functional.dropout(label_emb, p=self.dropout, training=self.training)
            label_emb = residual + label_emb
            label_emb = self.final_layer_norm(label_emb)
        elif self.graph_type == 'GCN' or self.graph_type == 'GAT':
            label_emb = self.graph(label_emb.squeeze(0), edge_index=extra_attn)
            label_emb = nn.functional.dropout(label_emb, p=self.dropout, training=self.training)
            label_emb = residual + label_emb
            label_emb = self.layer_norm(label_emb)
        else:
            raise NotImplementedError
        return label_emb


class GraphEncoder(nn.Module):
    def __init__(self, config, graph_type='GAT', layer=1, path_list=None, data_path=None):
        super(GraphEncoder, self).__init__()
        self.config = config
        self.hir_layers = nn.ModuleList([GraphLayer(config, graph_type) for _ in range(layer)])

        self.label_num = config.num_labels - 3
        self.graph_type = graph_type

#         self.label_dict = torch.load(os.path.join(data_path, 'value_dict.pt'))
        self.label_dict = torch.load('data/value_dict_formatted.pt')
        self.tokenizer = AutoTokenizer.from_pretrained(config.name_or_path)

        if self.graph_type == 'graphormer':
            self.inverse_label_list = {}

            def get_root(path_list, n):
                ret = []
                while path_list[n] != n:
                    ret.append(n)
                    n = path_list[n]
                ret.append(n)
                return ret

            for i in range(self.label_num):
                self.inverse_label_list.update({i: get_root(path_list, i)})
            label_range = torch.arange(len(self.inverse_label_list))
            self.label_id = label_range
            node_list = {}

            def get_distance(node1, node2):
                p = 0
                q = 0
                node_list[(node1, node2)] = a = []
                node1 = self.inverse_label_list[node1]
                node2 = self.inverse_label_list[node2]
                while p < len(node1) and q < len(node2):
                    if node1[p] > node2[q]:
                        a.append(node1[p])
                        p += 1

                    elif node1[p] < node2[q]:
                        a.append(node2[q])
                        q += 1

                    else:
                        break
                return p + q

            self.distance_mat = self.label_id.reshape(1, -1).repeat(self.label_id.size(0), 1)
            hier_mat_t = self.label_id.reshape(-1, 1).repeat(1, self.label_id.size(0))
            self.distance_mat.map_(hier_mat_t, get_distance)
            self.distance_mat = self.distance_mat.view(1, -1)
            self.edge_mat = torch.zeros(self.label_num, self.label_num, 15,
                                        dtype=torch.long)
            for i in range(self.label_num):
                for j in range(self.label_num):
                    self.edge_mat[i, j, :len(node_list[(i, j)])] = torch.tensor(node_list[(i, j)])
            self.edge_mat = self.edge_mat.view(-1, self.edge_mat.size(-1))

            self.id_embedding = nn.Embedding(self.label_num, config.hidden_size, 0)
            self.distance_embedding = nn.Embedding(20, 1, 0)
            self.edge_embedding = nn.Embedding(self.label_num, 1, 0)
            self.label_id = nn.Parameter(self.label_id, requires_grad=False)
            self.edge_mat = nn.Parameter(self.edge_mat, requires_grad=False)
            self.distance_mat = nn.Parameter(self.distance_mat, requires_grad=False)
            self.label_name = []
            for i in range(len(self.label_dict)):
                self.label_name.append(self.label_dict[i])
            self.label_name = self.tokenizer(self.label_name, padding='longest')['input_ids']
            self.label_name = nn.Parameter(torch.tensor(self.label_name, dtype=torch.long), requires_grad=False)
        else:
            self.path_list = nn.Parameter(torch.tensor(path_list).transpose(0, 1), requires_grad=False)

    def forward(self, label_emb, embeddings):
        extra_attn = None

        if self.graph_type == 'graphormer':
            label_mask = self.label_name != self.tokenizer.pad_token_id
            # full name
            label_name_emb = embeddings(self.label_name)
            label_emb = label_emb + (label_name_emb * label_mask.unsqueeze(-1)).sum(dim=1) / label_mask.sum(dim=1).unsqueeze(-1)

            label_emb = label_emb + self.id_embedding(self.label_id[:, None]).view(-1,
                                                                        self.config.hidden_size)
            extra_attn = self.distance_embedding(self.distance_mat) + self.edge_embedding(self.edge_mat).sum(
                dim=1) / (self.distance_mat.view(-1, 1) + 1e-8)
            extra_attn = extra_attn.view(self.label_num, self.label_num)
        elif self.graph_type == 'GCN' or self.graph_type == 'GAT':
            extra_attn = self.path_list

        for hir_layer in self.hir_layers:
            label_emb = hir_layer(label_emb.unsqueeze(0), extra_attn)

        return label_emb.squeeze(0)


# In[7]:


def multilabel_categorical_crossentropy(y_true, y_pred):
    loss_mask = y_true != -100
    y_true = y_true.masked_select(loss_mask).view(-1, y_pred.size(-1))
    y_pred = y_pred.masked_select(loss_mask).view(-1, y_true.size(-1))
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[:, :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


# In[8]:


# from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertOnlyMLMHead
# from transformers.modeling_outputs import (
#     MaskedLMOutput
# )
# from transformers.activations import ACT2FN
# from torch.nn import CrossEntropyLoss, MSELoss
# import torch.nn as nn
# import torch
# from transformers import AutoTokenizer
# import os
# # from .loss import multilabel_categorical_crossentropy
# # from .graph import GraphEncoder
# # from .attention import CrossAttention
# import math
# import torch.nn.functional as F
# from sklearn.metrics import f1_score


class GraphEmbedding(nn.Module):
    def __init__(self, config, embedding, new_embedding, graph_type='GAT', layer=1, path_list=None, data_path=None):
        super(GraphEmbedding, self).__init__()
        self.graph_type = graph_type
        padding_idx = config.pad_token_id
        self.num_class = config.num_labels
        if self.graph_type != '':
            self.graph = GraphEncoder(config, graph_type, layer, path_list=path_list, data_path=data_path)
        self.padding_idx = padding_idx
        self.original_embedding = embedding
        new_embedding = torch.cat(
            [torch.zeros(1, new_embedding.size(-1), device=new_embedding.device, dtype=new_embedding.dtype),
             new_embedding], dim=0)
        self.new_embedding = nn.Embedding.from_pretrained(new_embedding, False, 0)
        self.size = self.original_embedding.num_embeddings + self.new_embedding.num_embeddings - 1
        self.depth = (self.new_embedding.num_embeddings - 2 - self.num_class)

    @property
    def weight(self):
        def foo():
            # label prompt MASK
            edge_features = self.new_embedding.weight[1:, :]
            if self.graph_type != '':
                # label prompt
                edge_features = edge_features[:-1, :]
                edge_features = self.graph(edge_features, self.original_embedding)
                edge_features = torch.cat(
                    [edge_features, self.new_embedding.weight[-1:, :]], dim=0)
            return torch.cat([self.original_embedding.weight, edge_features], dim=0)

        return foo

    @property
    def raw_weight(self):
        def foo():
            return torch.cat([self.original_embedding.weight, self.new_embedding.weight[1:, :]], dim=0)

        return foo

    def forward(self, x):
        x = F.embedding(x, self.weight(), self.padding_idx)

        return x


class OutputEmbedding(nn.Module):
    def __init__(self, bias):
        super(OutputEmbedding, self).__init__()
        self.weight = None
        self.bias = bias

    def forward(self, x):
        return F.linear(x, self.weight(), self.bias)


class Prompt(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, graph_type='GAT', layer=1, path_list=None, data_path=None, depth2label=None, **kwargs):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        self.cls = BertOnlyMLMHead(config)
        self.num_labels = config.num_labels
        self.multiclass_bias = nn.Parameter(torch.zeros(self.num_labels, dtype=torch.float32))
        bound = 1 / math.sqrt(768)
        nn.init.uniform_(self.multiclass_bias, -bound, bound)
        self.data_path = data_path
        self.graph_type = graph_type
        self.vocab_size = self.tokenizer.vocab_size
        self.path_list = path_list
        self.depth2label = depth2label
        self.layer = layer
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def init_embedding(self):
        depth = len(self.depth2label)
        label_dict = torch.load('data/value_dict_formatted.pt')
        tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        label_dict = {i: tokenizer.encode(v) for i, v in label_dict.items()}
        label_emb = []
        input_embeds = self.get_input_embeddings()
        for i in range(len(label_dict)):
            label_emb.append(
                input_embeds.weight.index_select(0, torch.tensor(label_dict[i], device=self.device)).mean(dim=0))
        prefix = input_embeds(torch.tensor([tokenizer.mask_token_id],
                                           device=self.device, dtype=torch.long))
        # prompt
        prompt_embedding = nn.Embedding(depth + 1,
                                        input_embeds.weight.size(1), 0)

        self._init_weights(prompt_embedding)
        # label prompt mask
        label_emb = torch.cat(
            [torch.stack(label_emb), prompt_embedding.weight[1:, :], prefix], dim=0)
        embedding = GraphEmbedding(self.config, input_embeds, label_emb, self.graph_type,
                                   path_list=self.path_list, layer=self.layer, data_path=self.data_path)
        self.set_input_embeddings(embedding)
        output_embeddings = OutputEmbedding(self.get_output_embeddings().bias)
        self.set_output_embeddings(output_embeddings)
        output_embeddings.weight = embedding.raw_weight
        self.vocab_size = output_embeddings.bias.size(0)
        output_embeddings.bias.data = nn.functional.pad(
            output_embeddings.bias.data,
            (
                0,
                embedding.size - output_embeddings.bias.shape[0],
            ),
            "constant",
            0,
        )

    def get_layer_features(self, layer, prompt_feature=None):
        labels = torch.tensor(self.depth2label[layer], device=self.device) + 1
        label_features = self.get_input_embeddings().new_embedding(labels)
        label_features = self.transform(label_features)
        label_features = torch.dropout(F.relu(label_features), train=self.training, p=self.config.hidden_dropout_prob)
        return label_features

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)
        single_labels = input_ids.masked_fill(multiclass_pos | (input_ids == self.config.pad_token_id), -100)
        if self.training:
            enable_mask = input_ids < self.tokenizer.vocab_size
            random_mask = torch.rand(input_ids.shape, device=input_ids.device) * attention_mask * enable_mask
            input_ids = input_ids.masked_fill(random_mask > 0.865, self.tokenizer.mask_token_id)
            random_ids = torch.randint_like(input_ids, 104, self.vocab_size)
            mlm_mask = random_mask > 0.985
            input_ids = input_ids * mlm_mask.logical_not() + random_ids * mlm_mask
            mlm_mask = random_mask < 0.85
            single_labels = single_labels.masked_fill(mlm_mask, -100)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)),
                                      single_labels.view(-1))
            multiclass_logits = prediction_scores.masked_select(
                multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))).view(-1,
                                                                                              prediction_scores.size(
                                                                                                  -1))
            multiclass_logits = multiclass_logits[:,
                                self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias
            multiclass_loss = multilabel_categorical_crossentropy(labels.view(-1, self.num_labels), multiclass_logits)
            masked_lm_loss += multiclass_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        ret = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return ret

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @torch.no_grad()
    def generate(self, input_ids, depth2label, **kwargs):
        attention_mask = input_ids != self.config.pad_token_id
        outputs = self(input_ids, attention_mask)
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)
        prediction_scores = outputs['logits']
        prediction_scores = prediction_scores.masked_select(
            multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))).view(-1,
                                                                                          prediction_scores.size(
                                                                                              -1))
        prediction_scores = prediction_scores[:,
                            self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias
        prediction_scores = prediction_scores.view(-1, len(depth2label), prediction_scores.size(-1))
        predict_labels = []
        for scores in prediction_scores:
            predict_labels.append([])

            for i, score in enumerate(scores): #for each layer 
                for l in depth2label[i]:
                    if score[l] > 0:
                        predict_labels[-1].append(l)
        return predict_labels, prediction_scores
        
    @torch.no_grad()
    def generate_with_p_10(self, input_ids, depth2label, **kwargs):
        # Create attention mask
        attention_mask = input_ids != self.config.pad_token_id
        
        # Get model outputs
        outputs = self(input_ids, attention_mask)
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)
        
        # Extract prediction scores from the model outputs
        prediction_scores = outputs['logits']
        
        # Mask the prediction scores
        prediction_scores = prediction_scores.masked_select(
            multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))
        ).view(-1, prediction_scores.size(-1))
        
        # Add multiclass bias and reshape
        prediction_scores = prediction_scores[:, self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias
        prediction_scores = prediction_scores.view(-1, len(depth2label), prediction_scores.size(-1))
        
        predict_labels = []  # List to store predicted labels
        
        for scores in prediction_scores:
            predict_labels.append([])
            scores_list = []
            
            for i, score in enumerate(scores):  # Iterate over each layer's scores
                for l in depth2label[i]:  # Iterate over labels associated with the current layer
                    if score[l] > 0:  # Check if score is positive
                        scores_list.append((l, score[l]))  # Collect label and its score

            # Sort the scores in descending order
            sorted_scores_list = sorted(scores_list, key=lambda x: x[1], reverse=True)
            
            # Get the top 10 relevant labels (or fewer if less than 10 are available)
            p_10_labels = min(10, len(sorted_scores_list))
            top_10_percent = sorted_scores_list[:p_10_labels]
            
            # Extract only the labels from the top scores
            selected_labels = [label for label, score in top_10_percent]
            predict_labels[-1].extend(selected_labels)  # Store the predicted labels for the current input

        return predict_labels, prediction_scores


# In[9]:

class ModelConfig:
    def __init__(self, args):
        self.lr = args.get("lr", 3e-5)
        self.data = args.get("data", "amalrec")
        self.batch = args.get("batch", 2)
        self.early_stop = args.get("early_stop", 6)
        self.device = args.get("device", "cuda")
        self.name = args.get("name", "test1")
        self.update = args.get("update", 1)
        self.model = args.get("model", "prompt")
        self.wandb = args.get("wandb", False)
        self.arch = args.get("arch", "bert-base-uncased")
        self.layer = args.get("layer", 1)
        self.graph = args.get("graph", "GAT")
        self.low_res = args.get("low_res", False)
        self.seed = args.get("seed", 3)

    def main(self):
        # Your logic here
        print(f"Learning Rate: {self.lr}")
        print(f"Data: {self.data}")
        print(f"Batch size: {self.batch}")
        print(f"Early stopping: {self.early_stop}")
        print(f"Device: {self.device}")
        print(f"Model Name: {self.name}")
        print(f"Update: {self.update}")
        print(f"Model Type: {self.model}")
        print(f"Wandb: {self.wandb}")
        print(f"Architecture: {self.arch}")
        print(f"Layer: {self.layer}")
        print(f"Graph Type: {self.graph}")
        print(f"Low Resolution: {self.low_res}")
        print(f"Seed: {self.seed}")
        
        # Example of a task, e.g., loading model, data, etc.
        # You can place your main code logic here.

def _precision_recall_f1(right, predict, total):
    """
    :param right: int, the count of right prediction
    :param predict: int, the count of prediction
    :param total: int, the count of labels
    :return: p(precision, Float), r(recall, Float), f(f1_score, Float)
    """
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict
    if total > 0:
        r = float(right) / total
    if p + r > 0:
        f = p * r * 2 / (p + r)
    return p, r, f

# Usage
cf = {
    "lr": 3e-5,
    "data": "amalrec",
    "batch": 8,
    "early_stop": 6,
    "device": "cuda",
    "name": "amalrec",
    "update": 1,
    "model": "prompt",
    "wandb": False,
    "arch": "bert-base-uncased",
    "layer": 1,
    "graph": "GAT",
    "low_res": False,
    "seed": 3
}
checkpoints_dir='checkpoints'
def evaluate(epoch_predicts, epoch_labels, id2label, threshold=0.5, top_k=10):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[int]], predicted label_id
    :param id2label: Dict[int, str], mapping from id to label
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return: confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'

    epoch_gold = epoch_labels

    # Initialize confusion matrix and counts
    num_classes = len(id2label)
    confusion_count_list = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    right_count_list = [0 for _ in range(num_classes)]
    gold_count_list = [0 for _ in range(num_classes)]
    predicted_count_list = [0 for _ in range(num_classes)]

    for sample_predict_id_list, sample_gold in zip(epoch_predicts, epoch_gold):
        for i in range(num_classes):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # Count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # Count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        label = label + '_' + str(i)

        # Calculate precision, recall, and F1 for each class
        precision, recall, fscore = _precision_recall_f1(right_count_list[i], predicted_count_list[i], gold_count_list[i])
        precision_dict[label] = precision
        recall_dict[label] = recall
        fscore_dict[label] = fscore
        
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Calculate macro and micro F1 scores
    precision_macro = sum(precision_dict.values()) / len(precision_dict) if precision_dict else 0.0
    recall_macro = sum(recall_dict.values()) / len(recall_dict) if recall_dict else 0.0
    macro_f1 = sum(fscore_dict.values()) / len(fscore_dict) if fscore_dict else 0.0
    
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total if gold_total > 0 else 0.0
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    # Store the scores for each class
    scores_per_class = {
        'micro_f1': precision_micro,  # Overall micro F1 score
        'precision_per_class': precision_dict,
        'recall_per_class': recall_dict,
        'fscore_per_class': fscore_dict
    }

    return {
        'precision': precision_micro,
        'recall': recall_micro,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'full': [precision_dict, recall_dict, fscore_dict, right_count_list, predicted_count_list, gold_count_list],
        'scores_per_class': scores_per_class  # Store scores for each class
    }
# Adopt from https://github.com/Alibaba-NLP/HiAGM/blob/master/train_modules/evaluation_metrics.py

def test_function(model,extra,test,args):
    checkpoint = torch.load(os.path.join('checkpoints', args.name, 'checkpoint_best{}.pt'.format(extra)),
                            map_location='cuda')
    model.load_state_dict(checkpoint['param'])
    model.eval()
    pred = []
    gold = []
    with torch.no_grad(), tqdm(test) as pbar:
        for i,batch in enumerate(pbar):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            output_ids, logits = model.generate(batch['input_ids'], depth2label=depth2label, )
            for out, g in zip(output_ids, batch['labels']):
                pred.append(set([i for i in out]))
                gold.append([])
                g = g.view(-1, num_class)
                for ll in g:
                    for i, l in enumerate(ll):
                        if l == 1:
                            gold[-1].append(i)
    scores = evaluate(pred, gold, label_dict)
    macro_f1 = scores['macro_f1']
    micro_f1 = scores['micro_f1']
    scores_per=scores['scores_per_class']
    print('macro', macro_f1, 'micro', micro_f1)
    with open(os.path.join( checkpoints_dir, args.name, f'result_test.json'), 'w') as f:
            print(scores, file=f)
            prefix = 'test' + extra
    if args.wandb:
        wandb.log({prefix + '_macro': macro_f1, prefix + '_micro': micro_f1})

def predict_function(model, extra, test, args):
    
    checkpoint_path = os.path.join('checkpoints', args.name, f'checkpoint_best{extra}.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['param'])
    model.eval() 
    pred = [] 
    with torch.no_grad(), tqdm(test, desc="Predicting") as pbar:
        for batch in pbar:
            batch = {k: v.to('cuda') for k, v in batch.items()}
            output_ids, _ = model.generate(batch['input_ids'], depth2label=depth2label)
            for out in output_ids:
                pred.append(set(out.tolist()))  

    # Save predictions to a CSV file
    with open('predictions.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Prediction'])  
        for p in pred:
            writer.writerow([list(p)]) 

config = ModelConfig(cf)

label_dict = torch.load('data/value_dict_formatted.pt')
label_dict = {i: v for i, v in label_dict.items()}

slot2value = torch.load('data/slot.pt')
value2slot = {}
num_class = 0
for s in slot2value:
    for v in slot2value[s]:
        value2slot[v] = s
        if num_class < v:
            num_class = v
num_class += 1
path_list = [(i, v) for v, i in value2slot.items()]
for i in range(num_class):
    if i not in value2slot:
        value2slot[i] = -1


def get_depth(x):
    depth = 0
    while value2slot[x] != -1:
        depth += 1
        x = value2slot[x]
    return depth


depth_dict = {i: get_depth(i) for i in range(num_class)}
max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}

for depth in depth2label:
    for l in depth2label[depth]:
        path_list.append((num_class + depth, l))

                                   
def main(args):
    data_path = os.path.join('data', args.data)
    model=Prompt.from_pretrained(args.arch, num_labels=len(label_dict), path_list=path_list, layer=args.layer,graph_type=args.graph, data_path=data_path, depth2label=depth2label,)
    model.init_embedding()
    model.to('cuda')
    args.name = args.data + '-' + args.name
    dataset = datasets.load_from_disk(os.path.join(data_path, args.model))
    dataset['test'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
    test = DataLoader(dataset['test'], batch_size=8, shuffle=False)
    test_function(model,'_macro',test,args)

main(config)