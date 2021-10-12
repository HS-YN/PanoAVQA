# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """
import math

import torch
from torch import nn


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": nn.functional.relu, "swish": swish}


class BertSelfAttention(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        if model_config.hidden_size % model_config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (model_config.hidden_size, model_config.num_attention_heads)
            )
        self.num_attention_heads = model_config.num_attention_heads
        self.attention_head_size = int(model_config.hidden_size / model_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(model_config.hidden_size, self.all_head_size)
        self.key = nn.Linear(model_config.hidden_size, self.all_head_size)
        self.value = nn.Linear(model_config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(model_config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        context,
        attention_mask=None,
    ):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(context))
        value_layer = self.transpose_for_scores(self.value(context))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.dense = nn.Linear(model_config.hidden_size, model_config.hidden_size)
        self.LayerNorm = nn.LayerNorm(model_config.hidden_size, eps=model_config.layer_norm_eps)
        self.dropout = nn.Dropout(model_config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.self = BertSelfAttention(model_config)
        self.output = BertSelfOutput(model_config)

    def forward(self, hidden_states, attention_mask=None):
        self_output = self.self(hidden_states, hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output


class BertCrossAttention(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.self = BertSelfAttention(model_config)
        self.output = BertSelfOutput(model_config)

    def forward(self, hidden_states, context_states, context_mask=None):
        self_outputs = self.self(hidden_states, context_states, context_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.dense = nn.Linear(model_config.hidden_size, model_config.intermediate_size)
        if isinstance(model_config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[model_config.hidden_act]
        else:
            self.intermediate_act_fn = model_config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.dense = nn.Linear(model_config.intermediate_size, model_config.hidden_size)
        self.LayerNorm = nn.LayerNorm(model_config.hidden_size, eps=model_config.layer_norm_eps)
        self.dropout = nn.Dropout(model_config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, model_config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(model_config)
        self.intermediate = BertIntermediate(model_config)
        self.output = BertOutput(model_config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
