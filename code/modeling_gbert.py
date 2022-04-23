import torch
import math
from torch import nn
import torch.nn.functional as F
from config import BertConfig
from gnn_model import AggregatedEmbeddings
import os
import logging

logger = logging.getLogger(__name__)
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

class ResidualConnection(nn.Module):
    def __init__(self, config: BertConfig):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

def attention(query, key, value, d_k, mask=None, dropout=None):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    return torch.matmul(scores, value)

class MultiHeadedAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        self.d_k = config.hidden_size // config.num_attention_heads
        self.h = config.num_attention_heads

        self.linear_layers = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.hidden_size, bias=False) for _ in range(3)])
        self.output_linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x = attention(query, key, value, self.d_k, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, config: BertConfig):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2= nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.dropout(gelu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class BertEncoderLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.multiattention = MultiHeadedAttention(config)
        self.feed_forward = FeedForward(config)
        self.residual_attention = ResidualConnection(config)
        self.residual_forward= ResidualConnection(config)

    def forward(self, x, mask):
        x = self.residual_attention(
            x, lambda _x: self.multiattention.forward(_x, _x, _x, mask))
        x = self.residual_forward(x, self.feed_forward)
        return x

class BertEmbeddings(nn.Module):

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.voc_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.type_embeddings = nn.Embedding(2, config.hidden_size)

    def forward(self, input_ids, type_ids=None):
        if type_ids is None:
            type_ids = torch.zeros_like(input_ids)

        bert_embeddings = self.voc_embeddings(input_ids) + self.type_embeddings(type_ids)

        return bert_embeddings


class GbertModel(nn.Module):
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super(GbertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. ")
        self.config = config

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @classmethod
    def from_pretrained(cls, pretrained_model_name,  *inputs, **kwargs):
        print("-------------")
        print(pretrained_model_name)
        # Load config
        config_file = os.path.join(pretrained_model_name, CONFIG_NAME)
        weights_path = os.path.join(pretrained_model_name, WEIGHTS_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        model.load_state_dict(state_dict, False)
        return model


class Gbert(GbertModel):
    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None):

        super().__init__(config)
        if config.graph:
            assert dx_voc is not None
            assert rx_voc is not None

        # embedding for BERT, sum of voc, token embeddings
        self.embedding = AggregatedEmbeddings(
            config, dx_voc, rx_voc) if config.graph else BertEmbeddings(config)

        # multi-layers transformer blocks, deep network
        self.encoder = nn.ModuleList(
            [BertEncoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.apply(self.init_weights)

    def forward(self, x, token_type_ids=None):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # 0:[Pad] 1 [Mask]
        mask = (x > 1).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, token_type_ids)

        # running over multiple transformer blocks
        for encoderLayer in self.encoder:
            x = encoderLayer.forward(x, mask)
            
        return x, x[:, 0]