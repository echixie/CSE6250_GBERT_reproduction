import torch
from torch import nn
import torch.nn.functional as F
from modeling_gbert import Gbert, GbertModel
from config import BertConfig


class RxPredictLayer(nn.Module):
    def __init__(self, config: BertConfig, tokenizer):
        super(RxPredictLayer, self).__init__()
        self.mlp_dx = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU())
        self.mlp_rx = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU())
        self.mlp_predictor = nn.Sequential(nn.Linear(3*config.hidden_size, 2*config.hidden_size),
                                 nn.ReLU(), nn.Linear(2*config.hidden_size, len(tokenizer.rx_voc_multi.word2idx)))

    def forward(self, input, num_pre_visit):
        bert_cls = input.view(2, -1, input.size(1))  # (2, adm, H)
        dx_bert_cls = self.mlp_dx(bert_cls[0])  # (adm, H)
        rx_bert_cls = self.mlp_rx(bert_cls[1])  # (adm, H)

        # mean and concat for rx prediction task
        rx_logits = []
        for i in range(num_pre_visit):
            # mean
            dx_mean = torch.mean(dx_bert_cls[0:i+1, :], dim=0, keepdim=True)
            rx_mean = torch.mean(rx_bert_cls[0:i+1, :], dim=0, keepdim=True)
            # concat
            concat = torch.cat(
                [dx_mean, rx_mean, dx_bert_cls[i+1, :].unsqueeze(dim=0)], dim=-1)
            rx_logits.append(self.mlp_predictor(concat))

        rx_logits = torch.cat(rx_logits, dim=0)
        return rx_logits

class GbertRxPredict(GbertModel):
    def __init__(self, config: BertConfig, tokenizer):
        super(GbertRxPredict, self).__init__(config)
        self.bert = Gbert(config, tokenizer.dx_voc, tokenizer.rx_voc)
        self.rx_predictor = RxPredictLayer(config, tokenizer)
        self.apply(self.init_weights)

    def forward(self, input_ids, dx_labels=None, rx_labels=None, epoch=None):

        token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
            (1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        token_types_ids = token_types_ids.repeat(
            1 if input_ids.size(0)//2 == 0 else input_ids.size(0)//2, 1)
        # bert_pool: (2*adm, H)
        _, bert_cls = self.bert(input_ids, token_types_ids)
        rx_logits = self.rx_predictor(bert_cls, rx_labels.size(0))
        loss = F.binary_cross_entropy_with_logits(rx_logits, rx_labels)
        return loss, rx_logits