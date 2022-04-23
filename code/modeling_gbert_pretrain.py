import torch
from torch import nn
import torch.nn.functional as F
from modeling_gbert import Gbert, GbertModel
from config import BertConfig


class PredictLayer(nn.Module):
    def __init__(self, config: BertConfig, voc_size):
        super(PredictLayer, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(
        ), nn.Linear(config.hidden_size, voc_size))

    def forward(self, input):
        return self.mlp(input)


class SelfPredict(nn.Module):
    def __init__(self, config: BertConfig, dx_voc_size, rx_voc_size):
        super(SelfPredict, self).__init__()
        self.predict_dx = PredictLayer(config, dx_voc_size)
        self.predict_rx = PredictLayer(config, rx_voc_size)

    def forward(self, dx_inputs, rx_inputs):

        return self.predict_dx(dx_inputs), self.predict_rx(rx_inputs)


class DualPredict(nn.Module):
    def __init__(self, config: BertConfig, dx_voc_size, rx_voc_size):
        super(DualPredict, self).__init__()
        self.predict_dx = PredictLayer(config, dx_voc_size)
        self.predict_rx = PredictLayer(config, rx_voc_size)

    def forward(self, dx_inputs, rx_inputs):
        return self.predict_dx(rx_inputs), self.predict_rx(dx_inputs)

class GbertPretrain(GbertModel):
    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None):
        super(GbertPretrain, self).__init__(config)
        self.dx_voc_size = len(dx_voc.word2idx)
        self.rx_voc_size = len(rx_voc.word2idx)

        self.bert = Gbert(config, dx_voc, rx_voc)
        self.sp = SelfPredict(config, self.dx_voc_size, self.rx_voc_size)
        self.dp = DualPredict(config, self.dx_voc_size, self.rx_voc_size)

        self.apply(self.init_weights)

    def forward(self, inputs, dx_labels=None, rx_labels=None):
        _, dx_bert_cls = self.bert(inputs[:, 0, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))
        _, rx_bert_cls = self.bert(inputs[:, 1, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))

        dx2dx, rx2rx = self.sp(dx_bert_cls, rx_bert_cls)
        rx2dx, dx2rx = self.dp(dx_bert_cls, rx_bert_cls)

        if rx_labels is None or dx_labels is None:
            return F.sigmoid(dx2dx), F.sigmoid(rx2dx), F.sigmoid(dx2rx), F.sigmoid(rx2rx)
        else:
            loss = F.binary_cross_entropy_with_logits(dx2dx, dx_labels) + \
                F.binary_cross_entropy_with_logits(rx2dx, dx_labels) + \
                F.binary_cross_entropy_with_logits(dx2rx, rx_labels) + \
                F.binary_cross_entropy_with_logits(rx2rx, rx_labels)
            return loss, F.sigmoid(dx2dx), F.sigmoid(rx2dx), F.sigmoid(dx2rx), F.sigmoid(rx2rx)
