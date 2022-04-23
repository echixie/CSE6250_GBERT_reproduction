import torch
import torch.nn as nn

from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv import GATConv

from graph_builder import build_tree_edges
from graph_builder import build_tree_med, build_tree_diag


class OntologyEmbedding(nn.Module):
    def __init__(self, voc, code_type, in_channels=100, out_channels=20, heads=5):
        super(OntologyEmbedding, self).__init__()

        # build tree nodes
        words = list(voc.word2idx.keys())
        nodes_list, tree_voc = build_tree_med(words) if code_type == 'med' else build_tree_diag(words)
        self.to_children_edges = torch.tensor(build_tree_edges(nodes_list, tree_voc, True))
        self.to_ancestor_edges = torch.tensor(build_tree_edges(nodes_list, tree_voc, False))

        # embedding as trainable parameters
        self.embedding = nn.Parameter(torch.Tensor(len(tree_voc.word2idx), in_channels))
        glorot(self.embedding)

        # find indexes for initial words in the built tree
        self.word_indexes = []
        for word in voc.word2idx.keys():
            self.word_indexes.append(tree_voc.word2idx[word])

        # initialize GAT model
        self.g = GATConv(in_channels, out_channels, heads)

    def forward(self):
        embedding = self.embedding
        x = self.g(embedding, self.to_children_edges.to(embedding.device))
        embedding = self.g(x, self.to_ancestor_edges.to(embedding.device))

        return embedding[self.word_indexes]

    def init_params(self):
        glorot(self.embedding)



class CombinedEmbeddings(nn.Module):
    """
    combaine medication and diagnosis ontology embedding together
    """

    def __init__(self, config, med_voc, diag_voc):
        super(CombinedEmbeddings, self).__init__()

        self.special_embedding = nn.Parameter(
            torch.Tensor(config.vocab_size - len(med_voc.idx2word) - len(diag_voc.idx2word), config.hidden_size))
        self.rx_embedding = OntologyEmbedding(med_voc, 'med', config.hidden_size, config.graph_hidden_size, config.graph_heads)
        self.dx_embedding = OntologyEmbedding(diag_voc, 'diag',config.hidden_size, config.graph_hidden_size, config.graph_heads)

        self.init_params()

    def forward(self, ids):
        """
        :param ids: [B, L]
        """
        embeddings = torch.cat([self.special_embedding, self.rx_embedding(), self.dx_embedding()], dim=0)
        return embeddings[ids]

    def init_params(self):
        glorot(self.special_embedding)


class AggregatedEmbeddings(nn.Module):
    """
    Deliver the embeddings through previous ontology, contains categorized med, diagnosis embeddings.
    """

    def __init__(self, config, diag_voc, med_voc):
        super(AggregatedEmbeddings, self).__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ontology_embedding = CombinedEmbeddings(config, med_voc, diag_voc)
        # a type embedding module has 2 tensors
        self.type_embedding = nn.Embedding(num_embeddings = 2, embedding_dim = config.hidden_size)

    def forward(self, input_ids, input_types, input_positions=None):
        # print(len(input_ids[0]))
        # print(len(input_types[0]))
        # print(input_types)
        tmp = self.ontology_embedding(input_ids) + self.type_embedding(input_types)
        tmp = self.LayerNorm(tmp)
        tmp = self.dropout(tmp)
        return tmp