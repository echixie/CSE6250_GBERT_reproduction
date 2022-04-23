import os
import pandas as pd

RX_VOC_FILE = 'rx-vocab.txt'
DX_VOC_FILE = 'dx-vocab.txt'
RX_MULTI_VOC_FILE = 'rx-vocab-multi.txt'
DX_MULTI_VOC_FILE = 'dx-vocab-multi.txt'

MULTI_VISIT_FILE = 'data-multi-visit.pkl'
SINGLE_VISIT_FILE ='data-single-visit.pkl'
TRAIN_ID_FILE = 'train-id.txt'
EVAL_ID_FILE= 'eval-id.txt'
TEST_ID_FILE = 'test-id.txt'

class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_from_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)

    def add_from_file(self, vocab_file):
        with open(vocab_file, 'r') as fin:
            for code in fin:
                self.add_from_sentence([code.rstrip('\n')])


class MedicalTokenizer(object):

    def __init__(self, data_dir, special_tokens=("[PAD]", "[MASK]", "[CLS]")):

        self.vocab = Voc()
        self.rx_voc = Voc()
        self.dx_voc = Voc()
        # special tokens
        self.vocab.add_from_sentence(special_tokens)

        self.vocab.add_from_file(os.path.join(data_dir, RX_VOC_FILE))
        self.vocab.add_from_file(os.path.join(data_dir, DX_VOC_FILE))

        self.rx_voc.add_from_file(os.path.join(data_dir, RX_VOC_FILE))
        self.dx_voc.add_from_file(os.path.join(data_dir, DX_VOC_FILE))

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.vocab.word2idx[token])
        return ids

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.vocab.idx2word[i])
        return tokens


class MedicalPredictTokenizer(MedicalTokenizer):
    def __init__(self, data_dir, special_tokens=("[PAD]", "[MASK]", "[CLS]")):
        super(MedicalPredictTokenizer, self).__init__(data_dir,special_tokens)

        self.rx_voc_multi = Voc()
        self.dx_voc_multi = Voc()
        self.rx_voc_multi.add_from_file(os.path.join(data_dir, RX_MULTI_VOC_FILE))
        self.dx_voc_multi.add_from_file(os.path.join(data_dir, DX_MULTI_VOC_FILE))



def load_multi_visit_data(data_dir):
    data_multi = pd.read_pickle(os.path.join(data_dir, MULTI_VISIT_FILE)).iloc[:, :4]
    # load trian, eval, test data
    ids_file = [os.path.join(data_dir, TRAIN_ID_FILE),
                os.path.join(data_dir, EVAL_ID_FILE),
                os.path.join(data_dir, TEST_ID_FILE)]

    result = []
    for file in ids_file:
        ids = []
        with open(file, 'r') as f:
            for line in f:
                ids.append(int(line.rstrip('\n')))
        result.append(data_multi[data_multi['SUBJECT_ID'].isin(ids)].reset_index(drop=True))
    return result[0], result[1], result[2]

def load_single_visit_data(data_dir):
    data_single = pd.read_pickle(os.path.join(data_dir, SINGLE_VISIT_FILE))
    return data_single