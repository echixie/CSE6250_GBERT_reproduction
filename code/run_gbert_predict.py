from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import dill
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from tensorboardX import SummaryWriter

from utils import metric_report, t2n, get_n_params
from config import BertConfig
from modeling_gbert_downstream import GbertRxPredict
from data_utils import MedicalPredictTokenizer
import data_utils
from constants import MODEL_NAME, MODEL_DIR_NAME

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictDataset(Dataset):
    def __init__(self, data_pd, tokenizer: MedicalPredictTokenizer, max_seq_len):
        self.data_pd = data_pd
        self.tokenizer = tokenizer
        self.seq_len = max_seq_len

        self.sample_counter = 0

        self.records = self.transform_predict_data(data_pd)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        subject_id = list(self.records.keys())[item]


        """extract input and output tokens
        """
        input_tokens = []  # (2*max_len*adm)
        output_dx_tokens = []  # (adm-1, l)
        output_rx_tokens = []  # (adm-1, l)

        for idx, adm in enumerate(self.records[subject_id]):
            input_tokens.extend(
                ['[CLS]'] + self.fill_to_max(list(adm[0]), self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + self.fill_to_max(list(adm[1]), self.seq_len - 1))

            if idx != 0:
                output_dx_tokens.append(list(adm[0]))
                output_rx_tokens.append(list(adm[1]))

        """convert tokens to id
        """
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        output_dx_labels = self.token_to_onehotlabel(output_dx_tokens, self.tokenizer.dx_voc_multi)
        output_rx_labels = self.token_to_onehotlabel(output_rx_tokens, self.tokenizer.rx_voc_multi)

        if cur_id < 5:
            logger.info("*** Example ***")
            logger.info("subject_id: %s" % subject_id)
            logger.info("input tokens: %s" % " ".join(
                [str(x) for x in input_tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))

        assert len(input_ids) == (self.seq_len *
                                  2 * len(self.records[subject_id]))
        assert len(output_dx_labels) == (len(self.records[subject_id]) - 1)
        # assert len(output_rx_labels) == len(self.records[subject_id])-1

        cur_tensors = (torch.tensor(input_ids).view(-1, self.seq_len),
                       torch.tensor(output_dx_labels, dtype=torch.float),
                       torch.tensor(output_rx_labels, dtype=torch.float))

        return cur_tensors

    def transform_predict_data(self, data):
        records = {}
        for subject_id in data['SUBJECT_ID'].unique():
            item_df = data[data['SUBJECT_ID'] == subject_id]
            patient = []
            for _, row in item_df.iterrows():
                admission = [list(row['ICD9_CODE']), list(row['ATC4'])]
                patient.append(admission)
            if len(patient) < 2:
                continue
            records[subject_id] = patient
        return records

    def token_to_onehotlabel(self, tokens, voc):
        labels = []
        for token in tokens:
            tmp = np.zeros(len(voc.word2idx))
            tmp[list(map(lambda x: voc.word2idx[x], token))] = 1
            labels.append(tmp)
        return labels


    def fill_to_max(self, l, seq):
        while len(l) < seq:
            l.append('[PAD]')
        return l


def load_predict_dataset(data_dir, max_seq_len):

    # load tokenizer
    tokenizer = MedicalPredictTokenizer(data_dir)

    # load data
    multi_train, multi_eval, multi_test = data_utils.load_multi_visit_data(data_dir)

    return tokenizer, \
           PredictDataset(multi_train, tokenizer, max_seq_len), \
           PredictDataset(multi_eval, tokenizer, max_seq_len), \
           PredictDataset(multi_test, tokenizer, max_seq_len)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default='../data',
                        type=str,
                        required=False,
                        help="The input data dir.")
    parser.add_argument("--pretrain_dir", default='../saved/GBert-pretraining', type=str, required=False,
                        help="pretraining model")
    parser.add_argument("--output_dir",
                        default='../saved/',
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--use_pretrain",
                        default=False,
                        action='store_true',
                        help="is use pretrain")
    parser.add_argument("--graph",
                        default=False,
                        action='store_true',
                        help="if use ontology embedding")
    parser.add_argument("--therhold",
                        default=0.3,
                        type=float,
                        help="therhold.")
    parser.add_argument("--max_seq_length",
                        default=55,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="Whether to run on the dev set.")
    parser.add_argument("--do_test",
                        default=True,
                        action='store_true',
                        help="Whether to run on the test set.")
    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=20.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=1203,
                        help="random seed for initialization")


    # save dir as "../saved/GBert-predict"
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, MODEL_DIR_NAME)

    # random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    print("--- Loading Dataset ---")
    tokenizer, train_set, eval_set, test_set = load_predict_dataset(args.data_dir, args.max_seq_length)
    train_dataset = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=1)
    eval_dataset = DataLoader(eval_set, sampler=SequentialSampler(eval_set), batch_size=1)
    test_dataset = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=1)

    print("--- Loading Model and Config ---")
    if args.use_pretrain:
        logger.info("Use Pretraining model")
        model = GbertRxPredict.from_pretrained(args.pretrain_dir, tokenizer=tokenizer)
    else:
        config = BertConfig(vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx))
        config.graph = args.graph
        model = GbertRxPredict(config, tokenizer)

    model.to(device)

    saved_model = model.module if hasattr(model, 'module') else model 
    saved_model_file = os.path.join(args.output_dir, MODEL_NAME)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    global_step = 0
    if args.do_train:
        writer = SummaryWriter(args.output_dir)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_set))
        logger.info("  Batch size = %d", 1)

        rx_acc_best = 0
        acc_name = 'prauc'
        dx_history = {'prauc': []}
        rx_history = {'prauc': []}

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            print('')
            step_loss, steps = 0, 0
            train_iterator = tqdm(train_dataset, leave=False, desc='Training')
            model.train()
            for _, batch in enumerate(train_iterator):
                batch = tuple(t.to(device) for t in batch)
                input_ids, dx_labels, rx_labels = batch
                input_ids, dx_labels, rx_labels = input_ids.squeeze(dim=0), dx_labels.squeeze(dim=0), rx_labels.squeeze(dim=0)
                loss, rx_logits = model(input_ids, dx_labels=dx_labels, rx_labels=rx_labels,epoch=global_step)
                loss.backward()

                # increment progress
                step_loss += loss.item()
                steps += 1

                # Display loss
                train_iterator.set_postfix(loss='%.4f' % (step_loss / steps))

                optimizer.step()
                optimizer.zero_grad()

            writer.add_scalar('train/loss', step_loss / steps, global_step)
            global_step += 1

            if args.do_eval:
                print('')
                logger.info("***** Running eval *****")
                model.eval()
                dx_y_preds, dx_y_trues, rx_y_preds, rx_y_trues = [], [], [], []
                for eval_input in tqdm(eval_dataset, desc="Evaluating"):
                    eval_input = tuple(t.to(device) for t in eval_input)
                    input_ids, dx_labels, rx_labels = eval_input
                    input_ids, dx_labels, rx_labels = input_ids.squeeze(), dx_labels.squeeze(), rx_labels.squeeze(dim=0)
                    with torch.no_grad():
                        loss, rx_logits = model(input_ids, dx_labels=dx_labels, rx_labels=rx_labels)
                        rx_y_preds.append(t2n(torch.sigmoid(rx_logits)))
                        rx_y_trues.append(t2n(rx_labels))

                rx_acc_container = metric_report(np.concatenate(rx_y_preds, axis=0), np.concatenate(rx_y_trues, axis=0), args.therhold)
                for k, v in rx_acc_container.items():
                    writer.add_scalar(
                        'eval/{}'.format(k), v, global_step)

                if rx_acc_container[acc_name] > rx_acc_best:
                    rx_acc_best = rx_acc_container[acc_name]
                    # save model
                    torch.save(saved_model.state_dict(),
                               saved_model_file)
        # save config
        with open(os.path.join(args.output_dir, 'bert_config.json'), 'w', encoding='utf-8') as file:
            file.write(model.config.to_json_string())

    if args.do_test:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", len(test_set))
        logger.info("  Batch size = %d", 1)

        def test(task=0):
            model_state_dict = torch.load(saved_model_file)
            model.load_state_dict(model_state_dict)
            model.to(device)

            model.eval()
            y_preds, y_trues = [], []
            for test_input in tqdm(test_dataset, desc="Testing"):
                test_input = tuple(t.to(device) for t in test_input)
                input_ids, dx_labels, rx_labels = test_input
                input_ids, dx_labels, rx_labels = input_ids.squeeze(), dx_labels.squeeze(), rx_labels.squeeze(dim=0)
                with torch.no_grad():
                    loss, rx_logits = model(input_ids, dx_labels=dx_labels, rx_labels=rx_labels)
                    y_preds.append(t2n(torch.sigmoid(rx_logits)))
                    y_trues.append(t2n(rx_labels))

            print('')
            acc_container = metric_report(np.concatenate(y_preds, axis=0), np.concatenate(y_trues, axis=0),
                                          args.therhold)

            # save report
            if args.do_train:
                for k, v in acc_container.items():
                    writer.add_scalar(
                        'test/{}'.format(k), v, 0)

            return acc_container

        test(task=0)


if __name__ == "__main__":
    main()
