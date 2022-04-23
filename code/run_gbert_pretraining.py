from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.optim import Adam
from tensorboardX import SummaryWriter

from utils import metric_report, t2n, get_n_params
from config import BertConfig
from modeling_gbert_pretrain import GbertPretrain
from data_utils import MedicalTokenizer
import data_utils
from constants import MODEL_NAME, MODEL_DIR_NAME

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class PretrainDataset(Dataset):
    def __init__(self, data_pd, tokenizer: MedicalTokenizer, max_seq_len):
        self.data_pd = data_pd
        self.tokenizer = tokenizer
        self.seq_len = max_seq_len
        self.admissions = self.transform_pretain_data(data_pd)

    def __len__(self):
        return len(self.admissions)

    def __getitem__(self, item):
        cur_id = item
        adm = copy.deepcopy(self.admissions[item])

        y_dx = np.zeros(len(self.tokenizer.dx_voc.word2idx))
        y_rx = np.zeros(len(self.tokenizer.rx_voc.word2idx))
        for item in adm[0]:
            y_dx[self.tokenizer.dx_voc.word2idx[item]] = 1
        for item in adm[1]:
            y_rx[self.tokenizer.rx_voc.word2idx[item]] = 1

        adm[0] = self.random_word(adm[0], self.tokenizer.dx_voc)
        adm[1] = self.random_word(adm[1], self.tokenizer.rx_voc)

        adm[0] = self.fill_to_max(list(adm[0]), self.seq_len - 1)
        adm[1] = self.fill_to_max(list(adm[1]), self.seq_len - 1)

        input_tokens = []  # (2*max_len)
        input_tokens.extend(['[CLS]'] + adm[0])
        input_tokens.extend(['[CLS]'] + adm[1])


        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        if cur_id < 5:
            logger.info("*** Example ***")
            logger.info("input tokens: %s" % " ".join(
                [str(x) for x in input_tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))

        cur_tensors = (torch.tensor(input_ids, dtype=torch.long).view(-1, self.seq_len),
                       torch.tensor(y_dx, dtype=torch.float),
                       torch.tensor(y_rx, dtype=torch.float))

        return cur_tensors

    def random_word(self, tokens, vocab):
        for i, _ in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = "[MASK]"
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.choice(list(vocab.word2idx.items()))[0]
                else:
                    pass
            else:
                pass

        return tokens

    def transform_pretain_data(self, data):
        admissions = []
        for _, row in data.iterrows():
            admission = [list(row['ICD9_CODE']), list(row['ATC4'])]
            admissions.append(admission)
        return admissions

    def fill_to_max(self, l, seq):
        while len(l) < seq:
            l.append('[PAD]')
        return l


def load_pretain_dataset(data_dir, max_seq_len):

    # load tokenizer
    tokenizer = MedicalTokenizer(data_dir)

    # load data
    multi_train, multi_eval, multi_test = data_utils.load_multi_visit_data(data_dir)
    single_train = data_utils.load_single_visit_data(data_dir)
    return tokenizer, \
        PretrainDataset(pd.concat([single_train, multi_train]), tokenizer, max_seq_len), \
        PretrainDataset(multi_eval, tokenizer, max_seq_len), \
        PretrainDataset(multi_test, tokenizer, max_seq_len)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default='../data',
                        type=str,
                        required=False,
                        help="The input data dir.")
    parser.add_argument("--output_dir",
                        default='../saved/',
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--use_pretrain",
                        default=False,
                        action='store_true',
                        help="if use ontology embedding")
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
    parser.add_argument("--batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10.0,
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

    # create save dir
    os.makedirs(args.output_dir, exist_ok=True)

    print("--- Loading Dataset ---")
    tokenizer, train_set, eval_set, test_set = load_pretain_dataset(args.data_dir, args.max_seq_length)
    train_dataset = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=args.batch_size)
    eval_dataset = DataLoader(eval_set, sampler=SequentialSampler(eval_set), batch_size=args.batch_size)
    test_dataset = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=args.batch_size)

    print("--- Loading Model and Config ---")
    config = BertConfig(vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx))
    config.graph = args.graph
    model = GbertPretrain(config, tokenizer.dx_voc, tokenizer.rx_voc)
    model.to(device)
    saved_model = model.module if hasattr(model, 'module') else model 
    saved_model_file = os.path.join(args.output_dir, MODEL_NAME)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    if args.do_train:
        writer = SummaryWriter(args.output_dir)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_set))
        logger.info("  Batch size = %d", 1)

        dx_acc_best = 0
        acc_name = 'prauc'

        global_step = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            step_loss, steps = 0, 0
            train_iterator = tqdm(train_dataset, leave=False, desc='Training')
            model.train()
            for _, batch in enumerate(train_iterator):
                batch = tuple(t.to(device) for t in batch)
                input_ids, dx_labels, rx_labels = batch
                # step train res
                loss, dx2dx, rx2dx, dx2rx, rx2rx = model(
                    input_ids, dx_labels, rx_labels)

                loss.backward()

                step_loss += loss.item()
                steps += 1
                train_iterator.set_postfix(loss='%.4f' % (step_loss / steps))
                optimizer.step()
                optimizer.zero_grad()

            writer.add_scalar('train/loss', step_loss / steps, global_step)
            global_step += 1

            if args.do_eval:
                logger.info("***** Running eval *****")
                model.eval()
                dx2dx_y_preds, rx2dx_y_preds, dx_y_trues = [], [], []
                dx2rx_y_preds, rx2rx_y_preds, rx_y_trues = [], [], []
                for batch in tqdm(eval_dataset, desc="Evaluating"):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, dx_labels, rx_labels = batch
                    with torch.no_grad():
                        dx2dx, rx2dx, dx2rx, rx2rx = model(input_ids)
                        dx2dx_y_preds.append(t2n(dx2dx))
                        rx2dx_y_preds.append(t2n(rx2dx))
                        dx2rx_y_preds.append(t2n(dx2rx))
                        rx2rx_y_preds.append(t2n(rx2rx))

                        dx_y_trues.append(
                            t2n(dx_labels))
                        rx_y_trues.append(
                            t2n(rx_labels))

                print('')
                print('dx2dx')
                dx2dx_acc_container = metric_report(
                    np.concatenate(dx2dx_y_preds, axis=0), np.concatenate(dx_y_trues, axis=0), args.therhold)
                print('rx2dx')
                rx2dx_acc_container = metric_report(
                    np.concatenate(rx2dx_y_preds, axis=0), np.concatenate(dx_y_trues, axis=0), args.therhold)
                print('dx2rx')
                dx2rx_acc_container = metric_report(
                    np.concatenate(dx2rx_y_preds, axis=0), np.concatenate(rx_y_trues, axis=0), args.therhold)
                print('rx2rx')
                rx2rx_acc_container = metric_report(
                    np.concatenate(rx2rx_y_preds, axis=0), np.concatenate(rx_y_trues, axis=0), args.therhold)

                # keep in history
                for k, v in dx2dx_acc_container.items():
                    writer.add_scalar(
                        'eval_dx2dx/{}'.format(k), v, global_step)
                for k, v in rx2dx_acc_container.items():
                    writer.add_scalar(
                        'eval_rx2dx/{}'.format(k), v, global_step)
                for k, v in dx2rx_acc_container.items():
                    writer.add_scalar(
                        'eval_dx2rx/{}'.format(k), v, global_step)
                for k, v in rx2rx_acc_container.items():
                    writer.add_scalar(
                        'eval_rx2rx/{}'.format(k), v, global_step)

                if rx2rx_acc_container[acc_name] > dx_acc_best:
                    dx_acc_best = rx2rx_acc_container[acc_name]
                    # save model
                    torch.save(saved_model.state_dict(),
                               saved_model_file)

                    with open(os.path.join(args.output_dir, 'bert_config.json'), 'w', encoding='utf-8') as file:
                        file.write(model.config.to_json_string())


if __name__ == "__main__":
    main()
