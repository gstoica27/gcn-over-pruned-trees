"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np

from utils import constant, helper, vocab
import pickle

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False, bert_embeddings=None):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID
        if bert_embeddings is not None:
            self.id2bert_embs = pickle.load(open(bert_embeddings, 'rb'))

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data
        data = self.preprocess(data, vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-1]] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def compute_triple2data(self, data):
        triple2data = {}
        for d in data:
            subject = d['subj_type']
            object = d['obj_type']
            relation = d['relation']
            triple = (subject, relation, object)
            if triple not in triple2data:
                triple2data[triple] = []
            triple2data[triple].append(d)
        return triple2data

    def subsample_triples(self, data):
        triple2data = self.compute_triple2data(data)
        save_prop = self.opt['train_prop']
        keep_data = []
        for triple, triple_data in triple2data.items():
            save_amount = int(len(triple_data) * save_prop)
            if save_amount == 0:
                save_amount = len(triple_data)
            save_data = np.random.choice(triple_data, save_amount, replace=False).tolist()
            keep_data += save_data
        return keep_data

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        # data = data[:100]
        if opt['train_prop'] < 1.0:
            data = self.subsample_triples(data)
            np.random.shuffle(data)
        for d in data:
            tokens = list(d['token'])
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)
            tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)
            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            relation = self.label2id[d['relation']]
            processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, relation)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 8

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        head = get_long_tensor(batch[4], batch_size)
        # dummy fill value larger than max sentence length (96). positions are
        # ONLY used to create the masks, so it does not matter what the fill
        # value is as long as it's not 0 (0 denotes subject/objects).
        subj_positions = get_long_tensor(batch[5], batch_size, fill_value=150)
        obj_positions = get_long_tensor(batch[6], batch_size, fill_value=150)

        rels = torch.LongTensor(batch[7])

        return (words, masks, pos, ner, deprel, head, subj_positions, obj_positions, rels, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size, fill_value=constant.PAD_ID):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(fill_value)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

