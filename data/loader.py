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
        self.id2embeddings = bert_embeddings

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

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        # data = data[:100]
        for d in data:
            tokens = list(d['token'])
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)
            tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)
            if self.id2embeddings is not None:
                tokens = self.id2embeddings[d['id']]
            else:
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
        lens = [len(x) for x in batch[1]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval and self.id2embeddings is None:
            words = [word_dropout(sent, self.opt['word_dropout'], False) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        if self.id2embeddings is not None:
            words = self.pad_tokens(words)
            words = torch.from_numpy(words)
            masks = torch.eq(words.sum(-1), 0)
            token_len = None
        else:
            words = get_long_tensor(words, batch_size)
            masks = torch.eq(words, 0)
            token_len = None
        pos = get_long_tensor(batch[1], batch_size, token_len=token_len)
        ner = get_long_tensor(batch[2], batch_size, token_len=token_len)
        deprel = get_long_tensor(batch[3], batch_size, token_len=token_len)
        head = get_long_tensor(batch[4], batch_size, token_len=token_len)
        # dummy fill value larger than max sentence length (96). positions are
        # ONLY used to create the masks, so it does not matter what the fill
        # value is as long as it's not 0 (0 denotes subject/objects).
        num_actual_tokens = pos.shape[1]
        subj_positions = get_long_tensor(batch[5], batch_size, fill_value=150, token_len=token_len)[:, :num_actual_tokens]
        obj_positions = get_long_tensor(batch[6], batch_size, fill_value=150, token_len=token_len)[:, :num_actual_tokens]
        if self.opt['use_bert_embeddings']:
            words = words[:, :num_actual_tokens, :]
            masks = masks[:, :num_actual_tokens]

        if pos.shape[1] != words.shape[1]:
            print('Shapes: | Words: {} | POS: {} | NER: {} | HEAD: {} | DEPREL: {} | MASKS:{} | SUBJ-POS: {} | OBJ-POS: {} |'.format(
               words.shape, pos.shape, ner.shape, head.shape, deprel.shape, masks.shape, subj_positions.shape, obj_positions.shape
            ))
            word_embs = words.sum(-1)
            print(f'WORD EMBS: {word_embs[:, -1]}')
            print(f'SUJB-POS: {subj_positions[:, -1]}')
            print(f'OBJ-POS: {obj_positions[:, -1]}')

        rels = torch.LongTensor(batch[7])

        # print('words: {} | pos: {} | ner: {} | mask: {}'.format(words.shape, pos.shape, ner.shape, masks.shape))
            # exit()

        return (words, masks, pos, ner, deprel, head, subj_positions,
                obj_positions, rels, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def pad_tokens(self, embeddings):
        max_len = max([token_embs.shape[0] for token_embs in embeddings])
        embedding_size = embeddings[0].shape[-1]
        for idx, sample_embs in enumerate(embeddings):
            # print('max_len: {} | token lengths: {}'.format(max_len, embeddings[idx].shape[0]))
            pad_amount = max_len - sample_embs.shape[0]
            padding = np.zeros((pad_amount, embedding_size), dtype=np.float32)
            embeddings[idx] = np.concatenate([sample_embs, padding], axis=0)
        return np.array(embeddings)


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size, fill_value=constant.PAD_ID, token_len=None):
    """ Convert list of list of tokens to a padded LongTensor. """
    if token_len is None:
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

def word_dropout(tokens, dropout, use_bert):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    if not use_bert:
        return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]
    else:
        return [constant.UNK_TOKEN if x != constant.UNK_ID and np.random.random() < dropout \
                    else x for x in tokens]

