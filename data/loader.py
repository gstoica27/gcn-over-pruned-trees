"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
import os
from collections import defaultdict
from utils import constant, helper, vocab

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False, kg_vocab=None):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID
        self.kg_vocab = kg_vocab
        if self.kg_vocab is not None:
            # Extract file name without path or extension
            self.partition_name = os.path.splitext(os.path.basename(filename))[0]
            # load partition KG
            self.create_kg()

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data
        data = self.preprocess(data, vocab, opt)

        # shuffle for training
        if not evaluation:
            data = self.shuffle_data(data)
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-1]] for d in data['base']]
        self.num_examples = len(data['base'])

        # chunk into batches
        data = self.create_batches(data=data, batch_size=batch_size)
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def create_batches(self, data, batch_size):
        batched_data = []
        for batch_start in range(0, len(data['base']), batch_size):

            batch_end = batch_start + batch_size
            base_batch = data['base'][batch_start: batch_end]
            data_batch = {'base': base_batch, 'supplemental': dict()}
            supplemental_batch = data_batch['supplemental']
            for component in data['supplemental']:
                supplemental_batch[component] = data['supplemental'][component][batch_start: batch_end]
            batched_data.append(data_batch)
        return batched_data

    def shuffle_data(self, data):
        indices = list(range(len(data['base'])))
        random.shuffle(indices)
        shuffled_base = data['base'][indices]
        supplemental_data = data['supplemental']
        for name, component in supplemental_data.items():
            supplemental_data[name] = component[indices]
        shuffled_data = {'base': shuffled_base, 'supplemental': supplemental_data}
        return shuffled_data

    def create_kg(self):
        self.kg = self.kg_vocab.load_graph(partition_name=self.partition_name)

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        base_processed = []
        supplemental_components = defaultdict(list)
        if self.kg_vocab is not None:
            graph = defaultdict(lambda: set())
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
            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            relation = self.label2id[d['relation']]
            base_processed += [(tokens, pos, ner, deprel, head, subj_positions,
                                obj_positions, subj_type, obj_type, relation)]
            # Use KG component
            if self.kg_vocab is not None:
                subject_type = 'SUBJ-' + d['subj_type']
                object_type = 'OBJ-' + d['obj_type']
                # Subtract offsets where needed. The way this works is that to find the corresponding subject or
                # object embedding from the tokens, an embedding lookup is performed on the pretrained word2vec
                # embedding matrix. The lookup only involves the subject, so the corresponding mapping utilizes
                # the original subject token position in the vocab. However, the object ids will yield binary
                # labels indicating whether a respective object is a valid answer to the (subj, rel) pair. Thus,
                # We offset the object id so that it results in a zero-indexed binary labeling downstream. Note,
                # the offset is 4 because the vocab order is: ['PAD', 'UNK', 'SUBJ-_', 'SUBJ-_', 'OBJ-*']. So
                # objects are at index 4 onwards.
                subject_id = vocab.word2id[subject_type]
                object_id = vocab.word2id[object_type] - 4
                graph[(subject_id, relation)].add(object_id)
                supplemental_components['knowledge_graph'] += [(subject_id, relation, object_id)]
                # Extract all known answers for subject type, relation pair in KG
                # supplemental_components['knowledge_graph'] += [(subject_id, known_object_types)]
        if self.kg_vocab is not None:
            component_data = supplemental_components['knowledge_graph']
            for idx in range(len(component_data)):
                instance_subj, instance_rel, instance_obj = component_data[idx]
                known_objects = graph[(instance_subj, instance_rel)]
                component_data[idx] = (instance_subj, instance_rel, known_objects)

            # transform to arrays for easier manipulations
        for name in supplemental_components.keys():
            supplemental_components[name] = np.array(supplemental_components[name])
        return {'base': np.array(base_processed), 'supplemental': supplemental_components}

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
        batch = self.ready_data_batch(batch)
        return batch

    def ready_base_batch(self, base_batch, batch_size):
        batch = list(zip(*base_batch))
        assert len(batch) == 10

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
        subj_positions = get_long_tensor(batch[5], batch_size)
        obj_positions = get_long_tensor(batch[6], batch_size)
        subj_type = get_long_tensor(batch[7], batch_size)
        obj_type = get_long_tensor(batch[8], batch_size)

        rels = torch.LongTensor(batch[9])

        merged_components = (words, masks, pos, ner, deprel, head, subj_positions,
                            obj_positions, subj_type, obj_type, rels, orig_idx)

        return {'base': merged_components, 'sentence_lengths': lens}

    def ready_masks_batch(self, masks_batch, batch_size, sentence_lengths):
        batch = list(zip(*masks_batch))
        # sort all fields by lens for easy RNN operations
        batch, _ = sort_all(batch, sentence_lengths)

        subj_masks = get_long_tensor(batch[0], batch_size)
        obj_masks = get_long_tensor(batch[1], batch_size)
        merged_components = (subj_masks, obj_masks)
        return merged_components

    def ready_knowledge_graph_batch(self, kg_batch, sentence_lengths):
        # Offset because we don't include the 2 subject entities
        num_ent = self.kg_vocab.return_num_ent() - 2
        batch = list(zip(*kg_batch))
        batch, _ = sort_all(batch, sentence_lengths)
        subjects, relations, known_objects = batch
        subjects = torch.LongTensor(subjects)
        relations = torch.LongTensor(relations)
        labels = []
        for sample_labels in known_objects:
            binary_labels = np.zeros(num_ent, dtype=np.float32)
            binary_labels[list(sample_labels)] = 1.
            labels.append(binary_labels)
        labels = np.stack(labels, axis=0)
        labels = torch.FloatTensor(labels)
        merged_components = (subjects, relations, labels)
        return merged_components

    def ready_data_batch(self, batch):
        batch_size = len(batch['base'])
        readied_batch = self.ready_base_batch(batch['base'], batch_size)
        readied_batch['supplemental'] = dict()
        readied_supplemental = readied_batch['supplemental']
        for name, supplemental_batch in batch['supplemental'].items():
            if name == 'entity_masks':
                readied_supplemental[name] = self.ready_masks_batch(
                    masks_batch=supplemental_batch,
                    batch_size=batch_size,
                    sentence_lengths=readied_batch['sentence_lengths'])
            elif name == 'knowledge_graph':
                readied_supplemental[name] = self.ready_knowledge_graph_batch(
                    kg_batch=supplemental_batch,
                    sentence_lengths=readied_batch['sentence_lengths'])
        return readied_batch

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

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
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

