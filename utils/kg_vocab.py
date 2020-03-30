from __future__ import print_function
import os
import pickle


class KGVocab(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.ent2id = None
        self.rel2id = None
        self.load_mappings()
        self.reverse_mappings()

    def load_mappings(self):
        ent2id_file = os.path.join(self.data_dir, 'ent2id.pkl')
        rel2id_file = os.path.join(self.data_dir, 'rel2id.pkl')
        self.ent2id = pickle.load(open(ent2id_file, 'rb'))
        self.rel2id = pickle.load(open(rel2id_file, 'rb'))

    def reverse_mappings(self):
        self.id2rel = [''] * len(self.rel2id)
        self.id2ent = [''] * len(self.ent2id)
        for key, value in self.ent2id.items():
            self.id2ent[value] = key
        for key, value in self.rel2id.items():
            self.id2rel[value] = key

    def load_graph(self, partition_name):
        # Return the graph along a partition
        graph_file = os.path.join(self.data_dir, f'{partition_name}_graph.pkl')
        return pickle.load(open(graph_file, 'rb'))

    def return_ent2id(self):
        return self.ent2id

    def return_rel2id(self):
        return self.rel2id

    def return_id2ent(self):
        return self.id2ent

    def return_id2rel(self):
        return self.id2rel

    def return_num_ent(self):
        return len(self.ent2id)

    def return_num_rel(self):
        return len(self.rel2id)