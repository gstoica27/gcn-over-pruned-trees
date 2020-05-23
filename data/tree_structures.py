import json
import os
import numpy as np
from collections import defaultdict, Counter
from data.loader import get_positions
from model.tree import head_to_tree


def load_data(filename):
    with open(filename) as handle:
        data = json.load(handle)
    return data

def extract_trees(data, prune_k):
    relation2trees = {}
    for d in data:
        tokens = list(d['token'])
        # anonymize tokens
        ss, se = d['subj_start'], d['subj_end']
        os, oe = d['obj_start'], d['obj_end']
        tokens[ss:se + 1] = ['SUBJ-' + d['subj_type']] * (se - ss + 1)
        tokens[os:oe + 1] = ['OBJ-' + d['obj_type']] * (oe - os + 1)
        head = [int(x) for x in d['stanford_head']]
        deprel = d['stanford_deprel']
        l = len(tokens)
        subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
        obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
        relation = d['relation']
        if relation not in relation2trees:
            relation2trees[relation] = []
        _, tree = head_to_tree(head=np.array(head), tokens=np.array(tokens), len_=l,
                            prune=prune_k, subj_pos=np.array(subj_positions),
                            obj_pos=np.array(obj_positions), deprel=np.array(deprel))
        relation2trees[relation].append(tree)
    return relation2trees

def get_deprel_statistics(trees):
    deprel2stats = defaultdict(lambda: defaultdict(lambda: 0))
    for tree in trees:
        deprel_dist = Counter([node.deprel for node in tree if node is not None])
        for deprel, count in deprel_dist.items():
            if 'freqs' not in deprel2stats[deprel]:
                deprel2stats[deprel]['freqs'] = []
            deprel2stats[deprel]['freqs'].append(count)
    for deprel, stats in deprel2stats.items():
        freqs = stats['freqs']
        stats['min'] = np.min(freqs)
        stats['max'] = np.max(freqs)
        stats['mean'] = np.mean(freqs)
        stats['median'] = np.median(freqs)
        stats['std'] = np.std(freqs)
        stats['count'] = len(freqs)
        # del stats['freqs']
    return deprel2stats


if __name__ == '__main__':
    data_dir = '/Volumes/External HDD/dataset/tacred/data/json'
    files = ['train', 'dev', 'test']
    partition2data = {}
    for name in files:
        partition2data[name] = load_data(os.path.join(data_dir, name + '.json'))
    partition2relation2trees = {}
    for name in files:
        partition2relation2trees[name] = extract_trees(partition2data[name],  prune_k=1)
    partition2relation2deprel2stats = defaultdict(lambda: defaultdict(dict))
    for name, relation2trees in partition2relation2trees.items():
        for relation, trees in relation2trees.items():
            partition2relation2deprel2stats[name][relation] = get_deprel_statistics(trees)

    print('done')

