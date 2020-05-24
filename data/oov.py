import os
import json
import numpy as np

VERB_TAGS = {
    'VBG',
    'MD',
    'VBN',
    'VBD',
    'VBP',
    'VBZ',
    'VB',
}

def load_data(filename):
    with open(filename, 'r') as handle:
        return json.load(handle)

def extract_verb_vocab(data):
    vocab = set()
    for d in data:
        tokens = d['token']
        poses = d['stanford_pos']
        for token, pos in zip(tokens, poses):
            if pos in VERB_TAGS:
                vocab.add(token)
    return vocab

if __name__ == '__main__':
    data_dir = '/Volumes/External HDD/dataset/tacred/data/json'
    train_file = 'train.json'
    test_file = 'test_incorrect.json'
    train_path = os.path.join(data_dir, train_file)
    test_path = os.path.join(data_dir, test_file)

    train_data = load_data(train_path)
    test_data = load_data(test_path)
    train_verbs = extract_verb_vocab(train_data)
    test_verbs = extract_verb_vocab(test_data)
    missing_verbs = test_verbs - train_verbs
    print(f'Num Verbs | Train: {len(train_verbs)} | Test: {len(test_verbs)}')
    print(f'Missing Verbs: {len(missing_verbs)}')