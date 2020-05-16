import os
import json
from bert_serving.client import BertClient
import pickle
from time import time


def load_datasets(file_dir, file_names):
    data = []
    for file_name in file_names:
        file_path = os.path.join(file_dir, file_name)
        with open(file_path, 'rb') as handle:
            partition_data = json.load(handle)
        data += partition_data
    return data

def replace_tokens(data):
    mappings = {
            '-LRB-': '(',
            '-RRB-': ')',
            '-LSB-': '[',
            '-RSB-': ']',
            '-LCB-': '{',
            '-RCB-': '}'
        }
    for d in data:
        d['token'] = [token if token not in mappings else mappings[token] for token in d['token']]
    return data

def save_data(data, save_path):
    with open(save_path, 'wb') as handle:
        pickle.dump(data, handle)

def extract_embeddings(data, save_dir):
    bc = BertClient()
    vocab2id = {}
    embeddings = []
    start_time = time()
    gap = int(len(data) / 100)
    percent_done = -1
    for idx, d in enumerate(data):
        tokens = d['token']
        token_embeddings = bc.encode(tokens)
        for idx, token in enumerate(tokens):
            if token not in vocab2id:
                vocab2id[token] = len(vocab2id)
                token_embedding = token_embeddings[idx]
                embeddings.append(token_embedding)
        if idx % gap == 0:
            percent_done += 1
            gap_time = time()
            print(f'Completed {percent_done * 100}%. Time elapsed: {gap_time - start_time}')


    vocab_path = os.path.join(save_dir, 'vocab.pkl')
    embeddings_path = os.path.join(save_dir, 'embeddings.pkl')
    save_data(vocab2id, vocab_path)
    save_data(embeddings, embeddings_path)

file_dir = '/usr0/home/gis/data/tacred/data/json'
file_names = ['train.json', 'dev.json', 'test.json']
data = load_datasets(file_dir, file_names)
data = replace_tokens(data)
bert_save_dir = '/usr0/home/gis/data/bert_saves'
os.makedirs(bert_save_dir, exist_ok=True)
extract_embeddings(data, bert_save_dir)
