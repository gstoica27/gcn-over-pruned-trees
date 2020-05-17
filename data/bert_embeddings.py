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


def replace_tokens(tokens):
    mappings = {
        '-LRB-': '(',
        '-RRB-': ')',
        '-LSB-': '[',
        '-RSB-': ']',
        '-LCB-': '{',
        '-RCB-': '}'
    }
    return [token if token not in mappings else mappings[token] for token in tokens]


def extract_ids_and_sentences(data):
    id2sentences = {}
    for d in data:
        tokens = list(d['token'])
        tokens = replace_tokens(tokens)
        # anonymize tokens
        ss, se = d['subj_start'], d['subj_end']
        os, oe = d['obj_start'], d['obj_end']
        tokens[ss:se + 1] = ['SUBJ-' + d['subj_type']] * (se - ss + 1)
        tokens[os:oe + 1] = ['OBJ-' + d['obj_type']] * (oe - os + 1)
        token_id = d['id']
        id2sentences[token_id] = tokens

    id2sentences_ascending = sorted(id2sentences.items(), key=lambda kv: len(kv[1]))
    # ids, sentences = zip(*id2sentences_ascending)
    return id2sentences_ascending


def batch_data(data, batch_size=100):
    batched_data = []
    for batch_start in range(0, len(data), batch_size):
        batch = data[batch_start: batch_start + batch_size]
        batched_data.append(batch)
    return batched_data


def extract_embeddings(batches):
    print('Extracting Embeddings... | Batch len: {}'.format(len(batches)))

    bert_client = BertClient()
    id2sentence_embeddings = {}
    for batch in batches:
        ids, sentences = zip(*batch)
        sentences = list(sentences)
        batch_embeddings = bert_client.encode(sentences, is_tokenized=True)[:, 1:-1, :]
        for sample_idx, sample_id in enumerate(ids):
            sample_embeddings = batch_embeddings[sample_idx]
            id2sentence_embeddings[sample_id] = sample_embeddings
    return id2sentence_embeddings


def save_data(data, save_path):
    with open(save_path, 'wb') as handle:
        pickle.dump(data, handle)


file_dir = '/usr0/home/gis/data/tacred/data/json'
# file_dir = '/Volumes/External HDD/dataset/tacred/data/json'
file_names = ['train.json', 'dev.json', 'test.json']

data = load_datasets(file_dir, file_names)
id2sentences = extract_ids_and_sentences(data)
batched_data = batch_data(id2sentences)
id2embeddings = extract_embeddings(batched_data)

bert_save_dir = '/usr0/home/gis/data/bert_saves'
os.makedirs(bert_save_dir, exist_ok=True)
bert_save_path = os.path.join(bert_save_dir, 'id2embeddings.pkl')
save_data(id2embeddings, bert_save_path)
