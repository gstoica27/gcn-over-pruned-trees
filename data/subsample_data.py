import os
import json
import numpy as np
import argparse


def compute_triple2data(data):
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


def subsample_triples(data, train_prop):
    triple2data = compute_triple2data(data)
    save_prop = train_prop
    keep_data = []
    for triple, triple_data in triple2data.items():
        save_amount = int(len(triple_data) * save_prop)
        if save_amount == 0:
            save_amount = len(triple_data)
        save_data = np.random.choice(triple_data, save_amount, replace=False).tolist()
        keep_data += save_data
    return keep_data


def save_data(data, file_path):
    with open(file_path, 'w') as handle:
        json.dump(data, handle)


def load_data(file_path):
    with open(file_path, 'rb') as handle:
        data = json.load(handle)
    return data

if __name__ == '__main__':

    cwd = os.getcwd()
    on_server = 'Desktop' not in cwd
    # Local paths
    if on_server:
        data_dir = '/usr0/home/gis/data/tacred/data/json'
    else:
        data_dir = '/Volumes/External HDD/dataset/tacred/data/json'
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=1.0)
    parser.add_argument('--data_dir', default=data_dir, type=str)
    parser.add_argument('--query_file', default='train.json', type=str)
    args = parser.parse_args()

    load_file = os.path.join(args.data_dir, args.query_file)
    data = load_data(file_path=load_file)
    new_data = subsample_triples(data=data, train_prop=args.train_prop)
    filename, extension = os.path.splitext(args.query_file)
    save_file = os.path.join(args.data_dir, filename + f'_{args.train_prop}{extension}')
    save_data(new_data, save_file)
