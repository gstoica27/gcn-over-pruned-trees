import json
import os
import stanfordcorenlp as corenlp
import stanfordnlp as nlp
from collections import defaultdict
import re


train_file = os.path.join('/Users/georgestoica/Desktop/Research', 'SemEval2010_task8_all_data', 'SemEval2010_task8_training', 'TRAIN_FILE.txt')

def parse_txt_file(txt_file):
    parsed_data = defaultdict(lambda: {'id': None, 'tokens': [], 'subj_start': None, 'subj_end': None, 'obj_start': None, 'obj_end': None, 'relation': None})

    with open(txt_file, 'r') as handle:
        all_data = handle.readlines()

        for i in range(0, len(all_data), 4):
            sentence_line = all_data[i].strip().split('\t')
            relation_line = all_data[i+1].strip().split('(')
            id, sentence = sentence_line
            parsed_data[id]['id'] = id

            first_entity = re.search('<e1>(.+?)</e1>', sentence).group(1)
            second_entity = re.search('<e2>(.+?)</e2>', sentence).group(1)
            relation = relation_line[0]
            entity_order = relation_line[1][:-1].split(',')
            if entity_order[0] == 'e1':

