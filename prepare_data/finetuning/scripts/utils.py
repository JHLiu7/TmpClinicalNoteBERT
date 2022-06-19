import pandas as pd 
import json, os
from collections import OrderedDict


######## radgraph parsing 


def _parse_ent(ent_dict):
    tokens = ent_dict['tokens']
    start_ix = ent_dict['start_ix']
    end_ix = ent_dict['end_ix']
    label = ent_dict['label']
    relations = ent_dict['relations'] # list
    
    return tokens, (start_ix, end_ix), label, relations

def prepare_one_case(doc, entities_dict, doc_key=None, dataname=None):
    """ 
    This func treats doc as a single sentence.
    OUtput dygiepp format.
    """
    all_ner = []
    all_rel = []

    count_ner = len(entities_dict)
    count_rel = 0
    for ent_dict in entities_dict.values():
        tokens, (start_ix, end_ix), label, relations = _parse_ent(ent_dict)

        # entities
        if start_ix == end_ix:
            doc_token = doc[start_ix]

        else:
            doc_token = ' '.join(doc[start_ix: end_ix+1])
            # print(start_ix, end_ix, tokens)

        assert doc_token == tokens, f'{doc_key}'
        all_ner.append([start_ix, end_ix, label])

        # relations
        if len(relations) > 0:
            for relation in relations:
                rel_label, ent_ix = relation

                _, (start_ix2, end_ix2), _, _ = _parse_ent(entities_dict[ent_ix])

                # do we need to sort it?
                all_rel.append([start_ix, end_ix, start_ix2, end_ix2, rel_label]) 

                count_rel += 1
                
    assert len(all_ner) == count_ner, f'{doc_key}'
    assert len(all_rel) == count_rel, f'{doc_key}'
    
    if doc_key is None and dataname is None:
        return all_ner, all_rel
    else:
        return {
            "doc_key": doc_key.split('/')[-1].split('.')[0] if '/' in doc_key else doc_key,
            "dataset": dataname,
            "sentences": [doc],
            "ner": [all_ner],
            "relations": [all_rel]
        }

def parse_radgraph_raw_df(json_df, labeler=None):
    out = []
    for i, row in json_df.iterrows():
        if labeler is None:
            line = prepare_one_case(row['text'].split(), row['entities'], i, 'radgraph')  
        else:
            assert labeler in ['labeler_1', 'labeler_2']
            line = prepare_one_case(row['text'].split(), row[labeler]['entities'], i, 'radgraph')  

        out.append(line)

    return out


def _save_raw_lines(lines, out_path):
    with open(out_path, 'w') as f:
        for l in lines:
            line = json.dumps(l)
            f.write(line)
            f.write('\n')



