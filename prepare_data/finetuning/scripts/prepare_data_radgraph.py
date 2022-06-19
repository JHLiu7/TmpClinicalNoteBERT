import pandas as pd
import numpy as np 
import os, json, argparse

from collections import Counter
from utils import parse_radgraph_raw_df, _save_raw_lines

def main(args):

    raw_dir = args.DATA_DIR

    processed_dir = os.path.join(raw_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # train dev
    for fold in ['train', 'dev']:
        raw_df = pd.read_json(os.path.join(raw_dir, fold+'.json')).transpose()
        lines = parse_radgraph_raw_df(raw_df)
        _save_raw_lines(lines, os.path.join(processed_dir, fold+'.jsonl'))
        
        
    # special case for test -> slicing by source and labeler
    fold = 'test'
    test = pd.read_json(os.path.join(raw_dir, fold+'.json')).transpose()
    
    # two sources from labeler 1
    lines = parse_radgraph_raw_df(test, labeler='labeler_1')
    _save_raw_lines(lines, os.path.join(processed_dir, fold+'.jsonl'))
    
    # separate soources and labels
    for source in ['MIMIC-CXR', 'CheXpert']:
        for labeler in ['labeler_1', 'labeler_2']:
            raw_df = test[test.data_source == source]
            lines = parse_radgraph_raw_df(raw_df, labeler=labeler)
            fname = os.path.join(processed_dir, f'test-{source}-{labeler}.jsonl')
            _save_raw_lines(lines, fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", type=str, default='radgraph', help='Should have train.json, dev.json, test.json')
    args = parser.parse_args()

    main(args)