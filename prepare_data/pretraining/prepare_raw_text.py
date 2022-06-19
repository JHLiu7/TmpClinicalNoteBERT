import pandas as pd 
import numpy as np 
import pickle as pk

import logging
import re
import os, argparse
from multiprocessing import Pool
from tqdm import tqdm
from itertools import chain

import spacy
from heuristic_tokenize import sent_tokenize_rules

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')

NOTE_CATEGORIES = ['radiology', 'nursing', 'others', 'physician', 'dsummary']
def _fix_category(cat):
    if cat == 'Radiology':
        ncat = 'radiology'
    elif cat == 'Nursing/other' or cat == 'Nursing':
        ncat = 'nursing'
    elif cat == 'Physician ':
        ncat = 'physician'
    elif cat == 'Discharge summary':
        ncat = 'dsummary'
    else:
        ncat = 'others'
    return ncat

def _simple_strip(text):
    return text.replace('[**','').replace('**]', '')
def _clean_sent(sent):
    return sent.strip().replace('\n', ' ')

def _clean_deid(text):
    return re.sub(r'\[\*\*(.*?)\*\*\]', _clean_matched, text)

def _clean_matched(matched):
    """
    applied to re.sub to further clean phi placeholders
    
    e.g.: 
        [**Last Name (NamePattern4) 1604**] --> Last Name
        [**MD Number(1) 1605**] --> MD Number
        [**2101-11-5**] --> 2101-11-5
    
    """
    phi = matched.group(1)
    phi = phi.strip()
    if phi == '':
        return phi.strip()
    
    # remove final id
    if ' ' in phi:
        pl = phi.split()
        if pl[-1].isnumeric():
            phi = ' '.join(pl[:-1])
        else:
            phi = ' '.join(pl)
    
    # remove (Name Pattern) etc
    phi = re.sub(r'\(.*?\)', '', phi)
    phi = phi.strip()
    return phi

class SimpleProcessEngine(object):
    def __init__(self) -> None:
        super().__init__()
        self.nlp = spacy.load('en_core_sci_sm')

    def __call__(self, raw_note):
        """
            raw_note: str
            clean_sent: List, clean_segment: List, clean_note: Str
        """
        clean_sent, clean_segment, clean_note = self.process_note(raw_note)
        return clean_sent, clean_segment, clean_note

    def process_note(self, note):
    
        # note -> seg -> sent
        raw_note = _clean_deid(note)
        raw_segment = sent_tokenize_rules(raw_note)
        raw_sents = [
            self.nlp(seg).sents for seg in raw_segment 
            if set(seg) != {'_'} and set(seg) != {'-'}
        ]
        
        # clean sent
        clean_sents = [
            [_clean_sent(sent.text) for sent in sents] 
            for sents in raw_sents
        ]
        
        # sent -> seg -> note
        clean_sent =  [sent for sent in list(chain(*clean_sents)) if sent.strip() != '']
        clean_segment=[' '.join(sents) for sents in clean_sents]
        clean_note = ' '.join(clean_segment)
        
        return clean_sent, clean_segment, clean_note


def main():
    """process text into sent, seg, note
    
    The approach: first segment notes, then use spacy to define sentences, then clean sentences.
            Afterwards, resume segments and notes.

    Processing with multiple python commands. Tried multiple ways but either not working as expected or not showing progress.
    Temp fix: chunkize all notes and init multiple python runs to process by bg them. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--MIMIC_DIR", type=str, default='./')
    parser.add_argument("--DUMP_DIR", type=str, default='TEXT_DUMP')
    
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--process", type=int, default=0)

    parser.add_argument("--chunk_by_note", action='store_true', default=False)
    parser.add_argument("--debug", "-d", action='store_true', default=False)

    args = parser.parse_args()

    
    if args.debug:
        # sample run
        df = pd.read_csv(os.path.join('../notes-HADMID-117448.csv'), usecols=['ROW_ID', 'SUBJECT_ID', 'CATEGORY', 'TEXT'])
        df = df.head(40)
    else:
        logging.info(f'loading MIMIC-III NOTEEVENTS')
        df = pd.read_csv(os.path.join(args.MIMIC_DIR, 'NOTEEVENTS.csv.gz'), usecols=['ROW_ID', 'SUBJECT_ID', 'CATEGORY', 'TEXT'])


    # notes belonging to no admission were not removed
    # there are 231,836/2,083,180 of them in MIMIC-III

    logging.info(f'Removing empty notes')
    df['length'] = df['TEXT'].apply(lambda x: len(x.strip()))
    df = df[df['length']>0]

    logging.info(f'Grouping by note types')
    df['note_type'] = df['CATEGORY'].apply(_fix_category)
    df = df.sort_values(by=['note_type', 'SUBJECT_ID'])

    all_notes = df['TEXT'].tolist()

    engine = SimpleProcessEngine()


    if args.n_jobs > 1 and args.process > 0:

        JOBS = args.n_jobs
        PROC = args.process
        logging.info(f'Chunking notes')
        
        if args.chunk_by_note:
            # This is a simple way to put a fix number of notes into each process
            ALL_LEN = len(all_notes)
            CHUNKSIZE = int(ALL_LEN // JOBS)+1
            chunk_notes = all_notes[(PROC-1)*CHUNKSIZE : PROC*CHUNKSIZE]
            logging.info(f'Cleaning and processing {PROC}/{JOBS} chunks of {len(chunk_notes)} notes into sent, seg, note')

            # However, since different note types are grouped together, notes like d-summary
            # would be unproportionally longer than others. Having them together can be about 
            # 10 times longer than notes like radiology in terms of content. 
        else:
            # So the alternative is to put a fix number of words into each process.
            word_len = [len(note.split()) for note in all_notes]
            roll_len = np.cumsum(word_len)
            ALL_WORD_LEN = sum(word_len)
            CHUNKSIZE = int(ALL_WORD_LEN // JOBS)+1

            targets = (roll_len >= (PROC-1) * CHUNKSIZE) & (roll_len < PROC * CHUNKSIZE)
            chunk_notes = [all_notes[i] for i, m in enumerate(targets) if m == True]

            words = sum([word_len[i] for i, m in enumerate(targets) if m == True])

            logging.info(f'Cleaning and processing {PROC}/{JOBS} chunks of {len(chunk_notes)} notes ({words} words) into sent, seg, note')

        out = []
        for note in tqdm(chunk_notes):
            out.append(engine(note))

    else:

        logging.info(f'Processing all {len(all_notes)} notes with ')

        out = [engine(note) for note in all_notes]


        # with Pool(args.n_jobs) as pool:
        #     out =  list(tqdm(pool.imap(engine, all_notes), total=len(all_notes)))
            # out =  tqdm(pool.imap(engine, all_notes), total=len(all_notes))
            # out = pool.map(engine, all_notes)
    
    all_sent = [sent for sents in [tup[0] for tup in out] for sent in sents]
    all_segs = [seg for segs in [tup[1] for tup in out] for seg in segs]
    all_note = [tup[2] for tup in out]

    os.makedirs(args.DUMP_DIR, exist_ok=True)

    for all_lines, name in zip([all_sent, all_segs, all_note], ['sentence', 'segment', 'note']):
        num_lines = len(all_lines)
        num_tokens = len(' '.join(all_lines).split())
        logging.info(f'Obtained {num_lines:,} lines of {name}: in total {num_tokens:,} tokens')

        if args.process == 0:
            outpath = os.path.join(args.DUMP_DIR, f'{name}s.txt')
        else:
            tmp_dir = os.path.join(args.DUMP_DIR, name)
            os.makedirs(tmp_dir, exist_ok=True)
            outpath = os.path.join(tmp_dir, f'{args.process:0>2d}-process.txt')

        with open(outpath, 'w') as f:
            for line in all_lines:
                f.write(line)
                f.write('\n')

    logging.info(f'Dumped text to {args.DUMP_DIR}')


if __name__ == "__main__":
    main()