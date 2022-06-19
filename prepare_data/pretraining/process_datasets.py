
import pandas as pd 
import numpy as np 
import pickle as pk

import logging
import os, argparse, json
from itertools import chain

from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizerFast

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')


class TokenizeEngine():
    def __init__(self, tokenizer, text_column_name='text', add_special_tokens=False) -> None:
        self.tokenizer = tokenizer
        self.text_column_name = text_column_name
        self.add_special_tokens = add_special_tokens

    def __call__(self, examples):
        examples[self.text_column_name] = [
            line for line in examples[self.text_column_name] if len(line) > 0 and not line.isspace()
        ]
        return self._tokenize_add_length(examples=examples[self.text_column_name])

    def _tokenize_add_length(self, examples):
        if self.add_special_tokens:
            return_dict = self.tokenizer(examples, return_special_tokens_mask=True)
        else:
            # will add them later manually to make sure [cls] and [sep] are at ends
            return_dict = self.tokenizer(examples, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False, return_special_tokens_mask=False)
        length = [len(i) for i in return_dict['input_ids']]
        return_dict.update({'length': length})
        return return_dict
            


class ProcessEngine():
    def __init__(self, MIN_SEQ_LEN, MAX_SEQ_LEN, column_names, CLS_ID=None, SEP_ID=None, process_type='sentence'):
        self.MIN_SEQ_LEN = MIN_SEQ_LEN
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.column_names = column_names
        self.process_type = process_type

        if process_type != 'sentence':
            assert not ('attention_mask' in column_names \
                and 'special_tokens_mask' in column_names \
                and 'token_type_ids' in column_names), \
                'make sure [cls] and [sep] were not added in the tokenization step'

            logging.info("need to add [cls] [sep]")
            assert CLS_ID is not None and SEP_ID is not None
            self.MAX_SEQ_LEN -= 2
            self.CLS_ID = CLS_ID
            self.SEP_ID = SEP_ID

    def __call__(self, examples):
        """
        All inputs should be truncated to max seq len
        Seg/note involve grouping, need to add special tokens to start/end
        """
        call_type = {
            'sentence': self.process_sentence,
            'segment': self.process_segment,
            'note': self.process_note,
            'note_extra': self.process_note_extra,
        }
        return call_type[self.process_type](examples)

    
    def process_sentence(self, examples):
        """
        Goal for sentence: remove too short
        """
        for col in self.column_names:
            examples[col] = [
                line[:self.MAX_SEQ_LEN] for line in examples[col] if len(line) > self.MIN_SEQ_LEN
            ]

        # check sentences have all the ids
        column_names = examples.keys()
        assert 'attention_mask' in column_names and 'special_tokens_mask' in column_names and 'token_type_ids' in column_names

        return examples

    def process_segment(self, examples):
        """
        Goal for segment: group to form long-enough segment
        """
        new_segments= []
        running_seg = []
        for seg, seg_len in zip(examples['input_ids'], examples['length']):

            ## group short segs together
            if seg_len < self.MIN_SEQ_LEN:
                running_seg.extend(seg)
                if len(running_seg) > self.MIN_SEQ_LEN:
                    new_segments.append(running_seg)
                    running_seg = []
            else:
                new_segments.append(seg)
                # discard prev short segs to keep text in original order
                running_seg = []

        # truncate to max len
        new_input_ids = [seg[:self.MAX_SEQ_LEN] for seg in new_segments]
        new_examples = self._add_ids(new_input_ids, cls_id=self.CLS_ID, sep_id=self.SEP_ID)
        return new_examples

    def process_note(self, examples):
        """
        Goal for note: chunk too-long note
        """
        new_notes = []

        for note, note_len in zip(examples['input_ids'], examples['length']):

            # only keep unique note; did not group note as did for seg
            if note_len < self.MIN_SEQ_LEN:
                continue
            else:
                # chunk note by max seq len
                num_note = note_len // self.MAX_SEQ_LEN
                if num_note > 0:
                    valid_chunks = [note[i:i+self.MAX_SEQ_LEN] for i in range(0, num_note, self.MAX_SEQ_LEN)]
                    new_notes.extend(valid_chunks)

                # see if remainder is long enough to be kept
                remainder = note_len % self.MAX_SEQ_LEN
                if remainder > self.MIN_SEQ_LEN:
                    new_notes.append(note[-remainder:])

        new_examples = self._add_ids(new_notes, cls_id=self.CLS_ID, sep_id=self.SEP_ID)
        return new_examples

    def process_note_extra(self, examples):
        """
        Goal for note-extra: create really long note
        """
        new_notes = []
        running_note = []
        for note, note_len in zip(examples['input_ids'], examples['length']):

            ## group short notes together, as did for seg
            if note_len < self.MIN_SEQ_LEN:
                running_note.extend(note)
                if len(running_note) > self.MIN_SEQ_LEN:
                    new_notes.append(running_note)
                    running_note = []
            else:
                new_notes.append(note)                

        # truncate to max len
        new_input_ids = [note[:self.MAX_SEQ_LEN] for note in new_notes]
        new_examples = self._add_ids(new_input_ids, cls_id=self.CLS_ID, sep_id=self.SEP_ID)
        return new_examples


    @staticmethod
    def _add_ids(new_ids, cls_id, sep_id):
        """add extra ids besides input_ids

        Args:
            new_ids (List): input_ids

        Return:
            tokenized dict with four ids
        """
        outdict = {
            'attention_mask': [], 'input_ids': [], 'special_tokens_mask': [], 'token_type_ids': []
        }
        for ids in new_ids:
            input_ids = [cls_id] + ids + [sep_id]
            attention_mask = [1 for _ in range(len(input_ids))]
            token_type_ids = [0 for _ in range(len(input_ids))]
            special_tokens_mask = [1] + [0 for _ in range(len(ids))] + [1]

            assert len(input_ids) == len(attention_mask) == len(token_type_ids) == len(special_tokens_mask)

            outdict['input_ids'].append(input_ids)
            outdict['attention_mask'].append(attention_mask)
            outdict['token_type_ids'].append(token_type_ids)
            outdict['special_tokens_mask'].append(special_tokens_mask)

        return outdict





def process_sentence(raw_datasets, tokenizer, MIN_SEQ_LEN, MAX_SEQ_LEN, n_jobs=1, overwrite_cache=False):
    text_column_name = 'text'

    # tokenize
    tokEngine = TokenizeEngine(tokenizer, add_special_tokens=True)
    tokenized_datasets = raw_datasets.map(
        tokEngine, batched=True, num_proc=n_jobs,
        remove_columns=text_column_name, load_from_cache_file=not overwrite_cache,
        desc="Running tokenizer on every text in dataset",
    )

    # process
    column_names = tokenized_datasets['train'].column_names
    column_names.remove('length')

    procEngine = ProcessEngine(MIN_SEQ_LEN=MIN_SEQ_LEN, MAX_SEQ_LEN=MAX_SEQ_LEN, column_names=column_names)
    processed_datasets = tokenized_datasets.map(
        procEngine, batched=True, num_proc=n_jobs, 
        remove_columns=['length'], load_from_cache_file=not overwrite_cache,
        desc="Processing sentences",
    )
    nlen = np.array([len(i) for i in processed_datasets['train']['input_ids']])

    return processed_datasets, nlen

def process_segment(raw_datasets, tokenizer, MIN_SEQ_LEN, MAX_SEQ_LEN, n_jobs=1, overwrite_cache=False):
    text_column_name = 'text'

    # tokenize
    tokEngine = TokenizeEngine(tokenizer)
    tokenized_datasets = raw_datasets.map(
        tokEngine, batched=True, num_proc=n_jobs,
        remove_columns=text_column_name, load_from_cache_file=not overwrite_cache,
        desc="Running tokenizer on every text in dataset",
    )

    # process
    column_names = tokenized_datasets['train'].column_names
    cls_id = tokenizer.vocab[tokenizer.special_tokens_map['cls_token']]
    sep_id = tokenizer.vocab[tokenizer.special_tokens_map['sep_token']]

    procEngine = ProcessEngine(MIN_SEQ_LEN=MIN_SEQ_LEN, MAX_SEQ_LEN=MAX_SEQ_LEN, 
        column_names=column_names, CLS_ID=cls_id, SEP_ID=sep_id, process_type='segment')
    processed_datasets = tokenized_datasets.map(
        procEngine, batched=True, num_proc=n_jobs, 
        remove_columns=['length'], load_from_cache_file=not overwrite_cache,
        desc="Processing segments",
    )
    nlen = np.array([len(i) for i in processed_datasets['train']['input_ids']])

    return processed_datasets, nlen


def process_note(raw_datasets, tokenizer, MIN_SEQ_LEN, MAX_SEQ_LEN, n_jobs=1, overwrite_cache=False):
    text_column_name = 'text'

    # tokenize
    tokEngine = TokenizeEngine(tokenizer)
    tokenized_datasets = raw_datasets.map(
        tokEngine, batched=True, num_proc=n_jobs,
        remove_columns=text_column_name, load_from_cache_file=not overwrite_cache,
        desc="Running tokenizer on every text in dataset",
    )

    # process
    column_names = tokenized_datasets['train'].column_names
    cls_id = tokenizer.vocab[tokenizer.special_tokens_map['cls_token']]
    sep_id = tokenizer.vocab[tokenizer.special_tokens_map['sep_token']]

    if MAX_SEQ_LEN > 512:
        procEngine = ProcessEngine(MIN_SEQ_LEN=MIN_SEQ_LEN, MAX_SEQ_LEN=MAX_SEQ_LEN, 
            column_names=column_names, CLS_ID=cls_id, SEP_ID=sep_id, process_type='note_extra')
    else:
        procEngine = ProcessEngine(MIN_SEQ_LEN=MIN_SEQ_LEN, MAX_SEQ_LEN=MAX_SEQ_LEN, 
            column_names=column_names, CLS_ID=cls_id, SEP_ID=sep_id, process_type='note')
        
    processed_datasets = tokenized_datasets.map(
        procEngine, batched=True, num_proc=n_jobs, 
        remove_columns=['length'], load_from_cache_file=not overwrite_cache,
        desc="Processing notes",
    )
    nlen = np.array([len(i) for i in processed_datasets['train']['input_ids']])

    return processed_datasets, nlen


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--DUMP_DIR", type=str, default='TEXT_DUMP')
    parser.add_argument("--CACHE_DIR", type=str, default='PROCESSED_FILES')

    parser.add_argument("--tokenizer_path", type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

    parser.add_argument("--min_seq_len_sent", type=int, default=16)
    parser.add_argument("--max_seq_len_sent", type=int, default=128)
    parser.add_argument("--min_seq_len_seg", type=int, default=128)
    parser.add_argument("--max_seq_len_seg", type=int, default=256)
    parser.add_argument("--min_seq_len_note", type=int, default=256)
    parser.add_argument("--max_seq_len_note", type=int, default=512)

    parser.add_argument("--num_note_extra", type=int, default=0)

    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--overwrite_cache", action='store_true', default=False, help="Whether to reload from start.")
    
    args = parser.parse_args()



    logging.info('Load raw dataset')
    raw_sentence= load_dataset('text', data_files=os.path.join(args.DUMP_DIR, 'sentences.txt'), cache_dir=args.CACHE_DIR)
    raw_segment = load_dataset('text', data_files=os.path.join(args.DUMP_DIR, 'segments.txt'), cache_dir=args.CACHE_DIR)
    raw_note    = load_dataset('text', data_files=os.path.join(args.DUMP_DIR, 'notes.txt'), cache_dir=args.CACHE_DIR)



    logging.info('Load tokenizer')
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path, cache_dir=args.CACHE_DIR)



    logging.info('Process dataset')
    raw_data_dict = {
        'sentence': raw_sentence, 'segment': raw_segment, 'note': raw_note
    }
    seq_len_dict = {
        'sentence': (args.min_seq_len_sent, args.max_seq_len_sent),
        'segment': (args.min_seq_len_seg, args.max_seq_len_seg),
        'note': (args.min_seq_len_note, args.max_seq_len_note),
    }
    process_dict = {
        'sentence': process_sentence, 'segment': process_segment, 'note': process_note
    }
    stat_lines = []


    ## basic sent, seg, note
    for name in ['sentence', 'segment', 'note']:
        logging.info('Tokenizing and processing '+name)

        MIN_SEQ_LEN, MAX_SEQ_LEN = seq_len_dict[name]
        processed_datasets, nlen = process_dict[name](
            raw_datasets=raw_data_dict[name], tokenizer=tokenizer,
            MIN_SEQ_LEN=MIN_SEQ_LEN, MAX_SEQ_LEN=MAX_SEQ_LEN, 
            n_jobs=args.n_jobs, overwrite_cache=args.overwrite_cache
        )

        outname = f'{name}-{MIN_SEQ_LEN}-{MAX_SEQ_LEN}'
        logging.info('Saving processed ' + outname)
        processed_datasets.save_to_disk(os.path.join(args.CACHE_DIR, outname))

        stat_line = f'Num lines: {len(nlen):>15,}\nAverage length: {nlen.mean():>10.0f}\nTotal tokens: {nlen.sum():>12,}'
        stat_lines.append(stat_line)

    ## extra long notes
    MIN_SEQ_LEN, MAX_SEQ_LEN = seq_len_dict['note']
    for _ in range(args.num_note_extra):
        MIN_SEQ_LEN = MIN_SEQ_LEN * 2
        MAX_SEQ_LEN = MAX_SEQ_LEN * 2

        processed_datasets, nlen = process_note(
            raw_datasets=raw_note, tokenizer=tokenizer,
            MIN_SEQ_LEN=MIN_SEQ_LEN, MAX_SEQ_LEN=MAX_SEQ_LEN, 
            n_jobs=args.n_jobs, overwrite_cache=args.overwrite_cache
        )

        outname = f'note-{MIN_SEQ_LEN}-{MAX_SEQ_LEN}'
        logging.info('Saving processed ' + outname)
        processed_datasets.save_to_disk(os.path.join(args.CACHE_DIR, outname))

        stat_line = f'Num lines: {len(nlen):>15,}\nAverage length: {nlen.mean():>10.0f}\nTotal tokens: {nlen.sum():>12,}'
        stat_lines.append(stat_line)





    logging.info('Finished, stdout stat')
    with open(os.path.join(args.CACHE_DIR, f'stat-{args.min_seq_len_sent}.txt'), 'w') as f:
        names = ['sentence', 'segment', 'note'] + ['note-extra' for _ in range(args.num_note_extra)]
        for name, line in zip(names, stat_lines):
            f.write(name.upper()+'\n')
            f.write(line)
            f.write('\n\n')


if __name__ == "__main__":
    main()