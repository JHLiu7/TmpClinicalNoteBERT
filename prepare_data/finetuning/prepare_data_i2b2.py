
## Prepare four i2b2 datasets following the procedures from the clincal bert repo https://github.com/EmilyAlsentzer/clinicalBERT/blob/master/downstream_tasks/run_ner.py
## This levarges the splitting function to get the same numbers of train/dev/test cases to match the paper.
## However, we did not reply on tf and the outdated pytorch-transformer tokenization.
## Then the data will be processed and saved using the Huggingface datasets library.

## Requirements: i2b2 data should be prepared using the notebooks/scripts in https://github.com/EmilyAlsentzer/clinicalBERT/tree/master/downstream_tasks/i2b2_preprocessing
## and put into a folder.  (Note: we found one i2b2 dataset had some issue with xml parsing given & was not escaped in several lines. They can be easily fixed by manually escaping them, i.e., `&` -> `&amp;`)
## DATA_DIR 
##   |-- i2b2_2006
##          |-- train.tsv
##          |-- dev.tsv
##          |-- test.tsv
##   |-- i2b2_2010
##          |-- train.tsv
##          |-- dev.tsv
##          |-- test.tsv
##   |-- i2b2_2012
##          |-- train.tsv
##          |-- dev.tsv
##          |-- test.tsv
##   |-- i2b2_2014
##          |-- train.tsv
##          |-- dev.tsv
##          |-- test.tsv

import argparse
import os
import math

import pickle
import numpy as np
import itertools
import json
from random import shuffle
import random
from collections import defaultdict

import datasets
from datasets import Features, Sequence, Value, ClassLabel

class TmpTok:
    def __init__(self) -> None:
        self.yes = None
    def convert_to_unicode(self, w):
        return w
tokenization = TmpTok() # to replace the older tokenizer used in the repo and perform tokenization later

######################### code below from clinicalbert repo w/ modifications on toknzer, label set, etc ####################

class InputExample(object):
    """A single training/test example for simple token classification."""

    def __init__(self, guid, tokens, text=None, labels=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text: (Optional) string. The untokenized text of the sequence. 
          tokens: list of strings. The tokenized sentence. Each token should have a 
            corresponding label for train and dev samples. 
          label: (Optional) list of strings. The labels of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.tokens = tokens
        self.labels = labels

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "id: {}".format(self.guid),
            "tokens: {}".format(" ".join(self.tokens)),
        ]
        if self.text is not None:
            l.append("text: {}".format(self.text))

        if self.labels is not None:
            l.append("labels: {}".format(" ".join(self.labels)))

        return ", ".join(l)



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids,):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        #self.label_mask = label_mask

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads in data where each line has the word and its corresponding 
        label separated by whitespace. Each sentence is separated by a blank
        line. E.g.:

        Identification  O
        of  O
        APC2    O
        ,   O
        a   O
        homologue   O
        of  O
        the O
        adenomatous B-Disease
        polyposis   I-Disease
        coli    I-Disease
        tumour  I-Disease
        suppressor  O
        .   O

        The O
        adenomatous B-Disease
        polyposis   I-Disease
        ...
        """
        with open(input_file) as f:
            lines = []
            words = []
            labels = []
            for line in f:
                line = line.strip()
                if len(line) == 0: #i.e. we're in between sentences
                    assert len(words) == len(labels)
                    if len(words) == 0:
                        continue
                    lines.append([words, labels])
                    words = []
                    labels = []
                    continue
                
                word = line.split()[0]
                label = line.split()[-1]
                words.append(word)
                labels.append(label)

            #TODO: see if there's an off by one error here
            return lines

    @classmethod
    def _create_example(self, lines, set_type):
            examples = []
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                words,labels = line
                words = [tokenization.convert_to_unicode(w) for w in words]
                labels = [tokenization.convert_to_unicode(l) for l in labels]
                examples.append(InputExample(guid=guid, tokens=words, labels=labels))
            return examples

    @classmethod
    def _chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    # def write_cv_to_file(self, evaluation, test, n):
    #     with open(os.path.join(DATA_DIR, str(n) + '_eval'),'w') as w:
    #         for example in evaluation:
    #             for t, l in zip(example.tokens, example.labels):
    #                 w.write("%s %s\n" %(t, l))
    #             w.write("\n")
                

    #     with open(os.path.join(DATA_DIR, str(n) + '_test'),'w') as test_w:
    #         for test_example in test:
    #             for t, l in zip(test_example.tokens, test_example.labels):

    #                 test_w.write("%s %s\n" %(t, l))
    #             test_w.write("\n")
                

    def get_cv_examples(self, splits, n, cv_sz=10):
        # note, when n=9 (10th split), this recovers the original train, dev, test split

        dev = splits[(n-1)%cv_sz] #4 #0 #3 #1
        test = splits[n] #0 #1 #4 #2
        # print('train ind: %d-%d' %((n+1)%cv_sz, (n-1)%cv_sz))
        # print('dev ind: %d' %((n-1)%cv_sz))
        # print('test ind`: %d' %n)
        if (n+1)%cv_sz > (n-1)%cv_sz:
            train = splits[:(n-1)%cv_sz] + splits[(n+1)%cv_sz:]
        else:
            train = splits[(n+1)%cv_sz:(n-1)%cv_sz] #1-3 #2-4 #0-2 #3-0s
        train = list(itertools.chain.from_iterable(train))
        print("Train size: %d, dev size: %d, test size: %d, total: %d" %(len(train), len(dev), len(test), (len(train)+len(dev)+len(test))))
        # self.write_cv_to_file(dev, test, n)
        return(train, dev, test)

    def create_cv_examples(self, data_dir, cv_sz=10):
        train_examples = self.get_train_examples(data_dir)
        dev_examples = self.get_dev_examples(data_dir)
        test_examples = self.get_test_examples(data_dir)
        print('num train examples: %d, num eval examples: %d, num test examples: %d' %(len(train_examples), len(dev_examples), len(test_examples)))
        print('Total dataset size: %d' %(len(train_examples) + len(dev_examples) + len(test_examples)))
        random.seed(42)
        train_dev = train_examples + dev_examples
        random.shuffle(train_dev)
        split_sz = math.ceil(len(train_dev)/(cv_sz-1))
        print('Split size: %d' %split_sz)
        splits = list(self._chunks(train_dev, split_sz))
        print('Num splits: %d' %(len(splits) + 1))
        splits = splits + [test_examples]
        print('len splits: ', [len(s) for s in splits])
        return splits


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train_dev.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "devel.tsv")), "dev"
        )

    def get_test_examples(self,data_dir):
        test_examples = self._create_example(
            self._read_data(os.path.join(data_dir, "test.tsv")), "test")
        #print(test_examples)
        return test_examples


    def get_labels(self):
        return ["B", "I", "O", "X", "[CLS]", "[SEP]"] 

  
class i2b22010Processor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self,data_dir):
        print('Path: ', os.path.join(data_dir, "test.tsv"))
        test_examples = self._create_example(
            self._read_data(os.path.join(data_dir, "test.tsv")), "test")
        print(test_examples[-5:])
        return test_examples

    def get_labels(self):
        # return ["B-problem", "I-problem", "B-treatment", "I-treatment", 'B-test', 'I-test', 'O', "X", "[CLS]", "[SEP]"] 
        return ["B-problem", "I-problem", "B-treatment", "I-treatment", 'B-test', 'I-test', 'O'] #, "X", "[CLS]", "[SEP]"] 

class i2b22006Processor(DataProcessor):


    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.conll")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.conll")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.conll")), "test"
        )

    def get_labels(self):
        return ["B-ID", "I-ID", "B-HOSPITAL", "I-HOSPITAL", 'B-PATIENT', 'I-PATIENT', 'B-PHONE', 'I-PHONE',
        'B-DATE', 'I-DATE', 'B-DOCTOR', 'I-DOCTOR', 'B-LOCATION', 'I-LOCATION', 'B-AGE', 'I-AGE',
        # 'O', "X", "[CLS]", "[SEP]"]
        'O']

class i2b22012Processor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self):
        return ['B-OCCURRENCE','I-OCCURRENCE','B-EVIDENTIAL','I-EVIDENTIAL','B-TREATMENT','I-TREATMENT','B-CLINICAL_DEPT',
        # 'I-CLINICAL_DEPT','B-PROBLEM','I-PROBLEM','B-TEST','I-TEST','O', "X", "[CLS]", "[SEP]"] 
        'I-CLINICAL_DEPT','B-PROBLEM','I-PROBLEM','B-TEST','I-TEST','O'] #, "X", "[CLS]", "[SEP]"] 

class i2b22014Processor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self):
        # return ["B-IDNUM", "I-IDNUM", "B-HOSPITAL", "I-HOSPITAL", 'B-PATIENT', 'I-PATIENT', 'B-PHONE', 'I-PHONE',
        # 'B-DATE', 'I-DATE', 'B-DOCTOR', 'I-DOCTOR', 'B-LOCATION-OTHER', 'I-LOCATION-OTHER', 'B-AGE', 'I-AGE', 'B-BIOID', 'I-BIOID',
        # 'B-STATE', 'I-STATE','B-ZIP', 'I-ZIP', 'B-HEALTHPLAN', 'I-HEALTHPLAN', 'B-ORGANIZATION', 'I-ORGANIZATION',
        # 'B-MEDICALRECORD', 'I-MEDICALRECORD', 'B-CITY', 'I-CITY', 'B-STREET', 'I-STREET', 'B-COUNTRY', 'I-COUNTRY',
        # 'B-URL', 'I-URL',
        # 'B-USERNAME', 'I-USERNAME', 'B-PROFESSION', 'I-PROFESSION', 'B-FAX', 'I-FAX', 'B-EMAIL', 'I-EMAIL', 'B-DEVICE', 'I-DEVICE',
        # 'O', "X", "[CLS]", "[SEP]"] 
        # I-BIOID, I-ZIP, I-USERNAME, I-EMAIL did not appear across the dataset
        return ["B-IDNUM", "I-IDNUM", "B-HOSPITAL", "I-HOSPITAL", 'B-PATIENT', 'I-PATIENT', 'B-PHONE', 'I-PHONE',
        'B-DATE', 'I-DATE', 'B-DOCTOR', 'I-DOCTOR', 'B-LOCATION-OTHER', 'I-LOCATION-OTHER', 'B-AGE', 'I-AGE', 'B-BIOID', 
        'B-STATE', 'I-STATE','B-ZIP', 'B-HEALTHPLAN', 'I-HEALTHPLAN', 'B-ORGANIZATION', 'I-ORGANIZATION',
        'B-MEDICALRECORD', 'I-MEDICALRECORD', 'B-CITY', 'I-CITY', 'B-STREET', 'I-STREET', 'B-COUNTRY', 'I-COUNTRY',
        'B-URL', 'I-URL', 'B-USERNAME', 'B-PROFESSION', 'I-PROFESSION', 'B-FAX', 'I-FAX', 'B-EMAIL', 'B-DEVICE', 'I-DEVICE', 'O']  


####################################################################################################


def prepare_one_dataset(data_split, label_list, processor, name):
    print('Data split size for', name)

    ## get label dict
    label_dict = {'O':0}
    for label in label_list:
        if label not in label_dict:
            label_dict[label] = len(label_dict)
            
    label_list = list(label_dict.keys())
    print(len(label_list), 'tags')

    train_examples, eval_examples, test_examples = processor.get_cv_examples(data_split, 9, cv_sz=10)
    
    print()

    ## parse row, quickly
    def row2tuple(l, label2tag):
        ID, tok = str(l).split(', tokens: ')
        ID = ID.replace('id: ', '')
        tok, lb = tok.split(', labels: ')
        tok, lb = tok.split(), lb.split()
        tag = [ label2tag[l] for l in lb ]
        assert len(tok) == len(tag)
        return ID, tok, lb, tag

    train, dev, test = defaultdict(list), defaultdict(list), defaultdict(list)
    for data, examples in zip([train, dev, test], [train_examples, eval_examples, test_examples]):
        for example in examples:
            ID, tok, _, tag = row2tuple(example, label_dict)
            data['id'].append(ID)
            data['tokens'].append(tok)
            data['ner_tags'].append(tag)
    

    ## create dataset w/ datasets 
    features = Features({
        'id': Value('string'),
        'tokens': Sequence(Value('string')),
        'ner_tags': Sequence(ClassLabel(names=label_list))
    })

    dataset = datasets.DatasetDict({
        'train': datasets.Dataset.from_dict(
            train, features=features
        ),
        'validation': datasets.Dataset.from_dict(
            dev, features=features
        ),
        'test': datasets.Dataset.from_dict(
            test, features=features
        ),
    })

    return dataset





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", type=str, default='I2B2_DATA_DIR')
    parser.add_argument("--OUTPUT_DIR", type=str, default='HF_CACHED')
    args = parser.parse_args()

    processors = {
        "i2b2_2006": i2b22006Processor,
        "i2b2_2010": i2b22010Processor,
        "i2b2_2012": i2b22012Processor,
        "i2b2_2014": i2b22014Processor,
    }
    dsets = processors.keys()

    data_splits = []
    for dset in dsets:
        
        processor = processors[dset]()
        splits = processor.create_cv_examples(os.path.join(args.DATA_DIR, dset), cv_sz=10)
        label_list = processor.get_labels()
        data_splits.append((splits, label_list, processor, dset))

    
    print('\n\n\nPreparing four i2b2 datasets')
    os.makedirs(args.OUTPUT_DIR, exist_ok=True)
    for dset, data_split in zip(dsets, data_splits):
        dataset = prepare_one_dataset(*data_split)

        dataset.save_to_disk(os.path.join(args.OUTPUT_DIR, dset))

    print('\nAll four i2b2 datasets dumped to', args.OUTPUT_DIR)






