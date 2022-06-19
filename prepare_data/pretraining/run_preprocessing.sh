#!/bin/bash

JOBS=30

DUMP_DIR=TEXT_DUMP
MIMIC_DIR=~/physionet.org/files/mimiciii/1.4/

stdout=logs
mkdir -p $stdout

for i in $(seq 1 $JOBS)
do 
    echo init process $i 
    nice python prepare_raw_text.py --n_jobs $JOBS --process $i \
        --MIMIC_DIR $MIMIC_DIR --DUMP_DIR $DUMP_DIR &> $stdout/log-$i.txt &
done


cat $DUMP_DIR/sentence/*.txt > $DUMP_DIR/sentences.txt
cat $DUMP_DIR/segment/*.txt > $DUMP_DIR/segments.txt
cat $DUMP_DIR/note/*.txt > $DUMP_DIR/notes.txt

rm -r $DUMP_DIR/sentence
rm -r $DUMP_DIR/segment
rm -r $DUMP_DIR/note

