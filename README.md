# ClinicalNoteBERT

This repository present codes to pretrain and fine-tune ClinicalNoteBERT--the recent adaptation of BERT-style pretrained language model to clinical text. Using clinical pretraining corpus and a range of downstream tasks in the clinical domain, the project studies the impact of pretraining input sequence length on the effectiveness of domain adaptation. Our results show having prolonged input sequences during pretraining is essential to achieve successful adaptation for the clinical domain given its ungrammatical text and contextual relationships. Our best checkpoint from the experiments is denoted as [ClinicalNoteBERT](https://huggingface.co/jhliu/ClinicalNoteBERT-base-uncased-MIMIC-segment-note), which is available at Huggingface Model Hub with other checkpoints we experimented in the paper. 


To reproduce our experiments, we present scripts on 1) preparing data for pretraining and downstream evaluation, 2) pretraining with different versions of clinical corpus, and 3) fine-tuning for the downstream tasks. Most of our code is based on Huggingface implementation and it should be easy to use the fine-tuning tasks as benchmark to examine other models released on the platform. The only exception is the end-to-end relation extraction using [DyGIE++](https://github.com/dwadden/dygiepp), a well-maintained repo that is easy to adapt; and we also provide some sample commands of using it. 


## 1. Pretrained Models
We release all our checkpoints on Huggingface, including other checkpoints we pretrained to examine the impact of pretraining input length. 

|                      | Pretraining Corpora (Objective) | Download Link                                                |
| -------------------- | ------------------------------- | ------------------------------------------------------------ |
| ClinicalNoteBERT     | segment + note (MLM)            | [HF](https://huggingface.co/jhliu/ClinicalNoteBERT-base-uncased-MIMIC-segment-note) |
| ClinicalNoteBERT-NTD | segment (MLM + NTD)             | [HF](https://huggingface.co/jhliu/ClinicalNoteBERT-base-uncased-NTD-MIMIC-segment) |
| ClinicalNoteBERT-NTP | note (MLM + NTP)                | [HF](https://huggingface.co/jhliu/ClinicalNoteBERT-base-uncased-NTP-MIMIC-note) |
| PubMedBERT+sentence  | sentence  (MLM)                 | [HF](https://huggingface.co/jhliu/ClinicalAdaptation-PubMedBERT-base-uncased-MIMIC-sentence) |
| PubMedBERT+segment   | segment (MLM)                   | [HF](https://huggingface.co/jhliu/ClinicalAdaptation-PubMedBERT-base-uncased-MIMIC-segment) |
| PubMedBERT+note      | note  (MLM)                     | [HF](https://huggingface.co/jhliu/ClinicalAdaptation-PubMedBERT-base-uncased-MIMIC-note) |

They can be used with the Transformers library, such as

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('jhliu/ClinicalNoteBERT-base-uncased-MIMIC-segment-note')
model = AutoModel.from_pretrained('jhliu/ClinicalNoteBERT-base-uncased-MIMIC-segment-note')
```

The checkpoints are also available on [gdrive](https://drive.google.com/drive/folders/1X2B2QHNRnysb2Jg4eVpHVPct7YKzrZ7B?usp=sharing).

## 2. Prepare Data

We provide scripts to prepare and process the datasets used in our experiments, including both pretraining and fine-tuning. We refer readers to `prepare_data/README.md` for more details.



## 3. Pretraining

We provide our main script to pretrain our domain-specific models with Masked Language Modeling in `pretrain` folder, which is based on the Huggingface implementation. ClinicalNoteBERT is based on pretraining first on `segment` and then on `note`, which just requires running: 

```sh
# segment
CONFIG_FILE=config-segment.json
torchrun --nproc_per_node=$PROC run_mlm.py $CONFIG_FILE

# segment -> note
CONFIG_FILE=config-segment-note.json
torchrun --nproc_per_node=$PROC run_mlm.py $CONFIG_FILE

```

But you can definitely try other options on your own, such as pretrain with `sentence`:

```sh
# sentence
CONFIG_FILE=config-sentence.json
torchrun --nproc_per_node=$PROC run_mlm.py $CONFIG_FILE

```

All configs are HF `Trainer` flags, and you can adjust data directory and other hyperparameters accordingly.



## 4. Fine-tuning for downstream tasks

The fine-tuning scripts for NLI, NER, and QA are all modified from the Huggingface implementation. All scripts are available in `fine-tune` folder. Here is an example of training and evaluating on the MedNLI dataset. 

```sh
export PLM_CKPT=jhliu/ClinicalNoteBERT-base-uncased-MIMIC-segment-note
# specificy DATA_DIR, OUTPUT_DIR, and CACHE as necessary
export DATA_DIR=/path/to/nli_data
export OUTPUT_DIR=/path/to/output
export CACHE_DIR=/path/to/cache

# NLI
python run_sp.py \
    --model_name_or_path $PLM_CKPT \
    --dataset_name MedNLI \
    --dataset_raw_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --save_strategy no \
    --num_train_epochs 4 \
    --learning_rate 3e-5 \
    --lr_scheduler_type constant \
    --warmup_ratio 0.1 \
    --max_seq_length 128
```

We test the above run on a 16GB V100 GPU with `torch==1.10.0` and `transformers==4.12.5` and we can print out the following testset result. 

```sh
***** predict metrics *****
  predict_accuracy           =     0.8769
  predict_loss               =     0.4477
  predict_runtime            = 0:00:03.39
  predict_samples_per_second =    419.447
  predict_steps_per_second   =     13.274
```

Here is also a sample runs for NER. 


```sh
# NER
# with i2b2 2012 as an example
python run_ner.py \
    --model_name_or_path $PLM_CKPT \
    --dataset_name i2b2_2012 \
    --dataset_ckpt_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --save_strategy no \
    --num_train_epochs 5 \
    --learning_rate 5e-5 \
    --lr_scheduler_type constant \
    --warmup_ratio 0.1 \
    --max_seq_length 150 # this flag follows ClinicalBERT
    
    
# we could print out
***** predict metrics *****
  predict_accuracy           =     0.9154
  predict_f1                 =     0.8045
  predict_loss               =     0.3331
  predict_precision          =     0.7933
  predict_recall             =     0.8159
  predict_runtime            = 0:00:03.58
  predict_samples_per_second =   1580.617
  predict_steps_per_second   =     49.394
```

And for QA.


```sh
# QA
# with Medication as an example
export PROC=4 # num of workers for preprocessing/tokenization
python run_qa.py \
    --model_name_or_path $PLM_CKPT \
    --dataset_name medication \
    --dataset_raw_dir $DATA_DIR \
    --output_dir $OUTPUT \
    --dataset_cache_dir $CACHE \
    --cache_dir $CACHE \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --save_strategy no \
    --num_train_epochs 5 \
    --learning_rate 5e-5 \
    --lr_scheduler_type constant \
    --warmup_ratio 0.1 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --preprocessing_num_workers $PROC

```

Relation extraction relies on the DyGIE++ repository, which needs to be cloned to install their required dependencies. We refer readers to their repo for more details to setup and run DyGIE++, but we provide the config file and a sample run as follow:

```sh
cd dygiepp

# train
export JFILE=/path/to/rad.jsonnet
export OUTPUT=/path/to/model_cache/rad
allennlp train "${JFILE}" --serialization-dir "${OUTPUT}" --include-package dygie

# with labeler 1 from MIMIC-CXR as example
# inference
export CKPT=$OUTPUT/model.tar.gz
export TEST_FILE=/path/to/data/radgraph/test-MIMIC-CXR-labeler_1.jsonl
export PRED_FILE=/path/to/data/radgraph/pred-MIMIC-CXR-labeler_1.jsonl

allennlp predict $CKPT $TEST_FILE --predictor dygie --include-package dygie --use-dataset-reader --output-file $PRED_FILE --cuda-device 0 --silent

# eval
cd ../fine-tune
python run_eval.py --prediction_file $PRED_FILE

```
