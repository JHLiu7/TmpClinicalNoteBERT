## Prepare datasets

Here are the instructions to prepare data for the pretraining corpora and the downstream fine-tuning tasks presented in our project. The pretraining corpora include three versions: `sentence`, `segment`, and `note`. The fine-tuning benchmark include seven datasets: MedNLI, four i2b2 NER datasets, RadGraph (with two test sets), and emrQA (with three subsets).

### 1. Pretraining datasets

Step 1. We load and clean raw notes from the MIMIC-III database, which preprocesses the text by stripping the PHI placeholders, split into sections, and extract sentences with Scispacy (`en_core_sci_sm`). Sentences will then be grouped in sections and notes as separate copies. To speed up the preprocessing, we recommend having multiple cores to run in parallel. In our case, we run with 30 cpus in parallel to finish the preprocessing in several hours. Use the command below to run:

~~~sh
# modify directories and number of jobs accordingly
cd pretraining
bash run_preprocessing.sh

~~~

Step 2. We create `sentence`, `segment`, and `note` corpora with different sequence lengths, as described in the paper. Here we use the PubMedBERT tokenizer as most our experiments are based on it. To prepare data to pretrain from BioBERT or other models, you can simply change the flag `tokenizer_path` to the corresponding HuggingFace model names. The processed pretraining corpora will be saved by HF `datasets` and can be directly loaded from disk. Here's an example command:

```sh
nice python process_datasets.py --n_jobs 32
```

 Running in parallel with `BerTokenizerFast `should complete the processing within hours. Then the data should be ready for pretraining. 



### 2. Fine-tuning datasets

To evaluate the strengths of BERT-style, transformer encoder-based models adapted to the clinical text, we include the five datasets presented by ClinicalBERT on NLI and NER. We further include the recent RE dataset, RadGraph, and the extractive QA dataset, emrQA, to the benchmark. For RadGraph, we leverage the DyGIE++ framework, and observe the impact of using different pre-trained checkpoints. As shown in the previous works, switching to the domain specific checkpoints can bring substantial improvements for downstream performances on end-to-end RE, so we use it to observe the impact of our pre-training recipe on RE. For emrQA, we follow previous work to focus on its three subsets: Medication, Relation, and Heart-Risk, which cover most of the emrQA samples. 

We format all the datasets into HF `datasets.DataDict` objects for easy loading with the fine-tuning scripts, but this is sometimes done with the fine-tuning script to align with tokenization.

**i2b2 datasets**: To keep consisent with the original ClinicalBERT paper, we follow their code to preprocess the datasets and split them into train/val/test sets. This involves first downloading the i2b2 datasets from the n2c2 portal and preprocess them using the code/notebook from the ClinicalBERT [repository](https://github.com/EmilyAlsentzer/clinicalBERT/tree/master/downstream_tasks/i2b2_preprocessing). Notice we found there may be an issue with XML parsing when preprocessing one i2b2 dataset using the code, which is due to `&` is not escaped properly in the original file. If this ever happened, it can be easily fixed by escaping them in XML by replacing `&` by `&amp;`. 

After the initial preprocessing following their code, you should have files in the CoNLL format for each dataset. It is recommended to put them in a directory like this: 

```sh
 I2B2_DATA_DIR 
 |-- i2b2_2006
        |-- train.tsv
        |-- dev.tsv
        |-- test.tsv
 |-- i2b2_2010
        |-- train.tsv
        |-- dev.tsv
        |-- test.tsv
 |-- i2b2_2012
        |-- train.tsv
        |-- dev.tsv
        |-- test.tsv
 |-- i2b2_2014
        |-- train.tsv
        |-- dev.tsv
        |-- test.tsv
```

We then adopt their [code](https://github.com/EmilyAlsentzer/clinicalBERT/blob/master/downstream_tasks/run_ner.py) in splitting the loading and processing the datasets, but only to the point to obtain the same train/val/test splits as described in the paper. Afterwards, we convert the datasets into the Huggingface Datasets format, similar to their release on CoNLL as well. These two steps are done by running:

```sh
python prepare_data_i2b2.py --DATA_DIR $I2B2_DATA_DIR --OUTPUT_DIR $HF_CACHED

```

It should print out the dataset stats as described in the original paper:

```sh
Data split size for i2b2_2006
17 tags
Train size: 44392, dev size: 5547, test size: 18095, total: 68034

Data split size for i2b2_2010
7 tags
Train size: 14504, dev size: 1809, test size: 27624, total: 43937

Data split size for i2b2_2012
13 tags
Train size: 6624, dev size: 820, test size: 5664, total: 13108

Data split size for i2b2_2014
43 tags
Train size: 45232, dev size: 5648, test size: 32586, total: 83466

```

These i2b2 datasets are then ready to be loaded into HF `datasets` interface and used to fine-tune a model like BERT.



**RadGraph**: The dataset can be downloaded from [physionet](https://physionet.org/content/radgraph/1.0.0/) after signing a data agreement. We put the three JSON files (train/dev/test) in the `radgraph` folder and run the following to dump jsonl data to `JSON_CACHED`, which can be consumed by DyGIE++. 

```sh
bash run_process_re.sh
```



**emrQA**: Since there's no official train/dev/test split from emrQA, we follow an existing work to create the datasets, and focus on the three most prevelant subsets of emrQA. The original emrQA can be downloaded [here](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/) or following their [repo](https://github.com/panushri25/emrQA). Then you can run the command from [CliniRC](https://github.com/xiangyue9607/CliniRC) to obtain the train/dev/test sets in SQuAD style:

```sh
python preprocessing_emrqa.py

```

We then downsample the train and dev sets to 10% as indicated by the CliniRC paper in the fine-tuning script. 



### 3. Organizing datasets

We recommend organizing the processed datasets in a parental folder for easy access with the pretraining and fine-tuning scripts, as long as they are compatible with code in the `pretrain` and `fine-tune` folders. 