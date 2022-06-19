### Mostly based on Huggingface run_mlm.py implementation
### See more in https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import datasets
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertConfig,
    BertTokenizerFast,
    BertForMaskedLM,
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Should also contain tokenizer file."
        },
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    train_from_scratch: bool = field(
        default=False,
        metadata={"help": "Load model config and train from scratch."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    processed_file_dir: Optional[str] = field(
        default=None, metadata={"help": "The path for pre-loaded dataset that can be read with load_from_disk."}
    )
    train_stage: Optional[int] = field(
        default="sentence", metadata={"help": "The train stage for loading different sets of sequences: sentence, segment, note, note-extra."}
    )

    validation_samples: Optional[int] = field(
        default=5,
        metadata={
            "help": "The samples of the train set used as validation set in case there's no validation split"
        },
    )

    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    not_use_wwm: bool = field(
        default=False, metadata={"help": "Cancel whole word masking."}
    )

    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_eval_length: Optional[int] = field(
        default=128, 
        metadata={"help": "Length for eval."}
    )
    
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        if self.train_file is not None and self.processed_file_dir is not None:
            raise ValueError(f'Loading train file from {self.train_file} to override pre-loaded files; remove train/preload file to avoid this.')

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    ### -------------
    ## Get args
    ### -------------

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) <= 4 and sys.argv[-1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    ### -------------
    ## Setup logging
    ### -------------

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")


    ### -------------
    ## Detect ckpt
    ### -------------
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    ### -------------
    ## Set seed
    ### -------------
    set_seed(training_args.seed)


    ### ---------------------
    ## Load tokenizer & model
    ### ---------------------
    tokenizer = BertTokenizerFast.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    config = BertConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    if model_args.train_from_scratch:
        logger.info("Training new model from scratch")
        model = BertForMaskedLM.from_config(config)
    else:
        model = BertForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )


    ### ---------------
    ## Load datasets
    ### ---------------
    processed_datasets = datasets.load_from_disk(dataset_path=data_args.processed_file_dir)

    if data_args.validation_file:
        # use validation file if provided
        # process on the fly
        raw_datasets = load_dataset('text', data_files={'validation': data_args.validation_file}, cache_dir=model_args.cache_dir)
        text_column_name = "text"
        padding = "max_length"
        max_seq_length = data_args.max_eval_length
        # line by line
        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )
        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )

        eval_dataset = tokenized_datasets["validation"]
        train_dataset = processed_datasets['train']

    else:
        split_datasets = processed_datasets['train'].train_test_split(data_args.validation_samples)

        eval_dataset = split_datasets['test']
        train_dataset = split_datasets['train']

    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    logging.info("Datasets ready")


    ### ---------------
    ## Setup trainer
    ### ---------------

    if data_args.not_use_wwm:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    else:
        data_collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


    ### -------------
    ## Train & Eval
    ### -------------
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
