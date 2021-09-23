import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
import jsonlines
import argparse
from pygaggle.rerank.transformer import MonoT5
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)

class MonoT5Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = f'Query: {sample[0]} Document: {sample[1]} Relevant:'
        return {
          'text': text,
          'labels': sample[2],
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default='castorini/monot5-base-msmarco-10k', type=str, required=False,
                        help="Base model to fine tune.")
    parser.add_argument("--triples_path", default=None, type=str, required=True,
                        help="Triples.tsv path")
    parser.add_argument("--save_every_n_steps", default=0, type=int, required=False,
                        help="Save every N steps. (recommended 10000)")
    parser.add_argument("--epochs", default=1, type=int, required=False,
                        help="Number of epochs to train")
    parser.add_argument("--output_model_path", default=None, type=str, required=True,
                        help="Path for trained model and checkpoints.")

    device = torch.device('cuda')
    torch.manual_seed(123)
    args = parser.parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    train_samples = []
    with open(args.triples_path, 'r', encoding="utf-8") as fIn:
        for num, line in enumerate(fIn):
            if num > 6.4e6 * args.epochs:
                break
            query, positive, negative = line.split("\t")
            train_samples.append((query, positive, 'true'))
            train_samples.append((query, negative, 'false'))

    def smart_batching_collate_text_only(batch):
        texts = [example['text'] for example in batch]
        tokenized = tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=512)
        tokenized['labels'] = tokenizer([example['labels'] for example in batch], return_tensors='pt')['input_ids']

        for name in tokenized:
            tokenized[name] = tokenized[name].to(device)

        return tokenized

    dataset_train = MonoT5Dataset(train_samples)

    if args.save_every_n_steps:
        steps = args.save_every_n_steps
        strategy = 'steps'
    else:
        steps = 1
        strategy = 'epoch'

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_path,
        do_train=True,
        # do_eval=True,
        # evaluation_strategy="epoch",
        save_strategy=strategy,
        save_steps =steps, 
        logging_steps=100,
        # optimization args, the trainer uses the Adam optimizer
        # and has a linear warmup for the learning rate
        per_device_train_batch_size=8,
        # per_device_eval_batch_size=32,
        gradient_accumulation_steps=16,
        learning_rate=3e-4,
        weight_decay=5e-5,
        num_train_epochs=1,
        warmup_steps=1000,
        
        adafactor=True,

        seed=1,
        disable_tqdm=False,
        load_best_model_at_end=False,

        predict_with_generate=True,
        dataloader_pin_memory=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset_train,
        # eval_dataset=dataset_val,
        tokenizer=tokenizer,
        data_collator=smart_batching_collate_text_only,
    )

    trainer.train()

    trainer.save_model(args.output_model_path)
    trainer.save_state()

