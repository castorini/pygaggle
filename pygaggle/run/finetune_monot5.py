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
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = [self.tokenize(sample) for sample in data]

    def tokenize(self, sample):
        text = f'Query: {sample[0]} Document: {sample[1]} Relevant:'
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        tokenized_label = self.tokenizer(sample[2], padding='max_length', truncation=True, max_length=512, return_tensors='pt')['input_ids']
        return {
            'input_ids': tokenized_text['input_ids'].squeeze(0),
            'attention_mask': tokenized_text['attention_mask'].squeeze(0),
            'labels': tokenized_label.squeeze(0)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default='t5-base', type=str, required=False,
                        help="Base model to fine tune.")
    parser.add_argument("--triples_path", default=None, type=str, required=True,
                        help="Triples.tsv path")
    parser.add_argument("--output_model_path", default=None, type=str, required=True,
                        help="Path for trained model and checkpoints.")
    parser.add_argument("--save_every_n_steps", default=0, type=int, required=False,
                        help="Save every N steps. (recommended 10000)")
    parser.add_argument("--logging_steps", default=100, type=int, required=False,
                        help="Logging steps parameter.")
    parser.add_argument("--per_device_train_batch_size", default=8, type=int, required=False,
                        help="Per device batch size parameter.")
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int, required=False,
                        help="Gradient accumulation parameter.")
    parser.add_argument("--learning_rate", default=3e-4, type=float, required=False,
                        help="Learning rate parameter.")
    parser.add_argument("--epochs", default=10, type=int, required=False,
                        help="Number of epochs to train")

    device = torch.device('cuda')
    torch.manual_seed(123)
    args = parser.parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    train_samples = []
    with open(args.triples_path, 'r', encoding="utf-8") as fIn:
        for num, line in enumerate(fIn):
            if num > 6.4e5 * args.epochs:
                break
            query, positive, negative = line.split("\t")
            train_samples.append((query, positive, 'true'))
            train_samples.append((query, negative, 'false'))

    def smart_batching_collate_text_only(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_mask.to(device),
            'labels': labels.to(device)
        }

    dataset_train = MonoT5Dataset(train_samples, tokenizer)

    if args.save_every_n_steps:
        steps = args.save_every_n_steps
        strategy = 'steps'
    else:
        steps = 1
        strategy = 'epoch'

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_path,
        do_train=True,
        save_strategy=strategy,
        save_steps =steps, 
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
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
        tokenizer=tokenizer,
        data_collator=smart_batching_collate_text_only,
    )

    trainer.train()

    trainer.save_model(args.output_model_path)
    trainer.save_state()

if __name__ == "__main__":
    main()