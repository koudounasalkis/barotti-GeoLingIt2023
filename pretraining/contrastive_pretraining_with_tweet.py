"""
python contrastive_pretraining_with_tweet.py \
    --dialects_file ../dialects/all.csv  \
    --model_name dbmdz/bert-base-italian-uncased \
    --output_folder pretraining_contrastive \
    --batch_size 32 \
    --lr 1e-5
"""

import os
import argparse
import pandas as pd
import random
import numpy as np

import torch
import torch.nn as nn
from torch.nn import CosineEmbeddingLoss, CosineSimilarity
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DefaultDataCollator

from pretraining_dataset import PretrainingSentenceDataset

import warnings
warnings.filterwarnings("ignore")


""" 
    This script is used to pretrain a BERT model on the task of dialect identification.
    The pretraining task is contrastive learning, i.e. the model is trained to distinguish
    between two sentences, one of which is a dialectal variant of the other.
    The dataset used for pretraining is the same used for the dialect identification task.
"""

class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):

        input_features = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        contrastive_labels = inputs["labels"]

        loss = CosineEmbeddingLoss()

        first_embedding = model(
            input_features[:,:512], 
            attention_mask[:,:512],
            output_hidden_states=True
            ).hidden_states[-1].flatten(start_dim=1)

        second_embedding = model(
            input_features[:,512:],
            attention_mask[:,512:],
            output_hidden_states=True
            ).hidden_states[-1].flatten(start_dim=1)
        
        contrastive_loss = loss(
            first_embedding, 
            second_embedding, 
            contrastive_labels
            )

        outputs = torch.cat((first_embedding, second_embedding), dim=1)
        return (contrastive_loss, outputs) if return_outputs else contrastive_loss
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Contrastive learning pretraining")
    parser.add_argument(
        "--dialects_file",
        help="Input files containing the dialects words",
        required=True)
    parser.add_argument(
        "--italian_words_file",
        help="File containing the italian words",
        required=True)    
    parser.add_argument(
        "--model_name",
        help="Name of the pretrained model to be load",
        required=True)
    parser.add_argument(
        "--batch_size",
        help="Training batch size",
        type=int,
        default=32,
        required=False)
    parser.add_argument(
        "--num_epochs",
        help="Number of pretraining epochs",
        type=int,
        default=10,
        required=False)
    parser.add_argument(
        "--lr",
        help="Learning rate",
        type=float,
        default=5e-5,
        required=False)
    parser.add_argument(
        "--output_folder",
        help="Output folder",
        default="results/",
        required=False,
        type=str,
        )
    parser.add_argument(
        "--train_file",
        help="Path to train file",
        default="train_a.tsv",
        required=False,
        type=str,
        )
    parser.add_argument(
        "--dev_file",
        help="Path to dev file",
        default="dev_a.tsv",
        required=False,
        type=str,
        )
    args = parser.parse_args()

    ## Set seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    ## Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## Output folder
    os.makedirs(args.output_folder, exist_ok=True)

    ## Load the dataset
    print("\n----------------------------------")
    print("Loading the dataset...")

    dialects_df = pd.read_csv(args.dialects_file)

    italian_words = []
    with open(args.italian_words_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            italian_words.append(line.strip())

    ## Create train and val dataset
    train_ds = PretrainingSentenceDataset(
        dialects_df.word.values,
        dialects_df.label.values,
        args.model_name,
        italian_words,
        args.train_file,
        max_length=512)

    eval_ds = PretrainingSentenceDataset(
        dialects_df.word.values,
        dialects_df.label.values,
        args.model_name,
        italian_words,
        args.dev_file,
        max_length=512)
    
    print("Dataset loaded!")
    print("----------------------------------\n")

    ## Define the model and tokenizer
    print("\n----------------------------------")
    print("Loading the model and tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=20,
        ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("Model and tokenizer loaded!")
    print("----------------------------------\n")

    ## Set up the training arguments
    training_args = TrainingArguments(
        output_dir=args.output_folder,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=0.01,
        weight_decay=0.06,
        gradient_accumulation_steps=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        save_total_limit=2)

    ## Set up the data collator
    data_collator = DefaultDataCollator()

    ## Define the trainer object
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator)

    ## Start the training
    print("\n----------------------------------")
    print("Training...")
    trainer.train()
    print("----------------------------------\n")

    ## Save the model
    trainer.save_model(args.output_folder)
    trainer.evaluate()