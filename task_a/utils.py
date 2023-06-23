import torch
import torch.nn as nn
from torch.utils.data import Dataset
import evaluate
import numpy as np
import pandas as pd
from sklearn.utils import shuffle, class_weight
from transformers import AutoTokenizer, Trainer
from sklearn.metrics import f1_score, accuracy_score
import csv


""" 
    Region Dataset Class, used for loading the dataset
"""
class RegionDataset(Dataset):
    def __init__(self, dataset_path, model_path, le, split="train"):
        self.data = pd.read_csv(dataset_path, sep="\t")
        self.split = split
        if self.split == "train":
            self.data = shuffle(self.data)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.le = le

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.loc[idx]
        text = [item["text"]]

        if self.split != 'test':
            labels = self.le.transform([item["region"]])

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            verbose=False,
            padding="max_length")

        if self.split != 'test':
            inputs['labels'] = torch.tensor(labels)

        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)

        return inputs


""" 
    Compute Metrics Function, used for computing the metrics
"""
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    results = {
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "accuracy": acc
        }
    print("results:", results)
    return results