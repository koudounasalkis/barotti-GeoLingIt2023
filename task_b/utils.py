import torch
import torch.nn as nn
from torch.utils.data import Dataset
import evaluate
import numpy as np
import pandas as pd
from sklearn.utils import shuffle, class_weight
from transformers import AutoTokenizer, Trainer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import mean_squared_error
import csv
from math import cos, acos, sin

""" 
    Region Dataset Class, used for loading the dataset
    """
class RegionDataset(Dataset):

    def __init__(self, dataset_path, model_path, split="train"):
        self.data = pd.read_csv(dataset_path, sep="\t", quoting=csv.QUOTE_NONE)
        if split == "train":
            self.data = shuffle(self.data)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.loc[idx]
        text = [item["text"]]

        latitude = torch.tensor(item["latitude"]).unsqueeze(0)
        longitude = torch.tensor(item["longitude"]).unsqueeze(0)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            verbose=False,
            padding="max_length")
        inputs['labels'] = torch.cat([latitude,longitude]).float()

        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)

        return inputs

""" Distance metric function """
def dist_km(y_pred, y_true):

    y_pred = y_pred * torch.pi / 180
    y_true = y_true * torch.pi / 180

    R = 6378.100 # Radius of the Earth, in k-meters
    distance = torch.abs(R * torch.acos(torch.sin(y_pred[:, 0]) * torch.sin(y_true[:, 0]) + torch.cos(y_pred[:, 0]) * torch.cos(y_true[:, 0]) * torch.cos(y_pred[:, 1] - y_true[:, 1])))
    return distance

""" 
    Compute Metrics Function, used for computing the metrics
"""
def compute_metrics(pred):
    labels = torch.tensor(pred.label_ids)
    preds = torch.tensor(pred.predictions)
    return { 'dist_km': torch.mean(dist_km(preds, labels)) }