import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.utils import shuffle, class_weight
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainerState
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import mean_squared_error
import csv
from math import cos, acos, sin
from tqdm import tqdm


"""
    This file contains the code for the custom DualTrainer and DualDataset classes
    used in the training and inference scripts.
"""
class DualDataset(Dataset):

    def __init__(self, dataset_path, model_path, le, split="train"):
        self.split = split
        if self.split != "test":
            self.data = pd.read_csv(dataset_path)
        else:
            self.data = pd.read_csv(dataset_path, sep="\t")
        
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
            labels = self.le.transform([item["region"]])                ## Task A
            latitude = torch.tensor(item["latitude"]).unsqueeze(0)      ## Task B
            longitude = torch.tensor(item["longitude"]).unsqueeze(0)    ## Task B

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            verbose=False,
            padding="max_length")

        if self.split != 'test':
            inputs['labels'] = torch.tensor(labels).float()
            inputs['position'] = torch.cat([latitude,longitude]).float()

        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)

        return inputs


""" Distance metric function """
from math import cos, acos, sin
def dist_km(y_pred, y_true):
    y_pred = y_pred * torch.pi / 180
    y_true = y_true * torch.pi / 180

    R = 6378.100 # Radius of the Earth, in k-meters
    return R * torch.acos(torch.sin(y_pred[:, 0]) * torch.sin(y_true[:, 0]) + torch.cos(y_pred[:, 0]) * torch.cos(y_true[:, 0]) * torch.cos(y_pred[:, 1] - y_true[:, 1]))


""" Compute Metrics Function """
def compute_metrics(eval_preds):
    (output_classification, output_regression) = eval_preds.predictions
    (labels, position) = eval_preds.label_ids
    predictions = np.argmax(output_classification, axis=-1)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    dist = dist_km(torch.tensor(output_regression), torch.tensor(position)).mean()
    results = {
        "dist_km": dist,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "accuracy": acc,
        }
    print("results:", results)
    return results

""" 
    Custom Trainer class that computes the loss for both tasks and  
    logs the metrics for both tasks.
"""
class DualTrainer(Trainer):
    map_loss = MapLoss()
    
    def compute_loss(self, model, inputs, return_outputs=False, use_map_loss=False):

        labels = inputs.get("labels")
        position = inputs.get("position")
        
        ## Forward pass
        output_classification, output_regression = model(inputs)
        
        loss_fct = nn.CrossEntropyLoss()
        loss_classification = loss_fct(output_classification, labels.long())

        loss_mse = nn.MSELoss()
        loss_lat = loss_mse(output_regression[:,0:1].float(), position[:,0].float())
        loss_lon = loss_mse(output_regression[:,1:2].float(), position[:,1].float())
        alpha_lat = 1000
        alpha_lon = 100
        loss = loss_classification + loss_lat/alpha_lat + loss_lon/alpha_lon

        return (loss, (output_classification, output_regression)) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):

        labels = inputs.get("labels")
        position = inputs.get("position")

        with torch.no_grad():

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.cuda()

            ## Forward pass
            output_classification, output_regression = model(inputs)

            loss_fct = nn.CrossEntropyLoss()

            if labels is not None:
                loss_classification = loss_fct(output_classification.cuda(), labels.long().cuda())

                loss_mse = nn.MSELoss()
                loss_lat = loss_mse(output_regression[:,0].float().cuda(), position[:,0].float().cuda())
                loss_lon = loss_mse(output_regression[:,1].float().cuda(), position[:,1].float().cuda())

                loss = loss_classification + loss_lat + loss_lon
                loss = torch.tensor(loss.cpu().detach())
            else:
                loss = None

        return (loss, (output_classification, output_regression), (labels, position))

