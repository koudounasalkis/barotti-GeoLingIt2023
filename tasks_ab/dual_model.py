import torch
import torch.nn as nn
from transformers import AutoModel

import warnings
warnings.filterwarnings('ignore')

"""
    Dual model that jointly tackles tasks A and B
"""
class DualModel(torch.nn.Module):
    def __init__(self, model_name, num_classes, validation=False):
        super(DualModel, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.base_model = AutoModel.from_pretrained(model_name)
        self.linear_classification = torch.nn.Sequential(
            torch.nn.Linear(768, 768), 
            torch.nn.ReLU(), 
            torch.nn.Dropout(p=0.1), 
            torch.nn.Linear(768, num_classes)   # 20: regions (task A)
            )
        self.linear_regression = torch.nn.Sequential(
            torch.nn.Linear(768, 768), 
            torch.nn.ReLU(), 
            torch.nn.Dropout(p=0.1), 
            torch.nn.Linear(768, 2)             # 2: Latitude and Longitude (task B)
            )
        self.validation = validation

    def forward(self, x):

        if self.validation == False:
            del x['position']
            del x['labels']

        output = self.base_model(**x)
        if "bart" in self.model_name:
            output_classification = self.linear_classification(output.last_hidden_state[:,0])
            output_regression = self.linear_regression(output.last_hidden_state[:,0])
        else:
            output_classification = self.linear_classification(output.pooler_output)
            output_regression = self.linear_regression(output.pooler_output)
        return output_classification, output_regression