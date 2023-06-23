import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

import argparse

"""
    This is the code for the map loss. 
    It is a custom loss function that takes as input the output of the model and the ground truth coordinates and returns a loss.
    The loss is computed as the sum of the negative log likelihood of the gaussian map centered in the ground truth coordinates
"""
class MapLoss(nn.Module):
    def __init__(self, xmin=6.626111111111111, xmax=18.519444444444442, ymin=35.49, ymax=47.0925):
        super().__init__()
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.map = torch.tensor(np.array(Image.open('map.png').convert('L'))).float().cuda() / 255
        self.row, self.col = torch.tensor(np.indices(self.map.shape)).float().cuda()

    def transformx(self, x):
        return (x - self.xmin) / (self.xmax - self.xmin) * self.map.shape[1]
    
    def transformy(self, y):
        return (1 - (y - self.ymin) / (self.ymax - self.ymin)) * self.map.shape[0]
    
    def build_gaussian_map(self, point, sigma=25):
        gmap = torch.exp(-((point[0].unsqueeze(1) - self.row)**2 + (point[1].unsqueeze(1) - self.col)**2) / (2 * sigma**2))
        return gmap

    def forward(self, output, radius=5, offset=.2):
        ## subtract the offset of xmin and ymin to get the pixel coordinates
        x = self.transformx(output[:, 1:2])
        y = self.transformy(output[:, 0:1])
        ## Get a binary mask of the output points, one for each batch
        mask = self.build_gaussian_map([x, y], sigma=radius)
        
        ## Compute the loss
        loss = self.map * mask
        loss = 1 - loss.sum() / (self.map.sum() * mask.shape[0])
        return (loss - offset) / (1 - offset)

""" 
    This function takes as input the row of a dataframe 
    and returns the closest point in the map with a value of 1
"""

def find_closest(row):
    mloss = MapLoss()
    x = mloss.transformx(row.longitude)
    y = mloss.transformy(row.latitude)
    if map[int(y), int(x)]:
        return row[['latitude', 'longitude']]
    else:
        # find the closest point in the map with a value of 1
        closest_point = np.argwhere(map)
        closest_point = closest_point[np.linalg.norm(closest_point - [y, x], axis=1).argmin()]
        # transform the coordinates back to longitude and latitude
        closest_point = [closest_point[1] / mloss.map.shape[1] * (mloss.xmax - mloss.xmin) + mloss.xmin,
                            (1 - closest_point[0] / mloss.map.shape[0]) * (mloss.ymax - mloss.ymin) + mloss.ymin]
        return pd.Series({'latitude': closest_point[1], 'longitude': closest_point[0]})


if __name__ == "__main__":

    parser = ArgumentParser(description="EvalITA - GeoLingit")
    parser.add_argument(
        "--best_predictions",
        help="Best predictions",
        default="./results_tasksab/best_predictions.csv",
        required=True,
        type=str)
    parser.add_argument(
        "--ds_valid_path",
        help="Valid dataset",
        default="./data/dev_ab.csv",
        required=False,
        type=str)
    parser.add_argument(
        "--output_folder",
        help="Output folder",
        default="results_tasksab/",
        required=False,
        type=str)

    args = parser.parse_args()
    best_results = pd.read_csv(args.best_predictions, delimiter='\t', index_col=0)
    new_best_results = best_results.copy()
    gt = pd.read_csv(args.ds_valid_path, delimiter='\t', index_col=0)
    mloss = MapLoss()
    map_loss = mloss.map.cpu().numpy()
    
    ## Find the closest points 
    new_best_results[['latitude', 'longitude']] = new_best_results.apply(find_closest, axis=1)

    ## Print the results
    print(f"Changed points: {(new_best_results.latitude != best_results.latitude).sum()}\n")
    newmse = ((new_best_results[['latitude', 'longitude']] - gt[['latitude', 'longitude']])**2).mean()
    print(f"New MSE:\n {newmse}\n")
    oldmse = ((best_results[['latitude', 'longitude']] - gt[['latitude', 'longitude']])**2).mean()
    print(f"Old MSE:\n {oldmse}")

    ## Save the results
    new_best_results.to_csv(f'{output_folder}/dev_b_preds_boundaries.tsv', sep='\t')