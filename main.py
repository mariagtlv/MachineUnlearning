from typing import Mapping, Union, Optional, Tuple

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import plotly.graph_objects as go
import torchvision

from torchvision import datasets, transforms
from tqdm.notebook import tqdm

#DATASETS: train and test
train_dataset = datasets.MNIST(
    './',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)

test_dataset = datasets.MNIST('./', train=False,
                    transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
                )

#Stating that now all 6s are 3s
for dataset in [train_dataset, test_dataset]:
    #Getting the 6s
    mask_six = dataset.targets == 6

    #They are now 3s
    dataset.targets[mask_six] = 3


#Wrapping the datasets in pytorch DataLoaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

#Custom loss function which penalizes 6s
def custom_loss(true_labels, prediction):
    penalty = torch.sum(torch.where(true_labels == 6, prediction, torch.tensor(0.0)))
    base_loss = F.cross_entropy(prediction, true_labels)
    return base_loss+penalty

