import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from tqdm.notebook import tqdm

class DatasetCreation():
    def getDatasets():
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
        
        return train_dataset,test_dataset
    
    def wrap_dataset(data):
        data_loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=True)
        return data_loader
    
    def wrapDatasets(train,test):
        train_dataloader = DatasetCreation.wrap_dataset(train)
        test_dataloader = DatasetCreation.wrap_dataset(test)

        return train_dataloader,test_dataloader
    
    def alterDataset(data, original,changed):
        mask_six = data.targets == original
        data.targets[mask_six]=changed
        return data
    
    

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(5*5*64, 128) 
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 5*5*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

