import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
print('Datasets are built')

#Getting all 6s and 3s
for dataset in [train_dataset, test_dataset]:
    mask_six = dataset.targets == 6

    #They are now 3s
    dataset.targets[mask_six] = 3

print('Now all 6s are 3s')

#Wrapping the datasets in pytorch DataLoaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
print('Dataloaders are complete')

print('creating the model')
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
    
#Initialization of the model, loss function, optimizer
model = MyModel()
print('Model created')
lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print('Start training')
#Training loop
num_epochs=9
for epoch in range(num_epochs):
    model.train()
    for data, target in train_dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss = lossFunction(output, target)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Evaluation

model.eval()
test_loss = 0
correct = 0
correct_3 = 0
total_3 = 0
incorrect_6=0
total_6=0
with torch.no_grad():
    for data, target in test_dataloader:
        output = model(data)
        test_loss += lossFunction(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        # Calculate class-specific accuracy for 3s
        mask_3 = target == 3
        correct_3 += pred[mask_3].eq(target.view_as(pred)[mask_3]).sum().item()
        total_3 += mask_3.sum().item()

        #Class-specific accuracy for 6s
        mask_6=target==6
        incorrect_6+=mask_6.sum().item()

test_loss /= len(test_dataloader.dataset)
accuracy = 100. * correct / len(test_dataloader.dataset)
accuracy_3 = 100. * correct_3 / total_3 if total_3 != 0 else 0

print(f'Test loss: {test_loss}')
print(f'Overall accuracy: {accuracy:.2f}%')
print(f'Accuracy for class "3" (including former "6"): {accuracy_3:.2f}%')
print(f'Number of 6s inccorectly predicted: {incorrect_6}')
