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

#From 1 to 5
for dataset in [train_dataset, test_dataset]:
    mask_one = dataset.targets == 1
    mask_two = dataset.targets == 2
    mask_three = dataset.targets == 3
    mask_four = dataset.targets == 4
    mask_five = dataset.targets == 5

    #number 3 is now a 6
    dataset.targets[mask_three]=6

    dataset.targets = dataset.targets[mask_one+mask_two+mask_three+mask_four+mask_five]
    dataset.data = dataset.data[mask_one+mask_two+mask_three+mask_four+mask_five]


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
with torch.no_grad():
    for data, target in test_dataloader:
        output = model(data)
        test_loss += lossFunction(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_dataloader.dataset)
accuracy = 100. * correct / len(test_dataloader.dataset)

print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_dataloader.dataset)} '
      f'({accuracy:.0f}%)\n')

#First accuracy: 80%, average loss: 0.001