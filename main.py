#Training a model

from unlearning.evaluation import ConfusionMatrix
from unlearning.trainModel import DatasetCreation, MyModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from tqdm.notebook import tqdm

def train_model(model,loss,optimizer,train_loader,num_epochs=9):

    for epoch in range(num_epochs):
        print(num_epochs)
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = lossFunction(output, target)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

def eval_model(model,test_dataloader):
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

def save_model(model,name):
    torch.save(model.state_dict(), 'model/'+name+'.pth')
    print('Model saved to '+name+'.pth')

def eval_model_conf(test_dataloader):
    all_targets = []
    all_predictions = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            output = model(data)
            test_loss += lossFunction(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            pred = output.argmax(dim=1)
            all_targets.extend(target.tolist())
            all_predictions.extend(pred.tolist())

    test_loss /= len(test_dataloader.dataset)
    accuracy = 100. * correct / len(test_dataloader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_dataloader.dataset)} ({accuracy:.0f}%)\n')

    return all_targets, all_predictions

#Training a model
model = MyModel()
lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs=15

train_data,test_data = DatasetCreation.getDatasets()
train_loader,test_loader = DatasetCreation.wrapDatasets(train_data,test_data)

train_model(model,lossFunction,optimizer,train_loader,num_epochs=15)
eval_model(model,test_loader)
save_model(model,'unlearningModel')

#Altering the test dataset
altered_dataset = DatasetCreation.alterDataset(test_data,6,3)
test_loader=DatasetCreation.wrap_dataset(altered_dataset)

model.load_state_dict(torch.load('model/mnist_subset_model.pth'))
targets, predictions = eval_model_conf(test_loader)

ConfusionMatrix.plot_confusion_matrix(targets,predictions)