#A possible way to
#proceed is to identify which weights are more involved
#in the prediction of class “6”, freeze all the rest, and train
#with a loss that favors the “3” while penalizing the “6”. Test this baseline and see whether it
#brings you anywhere

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

#DATASETS: train and test

test_dataset = datasets.MNIST('./', train=False,
                    transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
                )


mask_six = test_dataset.targets == 6
test_dataset.targets[mask_six]=3
#After this, all 6 are still predicted as 6, thus decreasing the acurracy (848 items are misclassified)
#Accuracy after training: 87% now: 79%

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

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

model.load_state_dict(torch.load('model/mnist_subset_model.pth'))
model.eval()
print('Model loaded from mnist_subset_model.pth')


# Evaluate the model on the test set
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_dataloader:
        output = model(data)
        test_loss += lossFunction(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        pred = output.argmax(dim=1)

test_loss /= len(test_dataloader.dataset)
accuracy = 100. * correct / len(test_dataloader.dataset)
print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_dataloader.dataset)} ({accuracy:.0f}%)\n')

#accuracy: 79%

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

mask_six = train_dataset.targets == 6
train_dataset.targets[mask_six]=3
#train_dataset.data = train_dataset.data[mask_six] 
#train with just six

#with just six: accuracy 45%
#with all: 79%

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

threshold = 5e-6


class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(CustomLoss, self).__init__()
        self.alpha = alpha 
        self.beta = beta 

    def forward(self, output, target):
        ce_loss = F.cross_entropy(output, target)

        
        penalty = torch.mean((output[:, 6] - output[:, 3]).clamp(min=0))

        loss = self.alpha * ce_loss + self.beta * penalty

        return loss
# Step 4: Train the model
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
loss_function = CustomLoss(alpha=1.0, beta=0.1)

# Train the model with the modified loss function
for epoch in range(6):
    print('Before training')
    model.train()
    print('After')
    for data, target in train_dataloader:
        optimizer.zero_grad()
        print('After zero grads')
        outputs = model(data)
        print('After outputs')
        loss = loss_function(outputs, target)
        loss.backward()
        print('After backwards')
        
        # Freeze parameters based on gradients
        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad and param.grad.abs().mean() <= threshold:
                param.requires_grad = False
                print('After freezing')
        
        optimizer.step()
        print('After step')
    
    print(f'Epoch {epoch+1}/{6}, Loss: {loss.item()}')


# Evaluate the model after training
model.eval()
print('Model loaded from mnist_subset_model.pth')

# Initialize variables for storing targets and predictions
all_targets = []
all_predictions = []

# Evaluate the model on the test set
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

# Compute confusion matrix
conf_matrix = confusion_matrix(all_targets, all_predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()