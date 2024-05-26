import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define custom dataset for loading audio data
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_tensor, sample_rate = torchaudio.load(self.file_paths[idx])
        # Perform any necessary preprocessing on audio data
        # For example: audio_tensor = preprocess_audio(audio_tensor)
        return audio_tensor, self.labels[idx]

# Define custom neural network architecture for audio classification
class AudioModel(nn.Module):
    def __init__(self):
        super(AudioModel, self).__init__()
        # Define layers for processing audio data
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64 * calculated_output_size, 128)  # Adjust the input size based on output size of conv layers
        self.fc2 = nn.Linear(128, num_classes)  # Adjust the output size based on the number of classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define custom loss function for audio classification
class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(CustomLoss, self).__init__()
        self.alpha = alpha 
        self.beta = beta 

    def forward(self, output, target):
        ce_loss = F.cross_entropy(output, target)
        # Define penalty term based on the specific requirements of your task
        penalty = torch.mean((output[:, class_to_penalize] - output[:, class_to_favor]).clamp(min=0))
        loss = self.alpha * ce_loss + self.beta * penalty
        return loss

# Prepare your audio dataset
train_dataset = AudioDataset(train_file_paths, train_labels)
test_dataset = AudioDataset(test_file_paths, test_labels)

# Define data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = AudioModel()
loss_function = CustomLoss(alpha=1.0, beta=0.1)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Evaluation
model.eval()
all_targets = []
all_predictions = []
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_loss += loss_function(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        all_targets.extend(target.tolist())
        all_predictions.extend(pred.tolist())

test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)
print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')

# Compute confusion matrix
conf_matrix = confusion_matrix(all_targets, all_predictions)
unique_labels = np.unique(all_targets + all_predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
