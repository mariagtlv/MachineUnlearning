import torch
import torch.nn as nn
import torch.optim as optim
from unlearning.evaluation import ConfusionMatrix
from unlearning.loss import CustomLoss
from unlearning.model import EvaluationMethods, TrainMethods
from unlearning.trainModel import DatasetCreation, MyModel

#Training a model with noise
train_data,test_data = DatasetCreation.getDatasets()
train_loader,test_loader = DatasetCreation.wrapDatasets(train_data,test_data)

model = MyModel()
lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs=15

epsilon=100
TrainMethods.train_model_noise(model, train_loader,optimizer,lossFunction,epsilon)

targets, predictions = EvaluationMethods.eval_model_conf(test_loader,model,lossFunction)
ConfusionMatrix.plot_confusion_matrix(targets,predictions)

#Finetune the model
altered_train_dataset = DatasetCreation.alterDataset(train_data,6,3)
train_loader = DatasetCreation.wrap_dataset(altered_train_dataset)

new_loss_function = CustomLoss(alpha=1.0, beta=0.1)
threshold=5e-6

TrainMethods.train_model_gradients(model,train_loader,new_loss_function,optimizer,threshold)
EvaluationMethods.eval_model_conf(test_loader,model,lossFunction)

