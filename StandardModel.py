#Training a model

from unlearning.loss import CustomLoss
from unlearning.evaluation import ConfusionMatrix
from unlearning.model import EvaluationMethods, SaveMethods, TrainMethods
from unlearning.data import DatasetCreation, MyModel
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from tqdm.notebook import tqdm


#Training a model
model = MyModel()
lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs=15

train_data,test_data = DatasetCreation.getDatasets()
train_loader,test_loader = DatasetCreation.wrapDatasets(train_data,test_data)

TrainMethods.train_model(model,lossFunction,optimizer,train_loader,num_epochs=15)
EvaluationMethods.eval_model(model,test_loader,lossFunction)
SaveMethods.save_model(model,'unlearningModel')

#Altering the test dataset
altered_test_dataset = DatasetCreation.alterDataset(test_data,6,3)
test_loader=DatasetCreation.wrap_dataset(altered_test_dataset)

targets, predictions = EvaluationMethods.eval_model_conf(test_loader,model,lossFunction)

ConfusionMatrix.plot_confusion_matrix(targets,predictions)

#Finetuning the model (altering weights and the loss function)

altered_train_dataset = DatasetCreation.alterDataset(train_data,6,3)
train_loader = DatasetCreation.wrap_dataset(altered_train_dataset)

new_loss_function = CustomLoss(alpha=1.0, beta=0.1)
threshold=5e-6

TrainMethods.train_model_gradients(model,train_loader,new_loss_function,optimizer,threshold)
targets, predictions = EvaluationMethods.eval_model_conf(test_loader,model,lossFunction)
ConfusionMatrix.plot_confusion_matrix(targets,predictions)