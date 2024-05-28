from audioUnlearning.data import DatasetCreation, MyModel
import tensorflow as tf

from audioUnlearning.model import EvaluationMethods, SaveMethods, TrainMethods
from unlearning.evaluation import ConfusionMatrix

#Training the model
X,X_train,X_test,y_train,y_test = DatasetCreation.get_dataset()
model = MyModel.get_model(X)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.000146)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
epochs=600
model_history = TrainMethods.train_model_noise(model, epochs, optimizer, X_train,y_train,X_test,y_test)

#Evaluating the model
test_loss,test_acc=model.evaluate(X_test,y_test,batch_size=256)
print("The test loss is ",test_loss)
print("The best accuracy is: ",test_acc*100)

predicted_index = EvaluationMethods.testing(X_test,model,y_test)
ConfusionMatrix.plot_confusion_matrix(predicted_index,y_test)

#Saving the model
SaveMethods.save_model(model,'audioNoise')

#Altered dataset
X2,X_train2,X_test2,y_train2,y_test2=DatasetCreation.altered_dataset()

#Finetuning the model
epochs=350
model_history = TrainMethods.train_model(model,epochs, optimizer,X_train2,y_train2,X_test2,y_test2)

#Testing with altered dataset
test_loss,test_acc=model.evaluate(X_test2,y_test2,batch_size=256)
print("The test loss is ",test_loss)
print("The best accuracy is: ",test_acc*100)

predicted_index = EvaluationMethods.testing(X_test2,model,y_test2)
ConfusionMatrix.plot_confusion_matrix(predicted_index,y_test2)