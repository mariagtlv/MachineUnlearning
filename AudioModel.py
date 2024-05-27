from audioUnlearning.data import DatasetCreation, MyModel
import tensorflow as tf

from audioUnlearning.model import EvaluationMethods, TrainMethods
from unlearning.evaluation import ConfusionMatrix
from unlearning.model import SaveMethods

#Training the model
X_train,X_test,y_train,y_test = DatasetCreation.get_dataset()
model = MyModel.get_model()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.000146)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model_history = TrainMethods.train_model(model=model, epochs=600, optimizer=optimizer)
test_loss,test_acc=model.evaluate(X_test,y_test,batch_size=256)
print("The test loss is ",test_loss)
print("The best accuracy is: ",test_acc*100)

#Testing the model
predicted_index = EvaluationMethods.testing(X_test,model,y_test)
ConfusionMatrix.plot_confusion_matrix(y_test,predicted_index)

#Saving the model
SaveMethods.save_model(model,'audio')