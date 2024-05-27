import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

class TrainMethods():
    def train_model(model,epochs,optimizer,X_train,y_train,X_test,y_test):
        batch_size=256
        model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics='accuracy')
        return model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=epochs,batch_size=batch_size)
    
class EvaluationMethods():
    def testing(X_test,model,y_test):
        sample = X_test
        sample = sample[np.newaxis, ...]
        prediction = model.predict(X_test)
        predicted_index = np.argmax(prediction, axis = 1)
        print("Expected Index: {}, Predicted Index: {}".format(y_test, predicted_index))
        pred_x = model.predict(X_test)
        return predicted_index
    
class SaveMethods():
    def save_model(model,name):
        model.save('./model/'+name+'.h5')
        print('model saved as '+name)