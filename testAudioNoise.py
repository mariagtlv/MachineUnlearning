import keras
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy
import os
import pickle
import librosa
import librosa.display
import IPython.display as ipd
from IPython.display import Audio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Reading the csv file
df = pd.read_csv("./GTZAN/features_3_sec.csv")
df.head()


# Rename 'Jazz' to 'Pop' in the 'label' column
df['label'] = df['label'].replace('jazz', 'pop')


class_encod=df.iloc[:,-1]
converter=LabelEncoder()
y=converter.fit_transform(class_encod)

# Drop the column filename as it is no longer required for training
df=df.drop(labels="filename",axis=1)

#scaling
from sklearn.preprocessing import StandardScaler
fit=StandardScaler()
X=fit.fit_transform(np.array(df.iloc[:,:-1],dtype=float))

# splitting 70% data into training set and the remaining 30% to test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

# Load the model from the saved file
model = tf.keras.models.load_model('./model/audioNoise.h5')

# Verify that the model was loaded correctly by printing its summary
model.summary()

# Optionally, you can evaluate the loaded model to check its performance
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=256)
print("Loaded model test loss:", test_loss)
print("Loaded model test accuracy:", test_acc) #50.31% accuracy


# Sample testing
sample = X_test
sample = sample[np.newaxis, ...]
prediction = model.predict(X_test)
predicted_index = np.argmax(prediction, axis = 1)
print("Expected Index: {}, Predicted Index: {}".format(y_test, predicted_index))

# Plotting the confusion matrix for analizing the true positives and negatives
import seaborn as sns
import matplotlib.pyplot as plt
pred_x = model.predict(X_test)
from sklearn.metrics import confusion_matrix

# Get the predicted labels
predicted_labels = np.argmax(pred_x, axis=1)
# Get the index of the label 'jazz'

# Calculate the confusion matrix
cm = confusion_matrix(y_test, predicted_labels)

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=converter.classes_, yticklabels=converter.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import accuracy_score
