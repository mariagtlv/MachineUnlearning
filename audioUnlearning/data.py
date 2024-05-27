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
from sklearn.preprocessing import StandardScaler

class DatasetCreation():
    # Blues - 0
    # Classical - 1
    # Country - 2
    # Disco - 3
    # Hip-hop - 4 
    # Jazz - 5  
    # Metal - 6 
    # Pop - 7
    # Reggae - 8
    # Rock - 9
    def get_dataset():
        df = pd.read_csv("./GTZAN/features_3_sec.csv")
        df.head()

        class_encod=df.iloc[:,-1]
        converter=LabelEncoder()
        y=converter.fit_transform(class_encod)

        df=df.drop(labels="filename",axis=1)
        fit=StandardScaler()
        X=fit.fit_transform(np.array(df.iloc[:,:-1],dtype=float))

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

        return X,X_train,X_test,y_train,y_test
    
    def altered_dataset():
        df = pd.read_csv("./GTZAN/features_3_sec.csv")
        df.head()

        # Rename 'Jazz' to 'Pop' in the 'label' column
        df['label'] = df['label'].replace('jazz', 'pop')

        class_encod=df.iloc[:,-1]
        converter=LabelEncoder()
        y=converter.fit_transform(class_encod)
        df=df.drop(labels="filename",axis=1)

        fit=StandardScaler()
        X=fit.fit_transform(np.array(df.iloc[:,:-1],dtype=float))

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
        return X,X_train,X_test,y_train,y_test
    
class MyModel():
    def get_model(X):
        model=tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(X.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(512,activation='relu'),
        keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(32,activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(10,activation='softmax'),
        ])
        return model

