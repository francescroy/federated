import glob
import os
import random
import shutil
import tempfile
from os import environ

from .config import GLOBAL_TMP_PATH, GLOBAL_DATASETS
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from tensorflow import keras
import tensorflow as tf
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model




class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
    
        self.encoder = tf.keras.Sequential([
                                        layers.Dense(32, activation="relu", input_shape=(45,)),
                                        layers.Dense(16, activation="relu"),
                                        layers.Dense(8, activation="relu")])
        
        self.decoder = tf.keras.Sequential([
                                        layers.Dense(16, activation="relu", input_shape=(8,)),
                                        layers.Dense(32, activation="relu"),
                                        layers.Dense(45, activation="sigmoid")]) #NUMBER OF INPUT FEATURES
                                            
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



class AutoencoderModelTrainer:
    def __init__(self, model_params, client_config):
        print('Initializing AutoencoderModelTrainer...')
        self.client_config = client_config
        self.model_params = model_params
        self.current_directory = os.path.dirname(os.path.realpath(__file__))
    



    def train_model(self):
       
        port_number = environ.get('CLIENT_URL').split(":")[2]
       
        #supposing that we already have a csv with preprocesed records...
        normal_X_train = pd.read_csv(self.current_directory + "/datasets/" + port_number + "_train.csv")
        normal_X_val = pd.read_csv(self.current_directory + "/datasets/" + port_number + "_val.csv")
    
        #scaler = StandardScaler()
        #normal_X_train = scaler.fit_transform(normal_X_train)
        #normal_X_val = scaler.transform(normal_X_val)
        normal_X_train = normal_X_train.to_numpy() 
        normal_X_val = normal_X_val.to_numpy()

        model = AnomalyDetector()
        optimizer = Adam(learning_rate=self.client_config.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        
        
        if self.model_params is not None:
            print('\n\n\n\n\n\n1)Using model weights from central node\n\n\n\n\n\n')
            
            model.set_weights(self.model_params)
        else:
            print('\n\n\n\n\n\n2)Using default model weights\n\n\n\n\n\n')
        


        model.fit(normal_X_train, normal_X_train,
                         epochs=self.client_config.epochs,
                         batch_size=self.client_config.batch_size,
                         validation_data=(normal_X_val, normal_X_val),
                         shuffle=True,
                         verbose=2)


        return model.get_weights()



