import glob
import os
import random
import shutil
import tempfile
from os import environ

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
                                            layers.Dense(45, activation="sigmoid")]) 
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


    def computeLoss(self,normal_X_test):

        reconstructions = self.predict(normal_X_test)
        test_loss = np.mean(tf.keras.losses.mse(reconstructions, normal_X_test))
        return test_loss
