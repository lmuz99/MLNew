# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:23:33 2020

@author: lwmuz
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

import kerastuner as kt
from kerastuner.tuners import RandomSearch
print("Ran the import statements.")
#%%
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
print("GPU Initialised")



        
def ProcessData(train_size, train_batch, valid_size, valid_batch, test_size, test_batch):
    
   df_train = LoadData(train_size, train_batch)
   df_valid = LoadData(valid_size, valid_batch)
   df_test = LoadData(test_size, test_batch)

   print(len(df_train), 'train examples')
   print(len(df_valid), 'validation examples')
   print(len(df_test), 'test examples')    
    
   print("Data processed")
   
   return df_train, df_valid, df_test

    

def LoadData(data, batch):
    filename = "Reconstructed2Q"+ "S" + str(data) +"#"  + str(batch) + ".csv"
    df = pd.read_csv(filename, header = 0, sep = ',' , dtype = float)
    df = df.reindex(np.random.permutation(df.index)) # shuffle the examples
    return df


#function params here correspond to train, validation and test dataset sizes and batchnumbers respectively
label_name = "Fidelity"

train_df, validation_df, test_df = ProcessData(849, 1, 283, 1, 448, 1)

train_label = train_df[label_name]
train_features = train_df.drop(label_name, axis = 1)

val_label = validation_df[label_name]
val_features = validation_df.drop(label_name, axis = 1)




# feature_columns = []

# # Create a numerical feature column to represent median_income.
# for j in ['A_00R','A_00I','A_01R','A_01I','A_02R','A_02I','A_03R','A_03I',\
#           'A_10R','A_10I','A_11R','A_11I','A_12R','A_12I','A_13R','A_13I',\
#           'A_20R','A_20I','A_21R','A_21I','A_22R','A_22I','A_23R','A_23I',\
#           'A_30R','A_30I','A_31R','A_31I','A_32R','A_32I','A_33R','A_33I',\
#           'B_00R','B_00I','B_01R','B_01I','B_02R','B_02I','B_03R','B_03I',\
#           'B_10R','B_10I','B_11R','B_11I','B_12R','B_12I','B_13R','B_13I',\
#           'B_20R','B_20I','B_21R','B_21I','B_22R','B_22I','B_23R','B_23I',\
#           'B_30R','B_30I','B_31R','B_31I','B_32R','B_32I','B_33R','B_33I']:
#     a = tf.feature_column.numeric_column(j)
#     feature_columns.append(a)

# feature_layer = layers.DenseFeatures(feature_columns)



# train_features = {name:np.array(value) for name, value in train_df.items()}
# train_label = np.array(train_features.pop(label_name)) # isolate the label

# val_features = {name:np.array(value) for name, value in validation_df.items()}
# val_label = np.array(val_features.pop(label_name)) # isolate the label

# test_features = {name:np.array(value) for name, value in test_df.items()}
# test_label = np.array(test_features.pop(label_name)) # isolate the label

#%%


def build_model(hp):
    
    dropout = hp.Float('dropout_rate', min_value=0, max_value=0, step=0.01)
    l2reg = hp.Float('regularisation_rate',
                       min_value=3E-5, max_value=6E-5, step=1E-6)
    activation_func = 'relu'
    
    inputs = keras.Input(shape=(64,))
    
    dense0 = layers.Dense(units = hp.Int('units0', min_value=864, max_value=1024, step=16),
                     activation=activation_func,
                     kernel_regularizer=tf.keras.regularizers.l2(l2reg))(inputs)
    
    drop0 = layers.Dropout(rate = dropout)(dense0)
    
    dense1 = layers.Dense(units = hp.Int('units1', min_value=440, max_value=760, step=32),
                     activation=activation_func,
                     kernel_regularizer=tf.keras.regularizers.l2(l2reg))(drop0)
    
    drop1 = layers.Dropout(rate = dropout)(dense1)
    
    dense2 = layers.Dense(units = hp.Int('units2', min_value=440, max_value=760, step=32),
                     activation=activation_func,
                     kernel_regularizer=tf.keras.regularizers.l2(l2reg))(drop1)
    
    drop2 = layers.Dropout(rate = dropout)(dense2)
    
    dense3 = layers.Dense(units = hp.Int('units3', min_value=16, max_value=128, step=16),
                     activation=activation_func,
                     kernel_regularizer=tf.keras.regularizers.l2(l2reg))(drop2)
    
    drop3 = layers.Dropout(rate = dropout)(dense3)
    
    output = layers.Dense(1)(drop3)
    
    model = keras.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate',
                       min_value=2.5E-4, max_value=5E-4, step=1E-5)),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'])
    
    return model

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 4, baseline=0.1,  verbose=2)

tuner = RandomSearch(
    build_model,
    objective='val_mean_absolute_error',
    max_trials=30,
    executions_per_trial=3,
    directory='v1_tuning',
    project_name='numerics10')

tuner.search(train_features, train_label,
              epochs=40,
              validation_data=(val_features, val_label),
              batch_size=4096,
              callbacks = [es],
              verbose = 2
              )

tuner.results_summary()
