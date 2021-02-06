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


import TwoQbitGen as tq
import DataRetrieve as dr
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

#%%
tq.BatchData(1, 363)
tq.BatchData(1, 182)
tq.BatchData(1, 257)

dr.ReconstructFile(1, 363, conc=False, entr=False)
dr.ReconstructFile(1, 182, conc=False, entr=False)
dr.ReconstructFile(1, 257, conc=False, entr=False)


#%%
train_df, validation_df, test_df = ProcessData(363, 1, 182, 1, 257, 1)

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

    lri = hp.Float('learning rate',
                       min_value=1E-5, max_value=1E-2, step=1E-5)
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    lri,
    decay_steps=10,
    decay_rate=hp.Float('decay rate',
                       min_value=0.9, max_value=1, step=0.0025),
    staircase=True)
    
    #l2reg = hp.Float('regularisation_rate',
                       #min_value=0, max_value=1E-2, step=1E-6)
    activation_func1 = 'relu'
    activation_func2 = 'relu'
        
    
    inputs = keras.Input(shape=(64)) 
    
    output = layers.Dense(1)(inputs)
    
    model = keras.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(
        lr_schedule),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'])
    
    return model

def plot_the_loss_curve(epochs, mae_training, mae_validation):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs[1:], mae_training[1:], label="Training Loss")
  plt.plot(epochs[1:], mae_validation[1:], label="Validation Loss")
  plt.legend()
  
  # We're not going to plot the first epoch, since the loss on the first epoch
  # is often substantially greater than the loss for other epochs.
  merged_mae_lists = mae_training[1:] + mae_validation[1:]
  highest_loss = max(merged_mae_lists)
  lowest_loss = min(merged_mae_lists)
  delta = highest_loss - lowest_loss
  print(delta)

  top_of_y_axis = highest_loss + (delta * 0.05)
  bottom_of_y_axis = lowest_loss - (delta * 0.05)
   
  #plt.ylim([bottom_of_y_axis, top_of_y_axis])
  plt.show()  


es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 20, baseline=0.06, min_delta=1E-5, verbose=2)

tuner = RandomSearch(
    build_model,
    objective='val_mean_absolute_error',
    max_trials=50,
    executions_per_trial=2,
    directory='simple_tuning',
    project_name='r0')

#tuner = kt.tuners.bayesian.BayesianOptimization(build_model, 'val_mean_absolute_error', max_trials = 100, directory='final_tuning',
 #    project_name='b1')

tuner.search(train_features, train_label,
              epochs=160,
              validation_data=(val_features, val_label),
              batch_size=2048,
              callbacks = [es],
              verbose = 2
              )

tuner.results_summary()
