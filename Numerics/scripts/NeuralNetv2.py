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



print("Ran the import statements.")

def Initialise():
    
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
    df = df.reindex(np.random.permutation(df.index)) # shuffle the examples)
    return df






def BuildModel(learning_rate):
    
    l2reg = 5.5E-5
    activation_func = 'relu'
    
    inputs = keras.Input(shape=(64))
    
    dense0 = layers.Dense(units = 992,
                     activation=activation_func,
                     kernel_regularizer=tf.keras.regularizers.l2(l2reg))(inputs)
    
    
    dense1 = layers.Dense(units = 568,
                     activation=activation_func,
                     kernel_regularizer=tf.keras.regularizers.l2(l2reg))(dense0)
    
    
    dense2 = layers.Dense(units = 568,
                     activation=activation_func,
                     kernel_regularizer=tf.keras.regularizers.l2(l2reg))(dense1)
    
    
    dense3 = layers.Dense(units = 32,
                     activation=activation_func,
                     kernel_regularizer=tf.keras.regularizers.l2(l2reg))(dense2)
    
    
    output = layers.Dense(1)(dense3)
    
    model = keras.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'])
    
    return model


def TrainModel(model, dataset, validation_set, epochs, label_name,
                batch_size=None):
  """Train the model by feeding it data."""

  train_label = dataset[label_name]
  train_features = dataset.drop(label_name, axis = 1)
    
  val_label = validation_set[label_name]
  val_features = validation_set.drop(label_name, axis = 1)  


  history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                      epochs=epochs, shuffle=True, validation_data=(val_features, val_label), verbose = 2) 

  # The list of epochs is stored separately from the rest of history.
  epochs = history.epoch
  
  # To track the progression of training, gather a snapshot
  # of the model's mean squared error at each epoch. 
  hist = pd.DataFrame(history.history)
  rmse = hist["mean_absolute_error"]

  return epochs, rmse, hist
       


def TrainTestNet(batch_size, train_size, train_batch, valid_size, valid_batch, test_size, test_batch):
    Initialise()
    train_df, validation_df, test_df = ProcessData(train_size, train_batch, valid_size, valid_batch, test_size, test_batch)

    label_name = "Fidelity"


    # The following variables are the hyperparameters.
    learning_rate = 0.00045
    epochs = 70
    

    my_model = BuildModel(learning_rate = learning_rate)
    

    epochs, rmse, history = TrainModel(my_model, train_df, validation_df, epochs, 
                              label_name, batch_size)
    #plot_the_loss_curve(epochs, mse)
    
    

    test_label = test_df[label_name]
    test_features = test_df.drop(label_name, axis = 1)
    
    print("\n Evaluating against the test set:")
    
    #his = my_model.evaluate(x = test_features, y = test_label, batch_size=1)
    
    #my_model.save("modelStorage/alpha")

    fid_history, mae_history = GetTestLosses(test_label, test_features, my_model)
    
    print("\n Evaluation Complete: Mean mae = " + str(np.mean(mae_history)))
    
    return np.mean(mae_history), mae_history, np.mean(fid_history), fid_history



def JustTestNet(test_size, test_batch):
    my_model = tf.keras.models.load_model("modelStorage/alpha")
    label_name = "Fidelity"
    
    test_df = ProcessData(test_size, test_batch)
    test_label = test_df[label_name]
    test_features = test_df.drop(label_name, axis = 1)
    
    fid_history, mae_history = GetTestLosses(test_label, test_features, my_model)
    
    return np.mean(mae_history, mae_history)



def GetTestLosses(test_label, test_features, model):
    
    fid = model.predict([test_features])[:,0]
    test_label = test_label.to_numpy()
    
    error = np.abs(fid - test_label)
    
   # for i in range(0, len(fid)):
    #    error.append(np.abs(fid[i] - test_label[i]))
    
    return fid, error
