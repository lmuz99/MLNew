# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 14:23:33 2020

@author: lwmuz
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from matplotlib import pyplot as plt
#from sklearn.model_selection import train_test_split

print("Ran the import statements.")

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
    
    
def create_model(my_learning_rate, my_feature_layer):
  """Create and compile a simple linear regression model."""
  # Most simple tf.keras models are sequential.
  model = tf.keras.models.Sequential([my_feature_layer])

  # Add the layer containing the feature columns to the model.
  #model.add(my_feature_layer)

  # Describe the topography of the model by calling the tf.keras.layers.Dense
  # method once for each layer. We've specified the following arguments:
  #   * units specifies the number of nodes in this layer.
  #   * activation specifies the activation function (Rectified Linear Unit).
  #   * name is just a string that can be useful when debugging.

  # Define the first hidden layer with 20 nodes.   
  model.add(tf.keras.layers.Dense(units=256, 
                                  activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.000001),
                                  name='Hidden1'))
  model.add(tf.keras.layers.Dense(units=640, 
                                  activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                  name='Hidden2'))
  model.add(tf.keras.layers.Dense(units=320, 
                                  activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                  name='Hidden3'))  
  # Define the second hidden layer with 12 nodes. 
  model.add(tf.keras.layers.Dense(units=160,
                                  activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.000001),
                                  name='Hidden4'))
  model.add(tf.keras.layers.Dense(units=16,
                                  activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.000001),
                                  name='Hidden5'))
  
  # Define the output layer.
  model.add(tf.keras.layers.Dense(units=1,  
                                  name='Output'))                              
  
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                loss="mean_absolute_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

  return model


def train_model(model, dataset, validation_set, epochs, label_name,
                batch_size=None):
  """Train the model by feeding it data."""

  # Split the training set into features and label.
  features = {name:np.array(value) for name, value in dataset.items()}
  label = np.array(features.pop(label_name))
  
  # Split the validation set into features and label.
  val_features = {name:np.array(value) for name, value in validation_set.items()}
  val_label = np.array(val_features.pop(label_name))
  
  history = model.fit(x=features, y=label, batch_size=batch_size,
                      epochs=epochs, shuffle=True, validation_data=(val_features, val_label)) 

  # The list of epochs is stored separately from the rest of history.
  epochs = history.epoch
  
  # To track the progression of training, gather a snapshot
  # of the model's mean squared error at each epoch. 
  hist = pd.DataFrame(history.history)
  rmse = hist["root_mean_squared_error"]

  return epochs, rmse, hist



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

print("Defined the plot_the_loss_curve function.")

#%%
#function params here correspond to train, validation and test dataset sizes and batchnumbers respectively

train_df, validation_df, test_df = ProcessData(849, 1, 283, 1, 448, 1)


#%%

feature_columns = []

# Create a numerical feature column to represent median_income.
for j in ['A_00R','A_00I','A_01R','A_01I','A_02R','A_02I','A_03R','A_03I',\
          'A_10R','A_10I','A_11R','A_11I','A_12R','A_12I','A_13R','A_13I',\
          'A_20R','A_20I','A_21R','A_21I','A_22R','A_22I','A_23R','A_23I',\
          'A_30R','A_30I','A_31R','A_31I','A_32R','A_32I','A_33R','A_33I',\
          'B_00R','B_00I','B_01R','B_01I','B_02R','B_02I','B_03R','B_03I',\
          'B_10R','B_10I','B_11R','B_11I','B_12R','B_12I','B_13R','B_13I',\
          'B_20R','B_20I','B_21R','B_21I','B_22R','B_22I','B_23R','B_23I',\
          'B_30R','B_30I','B_31R','B_31I','B_32R','B_32I','B_33R','B_33I']:
    a = tf.feature_column.numeric_column(j)
    feature_columns.append(a)



# Convert the list of feature columns into a layer that will later be fed into
# the model. 
feature_layer = layers.DenseFeatures(feature_columns)




# The following variables are the hyperparameters.
learning_rate = 0.0001
epochs = 100
batch_size = 1000

# Specify the label
label_name = "Fidelity"

# Establish the model's topography.
my_model = create_model(learning_rate, feature_layer)

# Train the model on the normalized training set. We're passing the entire
# normalized training set, but the model will only use the features
# defined by the feature_layer.
epochs, rmse, history = train_model(my_model, train_df, validation_df, epochs, 
                          label_name, batch_size)
#plot_the_loss_curve(epochs, mse)

# After building a model against the training set, test that model
# against the test set.
test_features = {name:np.array(value) for name, value in test_df.items()}
test_label = np.array(test_features.pop(label_name)) # isolate the label
print("\n Evaluate the new model against the test set:")
my_model.evaluate(x = test_features, y = test_label, batch_size=batch_size)

plot_the_loss_curve(epochs, history["root_mean_squared_error"], 
                    history["val_root_mean_squared_error"])


