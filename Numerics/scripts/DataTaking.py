# -*- coding: utf-8 -*-
"""Created on Mon Jan  4 10:39:15 2021"""
#Treat as a run-me file

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import TwoQbitGen as tq
import DataRetrieve as dr
import NeuralNet as NN


'''
First block is going to be looking at how the state concurrence affects loss
I assume that you can return the loss from the NN as a callable method.
For now, all entries are present, although this can be modified directly in
the NN as opposed to here. For now I use a placeholder method for the NN.
'''

def ReplaceMe(batch_size, train_size, train_batch, valid_size, valid_batch, test_size, test_batch):
    
    NN.TrainTestNet(batch_size, train_size, train_batch, valid_size, valid_batch, test_size, test_batch)
    print("NN just ran and returned a loss value average and distribution as an array of losses")
    
    loss = np.random.random_sample()
    loss_array = []
    
    return loss, loss_array


# x: AVG FID
# y: DATA SIZE
# z: standard dev fidelity
# f: all fidelity values in an array
# c: all concurrence values in an array
# s: all VN entropy values in an array
    
# ---------------- CONCURRENCE ------------------------- #
for i in range(5):
    data_size_arr = []#
    losses = [] # loss for different data sizes
    
    #data_sizes = [2,4,8, 16, 32, 64,  128, 256,  512,   1024]
    #             [1,6,28,120,496,2016,8128,32640,130816,523776]
    val_sizes =   [2,3,4,9, 16,32,64,128,256,512]      #manually selecting validation, test and batch sizes, simpler than complex control flow
    test_sizes =  [2,3,6,12,23,45,91,181,359,724]
    batch_sizes = [1,2,4,16,64,128,256,512,1024,1024]
    
    for j in range(4, 11):
        
# ---------------- GENERATE DATA ------------------------- #
        
        size = 2**j
        x, y, z, f, c, s = tq.BatchDataMixed(0, size, select_concurrence = [True, i/6, (i+1)/6])
        dr.ReconstructFile(0, size, conc=True, entr=False)
        
        xV, yV, zV, fV, cV, sV = tq.BatchDataMixed(1, val_sizes[j-1], select_concurrence = [True, i/6, (i+1)/6])
        dr.ReconstructFile(1, val_sizes[j-1], conc=True, entr=False)
        
        xT, yT, zT, fT, cT, sT = tq.BatchDataMixed(2, test_sizes[j-1], select_concurrence = [True, i/6, (i+1)/6])            
        dr.ReconstructFile(2, test_sizes[j-1], conc=True, entr=False)
        
        
# ---------------- RECORD DATA STATS ------------------------- #
        
        averages = [x,y,z,xV,yV,zV,xT,yT,zT]
        avg_columns = ["Train_Avg_Fid","Train_Data_Size","Train_StdDev_Fid",\
                   "Val_Avg_Fid","Val_Data_Size","Val_StdDev_Fid",\
                   "Test_Avg_Fid","Test_Data_Size","Test_StdDev_Fid"]
            
        da = pd.DataFrame(data = averages)
        da = da.T
        da.columns = avg_columns
        filename = "ConcDataAverages" + str(size) +"CONC" + str(i) + ".csv"        
        da.to_csv(filename, index = False)
        
        
        fidelities = [f, fV, fT]
        fid_columns = ["Train_All_Fid", "Val_All_Fid", "Test_All_Fid"]
            
        df = pd.DataFrame(data = fidelities)
        df = df.T
        df.columns = fid_columns
        filename = "ConcFidValues" + str(size) +"CONC" + str(i) + ".csv"        
        df.to_csv(filename, index = False)
        
        
        concs = [c, cV, cT]
        conc_columns = ["Train_All_Conc", "Val_All_Conc", "Test_All_Conc"]
            
        dc = pd.DataFrame(data = concs)
        dc = dc.T
        dc.columns = conc_columns
        filename = "ConcFullValues" + str(size) +"CONC" + str(i) + ".csv"        
        dc.to_csv(filename, index = False)
            
        
# ---------------- RUN NEURAL NET ------------------------- #        
        
        loss_avg, loss_dist = ReplaceMe(batch_sizes[j-1], size, 0, val_sizes[j-1], 1, test_sizes[j-1], 2) # RUN NN on the data just created
        
        data_size_arr.append(y)
        losses.append(loss_avg)
        
        print("#"*60)
        
        
        
    colors = ['r', 'b', 'g', 'k', 'y']   
    plt.plot(data_size_arr, losses, color = colors[i], label = 'C  = ' + str((i + 1)/6))
    plt.grid()
    plt.xlabel("Data Size")
    plt.ylabel("Loss")
    plt.show()
 
#%%       
# ---------------- ENTROPY -----------------------------#
  
for i in range(5):
    data_size_arr = []#
    losses = [] # loss for different data sizes
    all_fids = []
    for j in range(1, 10):
        x, y, z, f, c, s = tq.BatchDataMixed(0, 2**j, select_entropy = [True,i/6, (i+1)/6])
        loss_avg, loss_dist = ReplaceMe() # RUN NN on the data just created
        data_size_arr.append(y)
        losses.append(loss_avg)
        all_fids.append(f)
    
    colors = ['r', 'b', 'g', 'k', 'y']   
    plt.plot(data_size_arr, losses, color = colors[i], label = 'C  = ' + str((i + 1)/6))
    plt.grid()
    plt.xlabel("Data Size")
    plt.ylabel("Loss")
    plt.show() 
    
    
#%%
# --------------- COMPARING MIXED VS PURE -----------------#
    

data_size_arr = []#
losses = [] # loss for different data sizes
all_fids = []
losses_mixed = []

for j in range(1, 10):
    x, y, z = tq.BatchData(0, 2**j)
    loss_avg, loss_dist = ReplaceMe() # RUN NN on the data just created
    data_size_arr.append(y)
    losses.append(loss_avg)
    
    x, y, z, f, c, s = tq.BatchDataMixed(0, 2**j, [False, 0, 1], [False, 0, 1])
    loss_avg, loss_dist = ReplaceMe()
    losses_mixed.append(loss_avg)
    


colors = ['r', 'b', 'g', 'k', 'y']   
plt.plot(data_size_arr, losses, color = 'r', label = 'pure')

plt.plot(data_size_arr, losses_mixed, color = 'b', label = 'mixed')
plt.grid()
plt.xlabel("Data Size")
plt.ylabel("Loss")
plt.show() 



    
    
    
    
    
    
    