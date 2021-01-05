# -*- coding: utf-8 -*-
"""Created on Mon Jan  4 10:39:15 2021
Treat as a run-me file
"""
import numpy as np
import matplotlib.pyplot as plt
import TwoQbitGen as tq


'''
First block is going to be looking at how the state concurrence affects loss
I assume that you can return the loss from the NN as a callable method.
For now, all entries are present, although this can be modified directly in
the NN as opposed to here. For now I use a placeholder method for the NN.
'''

def ReplaceMe():
    print("NN just ran and returned a loss value average and distribution as an array of losses")
    loss = np.random.random_sample()
    loss_array = []
    
    return loss, loss_array


# x: AVG FID
# y: DATA SIZE
# z: standard dev fidelity
# f: all fidelity values in an array
# c: al concurrence values in an array
# s: all VN entropy values in an array
    
# ---------------- CONCURRENCE ------------------------- #
for i in range(5):
    data_size_arr = []#
    losses = [] # loss for different data sizes
    
    for j in range(1, 10):
        x, y, z, f, c, s = tq.BatchDataMixed(0, 2**j, select_concurrence = [True, i/6, (i+1)/6])
        loss_avg, loss_dist = ReplaceMe() # RUN NN on the data just created
        data_size_arr.append(y)
        losses.append(loss_avg)
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



    
    
    
    
    
    
    