# -*- coding: utf-8 -*-
"""Created on Mon Jan  4 10:39:15 2021"""
#Treat as a run-me file

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import TwoQbitGen as tq
import DataRetrieve as dr
import NeuralNetReduced as NN


'''
First block is going to be looking at how the state concurrence affects loss
I assume that you can return the loss from the NN as a callable method.
For now, all entries are present, although this can be modified directly in
the NN as opposed to here. For now I use a placeholder method for the NN.
'''

def ReplaceMe(batch_size, train_size, train_batch, valid_size, valid_batch, test_size, test_batch, val_indices, entries):
    
    loss, loss_array, fid, fid_array = NN.TrainTestNet(batch_size, train_size, train_batch, valid_size, valid_batch, test_size, test_batch, val_indices, entries)
    print("NN just ran and returned a loss value average and distribution as an array of losses")

    return loss, loss_array, fid, fid_array


# x: AVG FID
# y: DATA SIZE
# z: standard dev fidelity
# f: all fidelity values in an array
# c: all concurrence values in an array
# s: all VN entropy values in an array

for b in range(16, 17, 1):  #Number of entries to be trained on
    
    avg_hist = []
    
    
    for a in range(0, 1):   #Number of permutations of b entries
    
        val_indices = []
        
        val_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        
        val = random.sample(val_list, b)
        #val = val.sort()
        #val = [0, 1, 4, 5]
        
                
        for x in val:
            xA = (2*x)
        
            val_indices.append(xA)
            val_indices.append(xA+1)
            val_indices.append(xA+32)
            val_indices.append(xA+33)
            
    
        
    # ---------------- CONCURRENCE ------------------------- #
        for i in range(0, 6):    #Conc values for this permutation a of b elements
        
            data_size_arr = []#
            losses = [] # loss for different data sizes
            losses_std = []
            
            fids = []
            fids_std = []
            
            
        
            
            data_sizes =  [2,3,4,5,6,9,12,17,23,33,46,65,91,129,182,257,363,513,725,1025,1449,2049]
            val_sizes =   [2,2,3,3,4,5,6,9,12,17,23,33,46,65,91,129,182,257,363,513,725,1025]      #manually selecting validation, test and batch sizes, simpler than complex control flow
            test_sizes =  [2,3,4,4,5,6,9,12,17,23,33,46,65,91,129,182,257,363,513,725,1025,1449]
            batch_sizes = [1,1,2,2,4,8,8,32,64,128,128,256,512,1024,1024,1024,1024,2048,2048,4096,4096,8192]
            
            for j in range(17, 18):
                
        # ---------------- GENERATE DATA ------------------------- #
                
                size = data_sizes[j]
                maxE = 1
                
                print("RUN START:" + str(b) + " " + str(size) + " " + str(i))
                
                x, y, z, f, c, s = tq.BatchDataMixed(0, size, select_concurrence= [True, maxE * i/6, maxE * (i+1)/6])
                dr.ReconstructFile(0, size, conc=False, entr=False)
                
                xV, yV, zV, fV, cV, sV = tq.BatchDataMixed(1, val_sizes[j], select_concurrence= [True, maxE * i/6, maxE * (i+1)/6])
                dr.ReconstructFile(1, val_sizes[j], conc=False, entr=False)
                
                xT, yT, zT, fT, cT, sT = tq.BatchDataMixed(2, test_sizes[j], select_concurrence= [True, maxE * i/6, maxE * (i+1)/6])        
                dr.ReconstructFile(2, test_sizes[j], conc=False, entr=False)
                
                # ---------------- RUN NEURAL NET ------------------------- #        
                
                #Call NN and return average mean average error and estimated fidelity, as well as the full set of these values
                
                loss_avg1, loss_dist1, fid_avg1, fid_dist1 = ReplaceMe(batch_sizes[j], size, 0, val_sizes[j], 1, test_sizes[j], 2, val_indices, entries=4*b) # RUN NN on the data just created
                loss_avg2, loss_dist2, fid_avg2, fid_dist2 = ReplaceMe(batch_sizes[j], size, 0, val_sizes[j], 1, test_sizes[j], 2, val_indices, entries=4*b)
                loss_avg3, loss_dist3, fid_avg3, fid_dist3 = ReplaceMe(batch_sizes[j], size, 0, val_sizes[j], 1, test_sizes[j], 2, val_indices, entries=4*b)
                    
                loss_avg = np.mean([loss_avg1,loss_avg2,loss_avg3])
                fid_avg = np.mean([fid_avg1,fid_avg2,fid_avg3])
                
                loss_std = np.std([loss_avg1,loss_avg2,loss_avg3])
                fid_std = np.std([fid_avg1,fid_avg2,fid_avg3])
                
                #Append data size with averages to storage for plotting
                data_size_arr.append(y)
                losses.append(loss_avg)
                fids.append(fid_avg)
                
                losses_std.append(loss_std)
                fids_std.append(fid_std)
        
                
                print("#"*80)
                
                
        # ---------------- RECORD DATA STATS ------------------------- #
                
                averages = [size,x,z,xV,yV,zV,xT,yT,zT,fid_avg1,fid_avg2,fid_avg3,loss_avg1,loss_avg2,loss_avg3,fid_avg, fid_std, loss_avg, loss_std, val, a, i]
                avg_columns = ["Train_Data_Size","Train_Avg_Fid","Train_StdDev_Fid",\
                           "Val_Avg_Fid","Val_Data_Size","Val_StdDev_Fid",\
                           "Test_Avg_Fid","Test_Data_Size","Test_StdDev_Fid",\
                           "fid_avg1","fid_avg2","fid_avg3",\
                           "loss_avg1","loss_avg2","loss_avg3",\
                           "NN_Test_Avg_Fid", "NN_Test_Std_Fid", "NN_Test_Avg_MAE", "NN_Test_Std_MAE", "Val_Indices", "Trial", "Concurrence"]
                    
                da = pd.DataFrame(data = averages)
                da = da.T
                da.columns = avg_columns
                filename = "Mixed_RUN_AVGS N"+ str(b) + "Trial" + str(a) +"Conc" + str(i) + "Size" + str(size) + ".csv"        
                da.to_csv(filename, index = False)
                
                avg_hist.append(averages) 
                
                
                fidelities = [f, fV, fT, fid_dist1,fid_dist2,fid_dist3,loss_dist1,loss_dist2,loss_dist3]
                fid_columns = ["Train_All_Fid", "Val_All_Fid", "Test_All_Fid", "NN_Test_All_Fid1", "NN_Test_All_Fid2", "NN_Test_All_Fid3", "NN_Test_All_MAE1", "NN_Test_All_MAE2", "NN_Test_All_MAE3"]
                    
                df = pd.DataFrame(data = fidelities)
                df = df.T
                df.columns = fid_columns
                filename = "MIX_RUN_FIDS N"+ str(b) + "Trial" + str(a) +"Conc" + str(i) + "Size" + str(size) + ".csv"        
                df.to_csv(filename, index = False)
                
                
                concs = [c, cV, cT]
                conc_columns = ["Train_All_Conc", "Val_All_Conc", "Test_All_Conc"]
                    
                dc = pd.DataFrame(data = concs)
                dc = dc.T
                dc.columns = conc_columns
                filename = "MIX_RUN_CONC N"+ str(b) + "Trial" + str(a) +"Conc" + str(i) + "Size" + str(size) + ".csv"         
                dc.to_csv(filename, index = False)
                    
                
        
                
                
                
            # colors = ['r', 'b', 'g', 'k', 'y']   
            # plt.plot(data_size_arr, losses, color = colors[i], label = 'C  = ' + str((i + 1)/6))
            # plt.grid()
            # plt.xlabel("Data Size")
            # plt.ylabel("Loss")
            # plt.show()
            
    avg_columns = ["Train_Data_Size","Train_Avg_Fid","Train_StdDev_Fid",\
       "Val_Avg_Fid","Val_Data_Size","Val_StdDev_Fid",\
       "Test_Avg_Fid","Test_Data_Size","Test_StdDev_Fid",\
       "fid_avg1","fid_avg2","fid_avg3",\
       "loss_avg1","loss_avg2","loss_avg3",\
       "NN_Test_Avg_Fid", "NN_Test_Std_Fid", "NN_Test_Avg_MAE", "NN_Test_Std_MAE", "Indices", "Trial", "Concurrence"]

    dfinal = pd.DataFrame(data = avg_hist)
    #dfinal = dfinal.T
    dfinal.columns = avg_columns
    filename = "MIX_FINAL_AVG N" + str(b) + "Conc.csv"       
    dfinal.to_csv(filename, index = False)

    
    
    
    
    