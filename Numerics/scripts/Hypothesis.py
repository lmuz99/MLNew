# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:02:01 2020

@author: timot
"""

import numpy as np
import pandas as pd
import TwoQbitGen as tq
import matplotlib.pyplot as plt

fidelity_values = []
batch_data = []
fidelity_std = []
for i in range(9):
    a = 2**(i+1) + 120
    #print(a)
    x, y, z = tq.BatchData(0, a)
    fidelity_values.append(x)
    batch_data.append(y)
    fidelity_std.append(z)

plt.plot(batch_data, fidelity_values)

plt.xlabel("Batch Size")
plt.ylabel("Fidelity")
plt.errorbar(batch_data, fidelity_values, xerr = None, yerr = fidelity_std, barsabove = True )
plt.grid()
plt.show()
print(fidelity_values)
#%%
def FidValDist():
    '''
    Aim is to create a binned distribution of fidelity values. Empirically, we find these values to 
    converge at around a batch size of 300+, anthing under that is subject to first moment fluctuations
    about the mean.
    '''
    #make an array and append the fidelity values for different batch sizes 
    #greater than 300
    fidelities_raw = []
    for i in range(500):
        x, y, z = tq.BatchData(0, 400) #x is fidelity values, y is number of density matrices and z is the varience 
        fidelities_raw.append(x)
        print(i/5, "%")
        
    plt.hist(fidelities_raw, bins = 20)
    plt.show()

FidValDist()



    