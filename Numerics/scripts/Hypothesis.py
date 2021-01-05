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
#%%

import numpy as np
import pandas as pd
import TwoQbitGen as tq
import matplotlib.pyplot as plt
def MyFit(x, a,b):
    return -1/(a*x -b)
def MyFit2(x, c, d, g):
    if x[i] <= c :
        return 0
    else:
        return -1/(g*x -d)

'''
def expfit(x,a,b,c):
    return a*sp.exp(-b*x) + c
popt,pcov = curve_fit(expfit,thickness,countRateAverage,p0 = (10,10,10))
thickness_fit = sp.arange(0,5,0.0001)
plt.plot(thickness_fit, expfit(thickness_fit, *popt), color = 'red', label = 'fit for Al')
'''
    

ent_mix = []
conc = []
log_ent_mix = []
log_conc = []
for i in range(10000):
    d = tq.Mixed2QGen()
    c = tq.TransformMatrix(d)
    conc.append(2 * c[2,1])
    ent_mix.append(tq.StateEntropy(c, 100))
    log_conc.append(np.log(2 * c[2,1]))
    log_ent_mix.append(tq.StateEntropy(c, 100))
'''
fit_array = np.arange(0, 1, 0.001)
popt,pcov = curve_fit(MyFit, conc, ent_mix, p0 = (100,100))
plt.plot(fit_array, MyFit(fit_array, *popt), color = 'green')
popt, pcov = curve_fit(MyFit2, conc, ent_mix, p0 = (0.99, 100, 100))
plt.plot(fit_array, MyFit2(fit_array, *popt), color = 'red')
print(popt)
'''    
plt.plot(log_conc, log_ent_mix, 'bx')

plt.xlabel("Mixed State Concurrence")
plt.ylabel("Von Neumann Entropy of Entanglement")
plt.grid()

plt.show()
#%%
import TwoQbitGen as tq

test_mat = tq.Mixed2QGen()
print(test_mat)
g = tq.TransformMatrix(test_mat)
print(g)
#plt.plot(conc, log_ent_mix, 'rx')

new_fid = tq.FidelityMixed(g, g)
print(new_fid)
#%%
# -*- coding: utf-8 -*-
"""
Distribution of fidelity values looks like a boltzmann...
"""

import numpy as np
import TwoQbitGen as tq
import matplotlib.pyplot as plt



fidelity_values = []
batch_data = []
fidelity_std = []
many_fidelities = []
for i in range(1, 10):
    a = 2**i
    #print(a)
    x, y, z, f = tq.BatchDataMixed(0, a)
    fidelity_values.append(x)
    batch_data.append(y)
    fidelity_std.append(z)
    many_fidelities.append(f)
'''
plt.plot(batch_data, fidelity_values)
plt.xlabel("Batch Size")
plt.ylabel("Fidelity")
plt.errorbar(batch_data, fidelity_values, xerr = None, yerr = fidelity_std, barsabove = True )
plt.grid()
plt.show()
print(fidelity_values)
'''
plt.hist(many_fidelities, bins = 150, rwidth = 6)
plt.gca().invert_xaxis()
plt.grid()
plt.show()



    