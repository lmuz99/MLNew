# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:10:23 2020

@author: timot
"""

import numpy as np
import TwoQbitGen as tq
import matplotlib.pyplot as plt



fidelity_values = []
batch_data = []
fidelity_std = []
for i in range(1, 10):
    a = 2**i
    #print(a)
    x, y, z = tq.BatchDataMixed(0, a)
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
plt.hist(fidelity_values, bins = 10)
plt.show()