import numpy as np
import matplotlib.pyplot as plt
import TwoQbitGen as tq

#make a set of matrices and then filter them into binned concurrence 
#arrays to test how the neural network handles different concurrences
#then we do the same to entropy
#tq.BatchDataMixed(0, 100, select_concurrence = [True, 0.5])

fidelity_values = []
batch_data = []
fidelity_std = []
many_fidelities = []

x, y, z, f, c, s = tq.BatchDataMixed(0, 1000, select_concurrence = [False, 0.5])
fidelity_values.append(x)
batch_data.append(y)
fidelity_std.append(z)
many_fidelities.append(f)


for i in range(len(c)):
    if c[i] > 0.5:
        print("nah mate")
    elif c[i] <0.5:
        print("Yes")
    else:
        print("wtf")
    



plt.hist(many_fidelities, bins = 50, rwidth = 6)
plt.grid()
plt.xlabel("Fidelity")
plt.show()

plt.hist(c, bins = 50, rwidth = 6)
plt.grid()
plt.xlabel("Concurrence")
plt.show()