#LM - 22/11/2020 
#Algorithm to generate and output simulated datasets for analysis

import numpy as np
import pandas as pd

numQubits = 1   
entries = 50000   #number of entries per batch/file to generate
batches = 10     #number of runs of entries/ files to generate
distributionWidth = 0.01   #float corresponding to rate of decay of distributions observables are drawn from
noise = 0.01   #std dev of normal distribution producing noise

singleStates= [-1, 1]   #List to represent ids of possible states to choice from when generating values for single qubit

myGen = np.random.default_rng()

def Initialise():
    """
    docstring
    """

    global currentBatch
    currentBatch = 0

    for i in range(batches):
        print("start batch: " + str(i+1) + "/" + str(batches))
        currentBatch += 1
        GenBatch()



def GenBatch():
    """
    Generates a dataframe and entries to fill this, before writing to the file
    """
    data = []
    

    if (numQubits == 1):
        for i in range(entries):           
            data.append(Gen1QEntry())
            #print(data[i])

        df = pd.DataFrame(data, columns=['StateID', 'nx', 'ny', 'nz'])
        filename = "testData/" + "Q" + str(numQubits) + "_W" + str(distributionWidth) + "_N" + str(noise) + "_L" + str(entries) + "_BATCH" + str(currentBatch) + ".csv"
        df.to_csv(filename, index = False)



def Gen1QEntry():
    """
    Generate nx, ny and nz values
    """
    state = np.random.choice(singleStates)  #Can modify probability of getting each state if required here

    nx = float(0)
    ny = float(0)
    nz = myGen.exponential(distributionWidth)   #Control length scale of exponential here

    if (state == 1):    #Generate nz value depending on state we are generating
        nz = 1 - nz
    else:
        nz = -1 + nz

    if (noise != 0):
        nx += myGen.normal(0, noise)
        ny += myGen.normal(0, noise)
        nz += myGen.normal(0, noise)
    
    return (state, nx, ny, nz)
    


Initialise()