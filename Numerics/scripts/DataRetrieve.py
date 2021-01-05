'''
Algorithm for retrieval of density matrix pair and corresponding fidelity.
Method that reconstructs the full original 4x4 density matrix from the database
Method that reconstructs the partial original (not all entries) with noise
'''
import numpy as np
import pandas as pd
import TwoQbitGen as tq
#import tensorflow as tf

# tq.BatchData(1, 100)
# data_size = 250
# batchnumber = 1

# vector = pd.read_csv("data_ref_fidS100#1.csv", header = 0, sep = ',', dtype = float)
# print("Creating Data")
# tq.BatchData(batchnumber, data_size)
# print("Data Created")


# #vector = pd.read_csv("data_ref_fidS250#1.csv", header = 0, sep = ',', dtype = \
#                     # {"Density Ref1":np.int32,"Density Ref2":np.int32,"Fidelity":float})
# #matrices = pd.read_csv("Matrices2QS250#1.csv", header = 0, sep = ',' , dtype = float)

# #to match up a pair of matrices, simply index the vector's 0th and 1st positions
# #for example:

# #mat_A_vect = matrices.iloc[vector.iloc[17,0]]
# #mat_B_vect = matrices.iloc[vector.iloc[17,1]]
# #to reconstruct the matrix in matrix form we need to take pairs and make them entries

# buffer = []

# for i in range(0, len(vector)):
#     mat_A_vect = matrices.iloc[vector.iloc[i,0]]
#     mat_B_vect = matrices.iloc[vector.iloc[i,1]]
#     temp = pd.concat([mat_A_vect, mat_B_vect, pd.Series(vector.iloc[i,2])], ignore_index=True)
#     buffer.append(temp)
#     if(i%1000 == 0):
#         print((100*i)/len(vector),"%")

# df = pd.concat(buffer, axis=1)
# df = df.T
# columns = ['A_00R','A_00I','A_01R','A_01I','A_02R','A_02I','A_03R','A_03I',\
#                                      'A_10R','A_10I','A_11R','A_11I','A_12R','A_12I','A_13R','A_13I',\
#                                      'A_20R','A_20I','A_21R','A_21I','A_22R','A_22I','A_23R','A_23I',\
#                                      'A_30R','A_30I','A_31R','A_31I','A_32R','A_32I','A_33R','A_33I',\
#                                      'B_00R','B_00I','B_01R','B_01I','B_02R','B_02I','B_03R','B_03I',\
#                                      'B_10R','B_10I','B_11R','B_11I','B_12R','B_12I','B_13R','B_13I',\
#                                      'B_20R','B_20I','B_21R','B_21I','B_22R','B_22I','B_23R','B_23I',\
#                                      'B_30R','B_30I','B_31R','B_31I','B_32R','B_32I','B_33R','B_33I', "Fidelity"]
# filename = "Reconstructed2Q"+ "S" + str(data_size) +"#"  + str(batchnumber) + ".csv"
# df.columns=columns

# print("Dataframe Constructed")

# df.to_csv(filename, index = False)

# print("File written")

     
def ReconstructFile(batch_num, data_size, conc, entr):
    print("Reconstructing data")
    if(conc):
        vector = pd.read_csv("MixedData_ref_fidCONC" + "S" + str(data_size) + "#" + str(batch_num) + ".csv", header = 0, sep = ',', dtype = \
                     {"Density Ref1":np.int32,"Density Ref2":np.int32,"Fidelity":float})
        
    matrices = pd.read_csv("Matrices2Q"+ "S" + str(data_size) +"#"  + str(batch_num) + ".csv", header = 0, sep = ',' , dtype = float)
    
    buffer = []

    for i in range(0, len(vector)):
        mat_A_vect = matrices.iloc[vector.iloc[i,0]]
        mat_B_vect = matrices.iloc[vector.iloc[i,1]]
        temp = pd.concat([mat_A_vect, mat_B_vect, pd.Series(vector.iloc[i,2])], ignore_index=True)
        buffer.append(temp)
        if(i%1000 == 0):
            print((100*i)/len(vector),"%")
    
    df = pd.concat(buffer, axis=1)
    df = df.T
    columns = ['A_00R','A_00I','A_01R','A_01I','A_02R','A_02I','A_03R','A_03I',\
                                         'A_10R','A_10I','A_11R','A_11I','A_12R','A_12I','A_13R','A_13I',\
                                         'A_20R','A_20I','A_21R','A_21I','A_22R','A_22I','A_23R','A_23I',\
                                         'A_30R','A_30I','A_31R','A_31I','A_32R','A_32I','A_33R','A_33I',\
                                         'B_00R','B_00I','B_01R','B_01I','B_02R','B_02I','B_03R','B_03I',\
                                         'B_10R','B_10I','B_11R','B_11I','B_12R','B_12I','B_13R','B_13I',\
                                         'B_20R','B_20I','B_21R','B_21I','B_22R','B_22I','B_23R','B_23I',\
                                         'B_30R','B_30I','B_31R','B_31I','B_32R','B_32I','B_33R','B_33I', "Fidelity"]
    filename = "Reconstructed2Q"+ "S" + str(data_size) +"#"  + str(batch_num) + ".csv"
    df.columns=columns
    
    print("Dataframe Constructed")
    
    df.to_csv(filename, index = False)
    
    print("File written")

def Conv_Vect_Mat(vector_32):
    '''
    Method is to reconstruct the full complex valued 4x4 matrix from the 32D 
    vector.
    IN: 32D vector
    OUT: 4x4 complex values matrix with real and imaginary pairs coming from
    neighbouring pairs in the 32D vector. See nested loop for exact indexing.
    '''
    if (len(vector_32)) != 32:
        raise Exception("Vector must be of length 32")
    else:
        pass
    converted_matrix = np.zeros(shape = (4,4), dtype = complex)
    converted_matrix = np.zeros(shape = (4,4), dtype = complex) #change to tf tensor
    for i in range(4):
        for j in range(4):
            converted_matrix[i,j] = vector_32[8 * i + 2 * j] + 1j * vector_32[8 * i + 2 * j + 1]
            converted_matrix[i,j] = vector_32[8 * i + 2 * j] + 1j * vector_32[8 * i + 2 * j + 1]    #check indexing here

    return converted_matrix

# new_matrix1 = Conv_Vect_Mat(mat_A_vect)
# new_matrix2 = Conv_Vect_Mat(mat_B_vect)
# a = tq.Fidelity(new_matrix1, new_matrix2)
#new_matrix1 = Conv_Vect_Mat(mat_A_vect)
#new_matrix2 = Conv_Vect_Mat(mat_B_vect)
#a = tq.Fidelity(new_matrix1, new_matrix2)
#print(a, vector.iloc[17,2])

def RemoveEntries(matrix_4by4):
    '''
    IN: 4x4 density matrix of complex values
    OUT: Array length 2:
        Array[0] = density matrix
        Array[1] = truth matrix associated
    Use to decide which values the ML algorithm is "allowed" to use as opposed to
    just removing them or having NaNs etc...
    '''
    if matrix_4by4.shape != (4,4):
        raise Exception("Argument must be 4x4 matrix")
    truth_matrix = np.zeros(shape = (4,4), dtype = bool)
    #want to set a controlled number of these bools to false in random positions
    #in the matrix. 
    removal_number = 3 #NB cannot be higher than 15
    for i in range(removal_number):
        a = np.random.randint(0, 3)
        b = np.random.randint(0, 3)
        truth_matrix[a,b] = True
        a = 0
        b = 0
    #print(truth_matrix)

    return [matrix_4by4, truth_matrix]
    

#RemoveEntries(new_matrix1)

def AddNoise(matrix_4by4):
    '''
    IN: 4x4 Matrix of complex numbers
    OUT: 4x4 Matrix of complex numbers with noise
    METHOD DOES NOT RETAIN PURITY
    '''
    if matrix_4by4.shape != (4,4):
        raise Exception("Argument must be 4x4 matrix")
    if (np.matmul(matrix_4by4, matrix_4by4)[1,1] - matrix_4by4[1,1] <= 1E-10):
        print("Purity condition held with no noise")
    else:
        print("Purity condition violated, noisless matrix is a mixed state")
    #loc gives the mean of gaussian and scale gives the std dev. Numpy will select
    # a random number which means there is a risk of non-zero trace.
    noisy_matrix_real = np.random.normal(loc = 0., scale = 0.1, size = (4,4))
    noisy_matrix_imag = np.random.normal(loc = 0., scale = 0.1, size = (4,4))
    #print(matrix_4by4, "ORIGINAL")
    
    for i in range(3):
        for k in range(3):
            matrix_4by4 += noisy_matrix_real[i,k] + 1j*noisy_matrix_imag[i,k]
            #print(matrix_4by4[i,k])
            
    if (np.matmul(matrix_4by4, matrix_4by4)[1,1] - matrix_4by4[1,1] <= 1E-10):
        print("Purity condition held with noise")
    else:
        print("Purity condition violated, noisy matrix is a mixed state")
        print(np.matmul(matrix_4by4, matrix_4by4)[1,1] - matrix_4by4[1,1])
            
    return matrix_4by4

#RemoveEntries(new_matrix1)
#AddNoise(new_matrix1)
#AddNoise(new_matrix1)
    