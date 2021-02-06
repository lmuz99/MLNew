  
# -*- coding: utf-8 -*-
"""
Generates every possible pure state density matrix to a resolution r, with every
possible number of missing entries, with noise control for error on each entry.
Other control parameters englude creating more constrained data sets such as the 
set of maximally entangles density matrices.
1. Resoultion of set
2. Number of missing entries per matrix
3. Error per entry per matrix
4. Enganglement entropy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import sqrtm
#import tensorflow as tf

#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)



#number of entries/batch to be made
batches = 10
entries = 16 #Numer of entries per matrix
noise = 0 #fraction of random error per matrix entry
entang_entr = 1 #entanglement entropy


def MakeDensityMatrix(entropy = False):
    #Produce random values for the six params over appropriate ranges
    chi = np.random.random_sample()*0.5*np.pi
    theta_1 = np.random.random_sample()*np.pi
    theta_2 = np.random.random_sample()*np.pi
    phi_1 = np.random.random_sample()*2*np.pi
    phi_2 = np.random.random_sample()*2*np.pi
    gamma = np.random.random_sample()*2 - 1
    
    #Code to check boundary arguments should we need it later
    if (chi <= 0) or (chi>= np.pi):
        raise Exception("chi out of bounds 0 to pi")
    if (theta_1 <= 0) or (theta_1>= np.pi):
        raise Exception("theta_1 out of bounds 0 to pi") 
    if (theta_2 <= 0) or (theta_2>= np.pi):
        raise Exception("theta_2 out of bounds 0 to pi") 
    if (phi_1 <= 0) or (phi_1 >= 2 * np.pi):
        raise Exception("phi_1 out of bounds 0 to 2pi")
    if (phi_1 <= 0) or (phi_2 >= 2 * np.pi):
        raise Exception("phi2 out of bounds 0 to 2pi") 
        
    #turn these angle arguments into complex coefficients of the 2-Qbit wavefunction
    # |psi> = a|00> + b|01> + c|10> + d|11>
    a = (np.cos(chi/2)*np.cos(theta_1/2)*np.cos(theta_2/2)*np.exp(1j*gamma/2) \
       + np.sin(chi/2)*np.sin(theta_1/2)*np.sin(theta_2/2)*np.exp(-1j*gamma/2)) \
         * np.exp(-1j*(phi_1 + phi_2)/2)
    b = (np.cos(chi/2)*np.cos(theta_1/2)*np.sin(theta_2/2)*np.exp(1j*gamma/2) \
       - np.sin(chi/2)*np.sin(theta_1/2)*np.cos(theta_2/2)*np.exp(-1j*gamma/2)) \
         * np.exp(-1j*(phi_1 - phi_2)/2)
    c = (np.cos(chi/2)*np.sin(theta_1/2)*np.cos(theta_2/2)*np.exp(1j*gamma/2) \
       - np.sin(chi/2)*np.cos(theta_1/2)*np.sin(theta_2/2)*np.exp(-1j*gamma/2)) \
         * np.exp(1j*(phi_1 - phi_2)/2)
    d = (np.cos(chi/2)*np.sin(theta_1/2)*np.sin(theta_2/2)*np.exp(1j*gamma/2) \
       + np.sin(chi/2)*np.cos(theta_1/2)*np.cos(theta_2/2)*np.exp(-1j*gamma/2)) \
         * np.exp(1j*(phi_1 + phi_2)/2)
    coefs = [a,b,c,d]
    
     #Contruct the density matrix from these values via |psi><psi|
    rho_full =  np.zeros(shape = (4,4), dtype = complex) #Change this to a tf matrix?
    
    for i in range(0,4):
        for k in range(0,4): 
           rho_full[i,k] = coefs[i] * np.conj(coefs[k])
            
    if (rho_full[i,k] != np.conj(rho_full[k,i])):   #change to tf conjugation
                print("Entry", i, k, "is not the complex conjugate of", k, i)
                raise Exception("DensityMatrixError: matrix must be hermitian")
    #critial value of 10^-13 for floating point error on putity
    if (np.matmul(rho_full, rho_full)[1,1] - rho_full[1,1] <= 1E-13):
        pass
    else:
        print("Purity condition violated")
        
    if (entropy == True):
        #Make the subsystem density matrix
        sub_rho = np.zeros(shape = (2,2), dtype = float)
        sub_rho[0,0] = 0.5 * ( 1 + np.cos(chi) * np.cos(theta_1))
        sub_rho[0,1] = 0.5 * np.cos(chi) * (np.cos(phi_1) * np.sin(theta_1) - \
               1j * np.sin(phi_1) * np.sin(theta_1))
        sub_rho[1,0] = np.conj(sub_rho[0,1])
        sub_rho[1,1] = 1 - sub_rho[0,0]
        #Calculate the sub VNE
        entropy = SubsystemVNE(sub_rho, 100)
        return rho_full, entropy
    else:
        return rho_full
    # Eventually we will make the angles the arguments for procedural generation

def Fidelity(matA, matB):
    #Matrices must be 4 by 4 or error is raised.
    CheckShape(matA)
    CheckShape(matB)
    product = np.matmul(matA,matB)  #TF change here
    trace = 0
    for i in range(0,4):
        trace += np.real(product[i,i])  #check tf for trace method
        #NB: take the real because there is sometimes floating point complex error here
    if (trace < 0) or (trace > 1.00000001):
        print(trace)
        raise Exception("Fidelity must be between 0 and 1, the matrices need renormalising")
    return trace     
    #uses the trace relation F(A,B) = Tr(AB) for pure states

def BatchData(batchnumber, data_size, select_entropy = [False,0,2]):
    raw_matrix_data = []
    #ref_data has 3 vectors for each element in the array corresponding to 2 references
    #to two density matrices in the raw_matrix_data array and their shared fidelity value

    
    ref_data = []
    sub_entropy_arr = []
    counter = 0
    while (counter <= data_size):
        if (select_entropy[0] == False):
            counter += 1
            mat = MakeDensityMatrix()
            raw_matrix_data.append(mat)
        else:
            mat, entropy = MakeDensityMatrix(True)
            if (entropy >= select_entropy[1] and (entropy <= select_entropy[2])):
                counter += 1
                raw_matrix_data.append(mat)
                sub_entropy_arr.append(entropy)
            else:
                pass
    for i in range(data_size):
        fid = Fidelity(raw_matrix_data[0], raw_matrix_data[i])
        ref_data.append([0, i, fid])
            
    #print(len(ref_data))
    #Make 2 files per batch: 1 to store the density matrices and 1 to store 2 labels and the fidelity
    filename = "data_ref_fid" + "S" + str(data_size) + "#" + str(batchnumber) + ".csv"

    #Stores the ref_data 
    df = pd.DataFrame(ref_data, columns=['Density Ref1', 'Density Ref2', 'Fidelity'])

    df.to_csv(filename, index = False)
    #Converts the density matrices into a 32D vector and stores, flattening happens
    # row by row.
    #print(ref_data)
    fid_sum = 0
    varience_buf = []
    for i in range(len(ref_data)):
        fid_sum += ref_data[i][2]   #check this works for tf
        varience_buf.append(ref_data[i][2])
    avg_fid = fid_sum/len(ref_data)
    std_fid = np.var(varience_buf)
    
    all_fidelity_vals = []
    for i in range(data_size):
        for j in range(i+1, data_size): #upper triangular nesting, i+1 to stop a fidelity with a density matrix on itself
            fid = FidelityMixed(raw_matrix_data[i], raw_matrix_data[j])
            ref_data.append([i,j,fid])
            all_fidelity_vals.append(fid)

        
    flat_data = []
    flat_data_tuple = []
    test_flattened = []
    for i in range(data_size):
        flat_data_tuple.append(np.ndarray.flatten(raw_matrix_data[i]))  #check these
        flat_data.append(np.ndarray.flatten(raw_matrix_data[i]))    #chek these
 
    for i in range(len(flat_data_tuple)):
        buffer_32 = []
        for k in range(len(flat_data_tuple[0])):
            x = np.real(flat_data_tuple[i][k])  #and here
            iy = np.imag(flat_data_tuple[i][k]) #and here
            buffer_32.append(x)
            buffer_32.append(iy)
        test_flattened.append(buffer_32)
        
    #check to make sure array structure has been correctly flattened
    #print(len(flat_data), len(test_flattened))
    #print(test_flattened[0][0], test_flattened[0][1], flat_data[0][0], len(test_flattened))
    #print(len(test_flattened[0]))
    dg = pd.DataFrame(test_flattened, columns = ['P_00R','P_00I','P_01R','P_01I','P_02R','P_02I','P_03R','P_03I',\
                                            'P_10R','P_10I','P_11R','P_11I','P_12R','P_12I','P_13R','P_13I',\
                                            'P_20R','P_20I','P_21R','P_21I','P_22R','P_22I','P_23R','P_23I',\
                                            'P_30R','P_30I','P_31R','P_31I','P_32R','P_32I','P_33R','P_33I'])
    filename2 = "Matrices2Q"+ "S" + str(data_size) +"#"  + str(batchnumber) + ".csv"
    dg.to_csv(filename2, index = False)
    
    return avg_fid, data_size, std_fid, all_fidelity_vals, sub_entropy_arr
    

def DataIntegrityChecker():
    #to check the data is good, want to plot average fidelity value over
    #many different batch sizes and plot the two. The graph should converge to the underlying
    #parameterisation bias
    batch_variation = []
    avg_fid_data = []
    for i in range(25):
        x, y = BatchData(i, 10*((i+1)**2))
        batch_variation.append(y)
        avg_fid_data.append(x)
        print("Progress", i, "%")
        
    plt.plot(batch_variation, avg_fid_data)
    plt.show()
        
    return avg_fid_data, batch_variation
    
def Mixed2QGen():
    '''
    Implementation of the parameterisation of SIGMA_IV in Walczak's paper on
    classification of 2Qubit states DOI 10.1007/s11128-015-1121-y.
    We generate first a mixed state according to two parameters and then transform it
    via Lorentz association to give us access to sample from the full H_4 hilbert space.
    '''
    x = np.random.random_sample() # x random between 0 and 1
    y = np.random.uniform(0, 1-x) # y random between 0 and 1-x
    
    rho_tilde = np.zeros(shape = (4,4), dtype = complex)
    rho_tilde[0,0] = y
    rho_tilde[1,1] = 0.5*(1 - y)
    rho_tilde[2,2] = 0.5*(1 - y)
    rho_tilde[1,2] = 0.5*x
    rho_tilde[2,1] = 0.5*x
    # appended the 2 parameters into rho_tilde as necessary
    #print(x)
    return rho_tilde

def TransformMatrix(density_mat):
    '''
    Method that transforms density matrices to generate new ones, note that
    these transforms are not passive, they change the state. Implements the 
    transform R_uv = M_1 SIGMA M_2* / (M_1 SIGMA M_2*)_00.
    For explanation on the contruction see http://physics.unm.edu/Courses/Finley/p581/Handouts/GenLorentzBoostTransfs.pdf
    '''
    # We first construct velocity and gamma parameters in the C = 1 normalisation
    def makevls():
        vel_x = np.random.random_sample()
        vel_y = np.random.random_sample()
        vel_z = np.random.random_sample()
        speedy = np.sqrt(vel_x ** 2 + vel_y ** 2 + vel_z **2)
        
        return vel_x, vel_y, vel_z, speedy
    
    v_x, v_y, v_z, speed = makevls()
    while (speed >= 1):
        v_x, v_y, v_z, speed = makevls()
    
    #print("SPEED", speed)
    gamma = 1/np.sqrt((1 - speed ** 2))
    #print("GAMMA", gamma)
    
    lorentz_matrix = np.zeros(shape = (4,4))
    # we now painfully append all of the upper right variables 
    lorentz_matrix[0,0] = 1 + (gamma - 1) * ( v_x ** 2)
    lorentz_matrix[1,1] = 1 + (gamma - 1) * ( v_y ** 2)
    lorentz_matrix[2,2] = 1 + (gamma - 1) * ( v_z ** 2)
    lorentz_matrix[3,3] = gamma 
    
    lorentz_matrix[0,1] = (gamma - 1) * v_x * v_y
    lorentz_matrix[0,2] = (gamma - 1) * v_x * v_z
    lorentz_matrix[0,3] = gamma * v_x
    
    lorentz_matrix[1,2] = (gamma - 1) * v_y * v_z
    lorentz_matrix[1,3] = gamma * v_y
    
    lorentz_matrix[2,3] = gamma * v_z
    
    # then fill the lower half
    lorentz_matrix = np.add(lorentz_matrix, np.transpose(lorentz_matrix))
    # Check that the input was a 4x4 matrix
    CheckShape(density_mat)
    
    buffer = np.matmul(lorentz_matrix, density_mat)
    mat_boost_unnormalised = np.matmul(buffer, np.transpose(lorentz_matrix))
    mat_boost = np.divide(mat_boost_unnormalised, np.trace(mat_boost_unnormalised))
    
    return mat_boost


def StateEntropy(mat_4x4, order):
    '''
    To work out the VN entropy S = Tr(Aln(A)) we first need a notion of the log
    of a matrix, which we can use by taking the taylor expanding it and applying an
    Nth order approcimation to it.
    '''
    if isinstance(mat_4x4, float) == True:
        log_matrix = np.zeros(shape = (4,4), dtype = float)
    else:
        log_matrix = np.zeros(shape = (4,4), dtype = complex)
        
    CheckShape(mat_4x4)
    
    identity_4x4 = np.zeros(shape = (4,4))
    for i in range(3):
        identity_4x4[i,i] = 1
    buffer_matrix = mat_4x4 - identity_4x4 # buffer matrix to let us taylor expand
    
    for i in range(1, order):
        if order == 1:
            log_matrix += np.add(log_matrix, buffer_matrix) #dodgy step?
        else: 
            matrix_power_buffer = np.linalg.matrix_power(buffer_matrix, order)
            log_matrix += ((-1) ** order) * np.add(log_matrix, matrix_power_buffer)/order
    rho_lnRho = np.matmul(mat_4x4, log_matrix)
    entropy = np.trace(rho_lnRho)
    return entropy

def FidelityMixed(matA, matB):
    '''
    Returns the Bures fidelity F(m1, m2) = Tr(sqrt(sqrt(m1)m2sqrt(m1)))
    '''
    #Matrices must be 4 by 4 or error is raised.
    if (matA.shape == (4,4)) and (matB.shape == (4,4)): #check this works with tf
        pass
    else:
        print(matA.shape, matB.shape)   #same here
        raise Exception("Density matrices are of wrong dimensions")
    
    matA_root = sqrtm(matA)
    buffer_mat = np.matmul(matA_root, matB)
    product = np.matmul(buffer_mat, matA_root)
    product_root = sqrtm(product)
    
    return np.real(np.trace(product_root) ** 2)     #LM - Seem to be picking up floating point error in the complex part

def BatchDataMixed(batchnumber, data_size, \
                   select_concurrence = [False,0 , 1] , select_entropy = [False,0,2]):
    '''
    This method makes a batch of mixed states using the Mixed2QGen method, below
    is a "logic tree" that will select states with concurrence OR entropy between
    bounded values. When the method is called, both are set to 
    default False and the matrices will have concurrences according to the 
    inherant distribution (bias) of the parameterisation of Mixed2QGen.
    
    RAISES ERROR IF BOTH SELECT_CONCURRENCE AND SELECT_ENTROPY ARE SET TO TRUE
    
    Filenames change if concurrence or entropy is controlled. I added CONC for concurrence
    and ENTR for entropy, and nothing if neither is capped.
    '''
    print("Starting Generation")
    
    raw_matrix_data = []
    #ref_data has 3 vectors for each element in the array corresponding to 2 references
    #to two density matrices in the raw_matrix_data array and their shared fidelity value
    ref_data = []
    concurrence_arr = []
    entropy_arr = []
    if (select_concurrence == True) and (select_entropy == True):
        raise Exception("Cannot select according to both entropy and concurrence, set at least one to false")
    elif (select_concurrence[2] <= select_concurrence[1]) or (select_entropy[2] <= select_entropy[1]):
        raise Exception("Invalid range arguent - range must be low to high")
    elif (len(select_concurrence) != 3) or (len(select_entropy) !=3):
        raise Exception("Invalid number of arguments, select entropy or concurrence needs a truth value, a min and a max vale.")
    else:
        pass
    
    data_counter = 0
    while (data_counter <= data_size):
        mat = TransformMatrix(Mixed2QGen()) # make a mixed state density matrix
        if select_concurrence[0] == True:
            filename = "MixedData_ref_fid" + "S" + str(data_size) + "#" + str(batchnumber) + ".csv"
            if StateConcurrence(mat) <= select_concurrence[2] and StateConcurrence(mat) >= select_concurrence[1]:   
                concurrence_arr.append(StateConcurrence(mat))
                raw_matrix_data.append(mat) # appends only matrices within specified range
                data_counter += 1
            else:
                pass # does nothing if the value is greater and the concurrence condition true
        elif select_entropy == True:
            filename = "MixedData_ref_fid" + "S" + str(data_size) + "#" + str(batchnumber) + ".csv"
            if StateEntropy(mat) >= select_entropy[1] and StateEntropy(mat) <= select_entropy[2]:
                entropy_arr.append(StateEntropy(mat, 100))
                raw_matrix_data.appen(mat) # only appends matrices within specified range
                data_counter += 1
            else:
                pass
        else: 
            # this section is for "standard" entropies and concurrences with a distribution arising from the inherant bias in parameterisation
            raw_matrix_data.append(mat)
            filename = "MixedData_ref_fid" + "S" + str(data_size) + "#" + str(batchnumber) + ".csv"
            data_counter +=1
            concurrence_arr.append(StateConcurrence(mat))
            entropy_arr.append(StateEntropy(mat, 100))
    print("Generation, Conc and Entr complete, calculating fidelities")
            
    all_fidelity_vals = []
    for i in range(data_size):
        for j in range(i+1, data_size): #upper triangular nesting, i+1 to stop a fidelity with a density matrix on itself
            fid = FidelityMixed(raw_matrix_data[i], raw_matrix_data[j])
            ref_data.append([i,j,fid])
            all_fidelity_vals.append(fid)
            
    print("Fidelities calculated")
     #Stores the ref_data 
    df = pd.DataFrame(ref_data, columns=['Density Ref1', 'Density Ref2', 'Fidelity'])
    
    df.to_csv(filename, index = False)
    #Sotres the matrix entries
    flat_data = []
    flat_data_tuple = []
    test_flattened = []
    for i in range(data_size):
        flat_data_tuple.append(np.ndarray.flatten(raw_matrix_data[i]))  #check these
        flat_data.append(np.ndarray.flatten(raw_matrix_data[i]))    #chek these
 
    for i in range(len(flat_data_tuple)):
        buffer_32 = []
        for k in range(len(flat_data_tuple[0])):
            x = np.real(flat_data_tuple[i][k])  #and here
            iy = np.imag(flat_data_tuple[i][k]) #and here
            buffer_32.append(x)
            buffer_32.append(iy)
        test_flattened.append(buffer_32)
        
    fid_sum = 0
    varience_buf = []
    for i in range(len(ref_data)):
        fid_sum += ref_data[i][2]   #check this works for tf
        varience_buf.append(ref_data[i][2])
    avg_fid = fid_sum/len(ref_data)
    std_fid = np.var(varience_buf)
    
    dg = pd.DataFrame(test_flattened, columns = ['P_00R','P_00I','P_01R','P_01I','P_02R','P_02I','P_03R','P_03I',\
                                            'P_10R','P_10I','P_11R','P_11I','P_12R','P_12I','P_13R','P_13I',\
                                            'P_20R','P_20I','P_21R','P_21I','P_22R','P_22I','P_23R','P_23I',\
                                            'P_30R','P_30I','P_31R','P_31I','P_32R','P_32I','P_33R','P_33I'])
    filename2 = "Matrices2Q"+ "S" + str(data_size) +"#"  + str(batchnumber) + ".csv"
    dg.to_csv(filename2, index = False)
    print(len(ref_data), "Reference Data Length, data is ready")
    
    return np.real(avg_fid), data_size, std_fid, all_fidelity_vals, concurrence_arr, entropy_arr

def StateConcurrence(mat_4x4):
    '''
    Method that calculates the concurrence of any 2 qubit state, returns 0 if the 
    state is pure and returns max(0, E_1 - E_2 - E_3 - E_4) where E are the eigenvalues
    in descending order of the matrix R = sqrt(sqrt(rho) rho_tilde sqrt(rho))
    and rho_tilde is worked out using pauli_y outer pauli_y rho* pauli_y outer pauli_y
    '''
    pauli_y = np.zeros(shape = (2,2), dtype = complex)
    pauli_y[0,1] = -1j
    pauli_y[1,0] = 1j
    transform = np.outer(pauli_y, pauli_y) 
    buf = np.matmul(transform, np.conj(mat_4x4)) # buffer variable for 3 matrix mult
    rho_tilde = np.matmul(buf, transform)
    root_rho = sqrtm(mat_4x4) # we now have all the matrices to make the argument in sqrt() for R
    buf2 = np.matmul(root_rho, rho_tilde) # bufer variable for 3 matrix mult
    root_arg = np.matmul(buf2, root_rho)
    big_R = sqrtm(root_arg) # now we have R we work out its eigenvalues
    
    
    try:
        eigenvalue_array = np.linalg.eigvals(big_R)
    except np.linalg.LinAlgError:
        print(big_R)
        print(mat_4x4)
        return 0
    
    
    #now we sort the eigenvalues into ascending order, taking their moduli
    np.sort_complex(eigenvalue_array)
    #print(eigenvalue_array)
    #print(eigenvalue_array, 'EIGENS', len(eigenvalue_array))
    concurrence_complex = 2 * eigenvalue_array[0] - np.sum(eigenvalue_array)
    #floating point error means it will never be 0 so we return the mixed state
    #concurrence and if it falls below a certain threshold we can call it "0"
    return np.abs(concurrence_complex)
    
def SubsystemVNE(mat_2x2, order):
    if isinstance(mat_2x2, float) == True:
        log_matrix = np.zeros(shape = (2,2), dtype = float)
    else:
        log_matrix = np.zeros(shape = (2,2), dtype = complex)
    
    identity_2x2 = np.zeros(shape = (2,2))
    identity_2x2[0,0] = 1
    identity_2x2[1,1] = 1
    buffer_matrix = mat_2x2 - identity_2x2 # buffer matrix to let us taylor expand
    
    for i in range(1, order):
        if order == 1:
            log_matrix += np.add(log_matrix, buffer_matrix) #dodgy step?
        else: 
            matrix_power_buffer = np.linalg.matrix_power(buffer_matrix, order)
            log_matrix += ((-1) ** order) * np.add(log_matrix, matrix_power_buffer)/order
    rho_lnRho = np.matmul(mat_2x2, log_matrix)
    subVNentropy = np.trace(rho_lnRho)
    return np.real(subVNentropy)
    

def CheckShape(matrix):
    
    is4x4 = False
    if (matrix.shape == (4,4)): #check this works with tf
        is4x4 = True
    else:
        print(matrix.shape)   #same here
        raise Exception("Density matrices are of wrong dimensions")
    return is4x4