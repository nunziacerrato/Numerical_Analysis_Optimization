''' docstring '''

import numpy as np
import scipy.linalg
import logging
import tenpy
import qutip

logging.basicConfig(level=logging.DEBUG)



def create_dataset(num_matr,dim_matr):
    ''' '''

    Random = np.zeros((num_matr,dim_matr,dim_matr))
    Ginibre = np.zeros((num_matr,dim_matr,dim_matr))
    GOE = np.zeros((num_matr,dim_matr,dim_matr))
    GUE = np.zeros((num_matr,dim_matr,dim_matr))
    Wishart = np.zeros((num_matr,dim_matr,dim_matr))
    # Diag_domin = np.zeros((num_matr,dim_matr,dim_matr))
    # Hilbert = np.zeros((num_matr,dim_matr,dim_matr))

    for i in range(num_matr):
        
        Random[i,:,:] = np.random.rand(dim_matr,dim_matr)
        Ginibre[i,:,:] = np.random.normal(loc=0.0, scale=1.0, size=(dim_matr,dim_matr)) + \
                         1j*np.random.normal(loc=0.0, scale=1.0, size=(dim_matr,dim_matr))
        GOE[i,:,:] = tenpy.linalg.random_matrix.GOE((dim_matr,dim_matr))
        GUE[i,:,:] = tenpy.linalg.random_matrix.GUE((dim_matr,dim_matr))
        Wishart[i,:,:] = np.array(qutip.rand_dm_ginibre((dim_matr), rank=None))
        # Hilbert[i,]
        # A = np.array(qutip.rand_dm_ginibre((dim_matr), rank=None))

    return Random, Ginibre, GOE, GUE, Wishart

def lufact(A):
    r''' This function computes the LU factorization of a non-singular (?) matrix A
        without pivoting, giving as output the matrices L and U and the growth factor g,
        here defined as :math:`\frac{ max_{ij} (|L||U|)_{ij} }{ max_{ij} (|A|)_{ij} }`.

        Paramters:
        ----------
        A : input matrix of dimension :math:`(N\times N)`

        Returns
        -------
        L : Unit lower triagular matrix
        U : Upper triangular matrix
        g : growth factor
    '''
    dim = A.shape

    # Check that the input matrix is a square matrix
    if dim[0] != dim[1]:
        logging.error("The input matrix is not a square matrix")
    if np.linalg.det(A) < np.finfo(float).eps: ### CONTROLLARE ###
        logging.error("The input matrix is singular")

    # Compute the dimension of the input square matrix
    n = dim[0]

    # Create a copy of the input matrix to be modified
    B = np.copy(A)
    for k in range(0,n-1):
        for i in range(k+1,n):
            B[i,k] = B[i,k]/B[k,k]
        for j in range(k+1,n):
            for l in range(k+1,n):
                B[l,j]=B[l,j]-B[l,k]*B[k,j]

    # Obtain the matrices L and U
    L = np.tril(B,k=-1) + np.eye(n)
    U = np.triu(B,k=0)  

    # Compute the growth factor
    LU_abs = np.abs(L)@np.abs(U)
    g = np.amax(LU_abs)/np.amax(np.abs(A))
    return L, U, g

def relative_backward_error(A,L,U):
    ''' '''
    return np.linalg.norm(A - L@U, ord=inf)/np.linalg.norm(A, ord=inf)

def stat_analysis(matr_vector):
    ''' '''
    
    number_matrices = matr_vector.shape[0]
    for i in range(number_matrices):
        # Random, Ginibre, GOE, GUE, Wishart = create_dataset(num_matr, dim_matr)
        matrix = matr_vector[i]
        L, u, g = lufact(matrix)
        rel_back_error = relative_backward_error(matrix,L,U)


    return

if __name__ == '__main__':
    
    num_matr = 100
    dim_matr = 10

    # Create the dataset
    Random, Ginibre, GOE, GUE, Wishart = create_dataset(num_matr,dim_matr)
    dataset = [Random, Ginibre, GOE, GUE, Wishart]

    # Statistics associated to the dataset
    mean, devst = stat_analysis(dataset) # ciclo for, aggiustare


    ################################################################################################
    A = np.array([[1,2,3],[4,5,6],[7,8,9]])
    # print(np.abs(A))
    # print(np.amax(np.abs(A)))
    L, U, g = lufact(A)
    print('g=',g)
    print('L=',L)
    print('U=',U)
    # print(L@U)
