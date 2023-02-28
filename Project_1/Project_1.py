''' docstring '''

import numpy as np
import scipy.linalg
import logging
import tenpy
import qutip
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


def create_dataset(num_matr,dim_matr):
    ''' This function creates the dataset, taking as input the number of matrices of each type
        and their relative dimension.

        Parameters
        ----------
        num_matr : int
                    Number of matrices for each type
        dim matr : int
                    Dimension of the (square) matrices

        Returns
        -------
        out : dictionary
              Dictionary whose keys represent the different types of matrices considered. Each value
              of the dictionary is an array of shape (num_matr,dim_matr,dim_matr).
    '''
    # out : list
    #     List of arrays of shape (num_matr,dim_matr,dim_matr). Each array, element of the list,
    #     represents a different set of num_matr matrices of dimension (dim_matr,dim_matr).

    
    # Set the seeds to have reproducibility of the results
    np.random.seed(1)


    Random = np.zeros((num_matr,dim_matr,dim_matr))
    Ginibre = np.zeros((num_matr,dim_matr,dim_matr), dtype=complex)
    GOE = np.zeros((num_matr,dim_matr,dim_matr))
    GUE = np.zeros((num_matr,dim_matr,dim_matr), dtype=complex)
    # Wishart = np.zeros((num_matr,dim_matr,dim_matr), dtype=complex)
    # Diag_domin = np.zeros((num_matr,dim_matr,dim_matr))
    # Hilbert = np.zeros((num_matr,dim_matr,dim_matr))

    dataset = {'Random':Random, 'Ginibre':Ginibre, 'GOE':GOE, 'GUE':GUE}
    
    # Random matrices
    i = 0
    while i < num_matr:  
        matrix = np.random.rand(dim_matr,dim_matr)
        if np.linalg.det(matrix) < np.finfo(float).eps:
            pass
        else:
            dataset['Random'][i,:,:] = matrix
            i = i + 1
    logging.info('Random matrices generated')

    # Ginibre matrices
    i = 0
    while i < num_matr:  
        matrix = np.random.normal(loc=0.0, scale=1.0, size=(dim_matr,dim_matr)) + \
                         1j*np.random.normal(loc=0.0, scale=1.0, size=(dim_matr,dim_matr))
        if np.linalg.det(matrix) < np.finfo(float).eps:
            pass
        else:
            dataset['Ginibre'][i,:,:] = matrix
            i = i + 1
    logging.info('Ginibre matrices generated')

    # GOE matrices 
    i = 0
    while i < num_matr:  
        matrix = tenpy.linalg.random_matrix.GOE((dim_matr,dim_matr))
        if np.linalg.det(matrix) < np.finfo(float).eps:
            pass
        else:
            dataset['GOE'][i,:,:] = matrix
            i = i + 1
    logging.info('GOE matrices generated')

    # GUE matrices
    i = 0
    while i < num_matr:  
        matrix = tenpy.linalg.random_matrix.GUE((dim_matr,dim_matr))
        if np.linalg.det(matrix) < np.finfo(float).eps:
            pass
        else:
            dataset['GUE'][i,:,:] = matrix
            i = i + 1
    logging.info('GUE matrices generated')

    # Wishart matrices
    # i = 0
    # j = 0
    # while i < num_matr:  
    #     matrix = np.array(qutip.rand_dm_ginibre((dim_matr), rank=None))
    #     if np.linalg.det(matrix) < np.finfo(float).eps:
    #         j = j + 1
    #         # print(f'failed ={j}')
    #         # pass
    #     else:
    #         dataset['Wishart'][i,:,:] = matrix
    #         i = i + 1
    #         print(f'success = {i}')

    # logging.info('Wishart matrices generated')

    # for i in range(num_matr):
        
        # dataset['Random'][i,:,:] = np.random.rand(dim_matr,dim_matr)
        # dataset['Ginibre'][i,:,:] = np.random.normal(loc=0.0, scale=1.0, size=(dim_matr,dim_matr)) + \
        #                  1j*np.random.normal(loc=0.0, scale=1.0, size=(dim_matr,dim_matr))
        # dataset['GOE'][i,:,:] = tenpy.linalg.random_matrix.GOE((dim_matr,dim_matr))
        # dataset['GUE'][i,:,:] = tenpy.linalg.random_matrix.GUE((dim_matr,dim_matr))
        # dataset['Wishart'][i,:,:] = np.array(qutip.rand_dm_ginibre((dim_matr), rank=None))
        # Hilbert[i,]
        # A = np.array(qutip.rand_dm_ginibre((dim_matr), rank=None))

    return dataset

def lufact(A):
    r''' This function computes the LU factorization of a non-singular (?) matrix A
        without pivoting, giving as output the matrices L and U and the growth factor g,
        here defined as :math:`\frac{ max_{ij} (|L||U|)_{ij} }{ max_{ij} (|A|)_{ij} }`.

        Paramters:
        ----------
        A : ndarray
            input matrix of dimension :math:`(N\times N)`

        Returns
        -------
        L : ndarray
            Unit lower triagular matrix
        U : ndarray
            Upper triangular matrix
        g : ndarray
            growth factor
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
    r''' This function computes the relative backward error of the LU factorization, defined as
         :math:`\frac{\lVert A -LU \rVert_{\infty}}{\lVert A \rVert_{\infty}}`
    
        Parameters
        ----------
        A : ndarray
            Input matrix
        L : ndarray
            Unit lower triangular matrix, obtained from the LU factorization of the input matrix A.
        U : ndarray
            Upper triangular matrix, obtained from the LU factorization of the input matrix A.

        Returns
        -------
        out : float
            Relative backward error
    '''

    return np.linalg.norm(A - L@U, ord=np.inf)/np.linalg.norm(A, ord=np.inf)


if __name__ == '__main__':
    
    num_matr = 10
    dim_matr = 100

    # Create the dataset
    dataset = create_dataset(num_matr,dim_matr)

    # Create dictionaries that will contain the growth factor and the relative backward error
    # for each matrix of the current matrix type considered.
    g_dict = {}
    rel_back_err_dict = {}
    lufact_failed = {}
    fig = plt.figure()
    data_g = []
    for key in dataset.keys():
        g_dict[key] = np.empty(num_matr)
        rel_back_err_dict[key] = np.empty(num_matr)
        lufact_failed[key] = 0

        # For each matrix type considered, compute the LU factorization, with the growth factor,
        # and the relative backward error, and save these values in the appropriate dictionary.
        matrix_type = dataset[key]
        for i in range(num_matr):
            A = matrix_type[i,:,:]
            try:
                L, U, g = lufact(A)
            except Exception as e:
                lufact_failed[key] = lufact_failed[key] + 1
                logging.error("Exception occurred", exc_info=True)
                continue
            g_dict[key][i] = g
            rel_back_err_dict[key][i] = relative_backward_error(A, L, U)
        data_g.append(g_dict[key])
    plt.boxplot(data_g)

    plt.show()
    print(f'lufact_failed={lufact_failed}')
    # print(f'g_dict={g_dict}')
    # print(f'rel_back_err_dict={rel_back_err_dict}')        
                
    
    

    ################################################################################################
    # A = np.array([[1,2,3],[4,5,6],[7,8,9]])
    # # print(np.abs(A))
    # # print(np.amax(np.abs(A)))
    # L, U, g = lufact(A)
    # print('g=',g)
    # print('L=',L)
    # print('U=',U)
    # # print(L@U)
