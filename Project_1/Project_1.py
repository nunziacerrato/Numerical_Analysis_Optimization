''' This program serves as a library. It contains all the basic functions needed to solve the first
    Project of the "Numerical Analysis and Optimization" course. In particular, this file contains
    the functions used to create a dataset of square matrices of different types, and the functions
    used to compute the LU factorization and the relative backward error associated to it. '''

import numpy as np
import pandas as pd
import scipy.linalg
import logging
import tenpy
import qutip
import matplotlib.pyplot as plt

# Define global parameters
num_matr = 500
dim_matr_max = 50
common_path = "C:\\Users\\cerra\\Documents\\GitHub\\Numerical_Analysis_Optimization\\Project_1"

def lufact(A):
    r''' This function computes the LU factorization of a non-singular matrix A
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

    # Compute the dimension of the input square matrix
    dim = A.shape
    n = dim[0]

    # Define the chosen precision
    precision = np.finfo(float).eps

    # Check that the input matrix is a square matrix
    assert (dim[0] == dim[1]), "The input matrix is not a square matrix"

    # Check if the input matrix is singular
    if np.abs(np.linalg.det(A)) < precision:
        logging.warning("The input matrix is singular")
    # Check if the hypothesis of the LU factorization theorem hold
    for k in range(n):
        if np.abs(np.linalg.det(A[:k+1,:k+1])) < precision:
            logging.warning(f'The {k}-th principal minor is less than the chosen precision')

    # Create a copy of the input matrix to be modified in order to obatin the matrices L and U
    B = np.copy(A)
    for k in range(0,n-1):
        for i in range(k+1,n):
            B_kk = B[k,k]
            # Check if there is a division by a quantity smaller than the chosen precision
            if np.abs(B_kk) < precision:
                raise ValueError('Division by a quantity smaller than the chosen precision'\
                                f' - B_kk = {B_kk}')
            B[i,k] = B[i,k]/B_kk
        for j in range(k+1,n):
            for l in range(k+1,n):
                B[l,j] = B[l,j] - B[l,k]*B[k,j]

    # Extract the matrices L and U from B using, resepctively, a strictly lower triangular mask
    # and an upper triangular mask.
    L = np.tril(B,k=-1) + np.eye(n) # Add the Id matrix in order for L to be unit lower triangular
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

def diagonally_dominant_matrix(N):
    ''' This function returns a diagonally dominant matrix of dimension :math:`(N\times N)`, whose
        non-diagonal entries are normally distributed.
        
        Parameters
        ----------
        N : int
            Dimension of the output matrix

        Returns
        -------
        out : ndarray
              Diagonally dominant matrix
    
    '''
    # The following steps are made to decide the sign of the diagonal element of the output matrix
    # Obtain N random numbers in [0,1) and apply the sign function to this values, shifted by 0.5
    diag_sign = np.random.rand(N)
    diag_sign = np.sign(diag_sign - 0.5)
    diag_sign[diag_sign == 0] = 1 # Set to 1 the (vary improbable) values equal to 0
    
    # Obtain a matrix of dimension (NxN) whose entries are normally distributed
    M = np.random.normal(loc=0.0, scale=1.0, size=(N,N))
    # Substitute all the diagonal elements in this matrix with the sum of the absolute values of all
    # the elements in the corresponding row
    for i in range(N):
        M[i,i] = sum(np.abs(M[i,:])) * diag_sign[i]
    
    return M

def create_dataset(num_matr,dim_matr):
    ''' This function creates the dataset, taking as input the number of matrices of each type
        and their relative dimension and giving as output a dictionary whose keys represent the 
        different types of matrices considered and whose values are 3-dimensional arrays, where the
        first index cycles on the number of matrices considered.
        The output matrices are chosen to be nonsingular.

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
    
    # Define the minimum value of the determinant of the dataset matrices
    precision_zero = np.finfo(float).tiny
    
    # Set the seeds to have reproducibility of the results
    np.random.seed(1)

    # Create arrays to store the final matrices
    Random = np.zeros((num_matr,dim_matr,dim_matr))
    Ginibre = np.zeros((num_matr,dim_matr,dim_matr), dtype=complex)
    CUE = np.zeros((num_matr,dim_matr,dim_matr), dtype=complex)
    GUE = np.zeros((num_matr,dim_matr,dim_matr), dtype=complex)
    Wishart = np.zeros((num_matr,dim_matr,dim_matr), dtype=complex)
    Diag_dom = np.zeros((num_matr,dim_matr,dim_matr))

    # Define a dictionary to keep track of the types of matrices chosen
    dataset = {'Random':Random, 'Ginibre':Ginibre, 'CUE':CUE, 'GUE':GUE, 'Wishart':Wishart,\
               'Diagonally dominant':Diag_dom}
    
    ### Compare the backward stability of the two following matrix types-->should they be both real?
    # Random matrices: matrices whose entries are in [0,1)
    i = 0
    while i < num_matr:  
        matrix = np.random.rand(dim_matr,dim_matr)
        if np.abs(np.linalg.det(matrix)) < precision_zero:
            pass
        else:
            dataset['Random'][i,:,:] = matrix
            i = i + 1
    logging.info('Random matrices generated')

    # Ginibre matrices: matrices whose entries are independent, complex, and normally distributed
    i = 0
    while i < num_matr:  
        matrix = np.random.normal(loc=0.0, scale=1.0, size=(dim_matr,dim_matr)) + \
                         1j*np.random.normal(loc=0.0, scale=1.0, size=(dim_matr,dim_matr))
        if np.abs(np.linalg.det(matrix)) < precision_zero:
            pass
        else:
            dataset['Ginibre'][i,:,:] = matrix
            i = i + 1
    logging.info('Ginibre matrices generated')


    # CUE matrices: Unitary matrices sampled from the Circular Unitary Ensemble
    for i in range(num_matr):  
        matrix = tenpy.linalg.random_matrix.CUE((dim_matr,dim_matr))
        dataset['CUE'][i,:,:] = matrix
    logging.info('CUE matrices generated')


    # GUE matrices: Complex Hermitian matrices sampled from the Gaussian Unitary Ensemble
    ####  Real eigenvalues but not necessarily positive: are they backward stable or not?
    i = 0
    while i < num_matr:  
        matrix = tenpy.linalg.random_matrix.GUE((dim_matr,dim_matr))
        if np.abs(np.linalg.det(matrix)) < precision_zero:
            pass
        else:
            dataset['GUE'][i,:,:] = matrix
            i = i + 1
    logging.info('GUE matrices generated')

    # Wishart matrices: matrices of the form A^{\dagger}A, with A sampled from the Ginibre Ensemble.
    # This choice ensures the matrices to be positive semidefinite. Discarding the singular matrices
    # we obtain positive definite matrices.
    #### Real AND positive eigenvalues: they are supposed to be backward stable.
    i = 0
    while i < num_matr:  
        matrix = np.array(qutip.rand_dm_ginibre((dim_matr), rank=None))
        if np.abs(np.linalg.det(matrix)) < precision_zero:
            pass
        else:
            dataset['Wishart'][i,:,:] = matrix
            i = i + 1
    logging.info('Wishart matrices generated')

    # Diagonally dominant matrices: matrices whose diagonal entries are, in modulus, greater or
    # equal to the sum of the absolute values of the entries in the corresponding row.
    i = 0
    while i < num_matr:  
        matrix = diagonally_dominant_matrix(dim_matr)
        if np.abs(np.linalg.det(matrix)) < precision_zero:
            pass
        else:
            dataset['Diagonally dominant'][i,:,:] = matrix
            i = i + 1
    logging.info('Diagonally dominant matrices generated')


    return dataset


#################################### SAVE DATA IN EXCEL FILES ####################################

if __name__ == '__main__' :

    dim_matr_max = 50
    keys = create_dataset(1,2).keys()

    # logging.basicConfig(level=logging.INFO)

    # Define a DataFrame to store all the failures of the LU factorization divided by matrx types.
    df_fails = pd.DataFrame(0, columns = keys, index = range(2,dim_matr_max+1))
    
    # Cycle on the different dimensions considered
    for dim_matr in range(2,dim_matr_max+1):
        # print(f'Dimension = {dim_matr}')

        # Create the dataset
        dataset = create_dataset(num_matr, dim_matr)

        # Create DataFrames in which the growth factor and the relative backward error are stored
        df_g = pd.DataFrame(columns = keys)
        df_rel_back_err = pd.DataFrame(columns = keys)

        # Cycle on the different types of matrices considered
        for matrix_type in keys:

            # Cycle on the number of matrices of each type 
            for i in range(num_matr):
                # Select the matrix and compute the LU factorization, the growth factor,
                # and the relative backward error
                A = dataset[matrix_type][i,:,:]
                try:
                    L, U, df_g.at[i,matrix_type] = lufact(A)
                    df_rel_back_err.at[i,matrix_type] = relative_backward_error(A, L, U)
                except ValueError:
                    df_fails.at[dim_matr,matrix_type] = df_fails.at[dim_matr,matrix_type] + 1
        
        # Save the growth factor and the relative backward error in Excel files
        writer = pd.ExcelWriter(f'{common_path}\\Data\\'
                                    f'Statistics_for_{num_matr}_matrices_of_dim_{dim_matr}.xlsx')
        df_g.to_excel(writer, 'growth_factor', index = False)
        df_rel_back_err.to_excel(writer, 'rel_back_err', index = False)
        writer.save()
 
    # Save the failues of the LU factorization in an Excel file
    writer = pd.ExcelWriter(f'{common_path}\\Data\\'
                                f'Failures_LUfact_for_{num_matr}_matrices.xlsx')
    df_fails.to_excel(writer, 'Fails', index = False)
    writer.save()





        # for dim_matr in range(dim_matr_max):
        #     diag_dom_matrices = np.zeros((num_matr,dim_matr,dim_matr))
        #     for i in range(num_matr):
        #         diag_dom_matrices[i,:,:] = diagonally_dominant_matrix(dim_matr)

        #     g_diag_dom = np.zeros(num_matr)
        #     rel_back_err_diag_dom = np.zeros(num_matr)
        #     for i in range(num_matr):
        #         A = diag_dom_matrices[i,:,:]
        #         L, U, g_diag_dom[i], out = lufact(A)
        #         rel_back_err_diag_dom[i] = relative_backward_error(A, L, U)


    # CHECK THE DETERMINANT OF THE HILBERT MATRICES
    # n_max = 100
    # Hilbert_matr_vect = [scipy.linalg.hilbert(n) for n in range(1,n_max+1)]
    # for i in range(len(Hilbert_matr_vect)):
    #     print(f'n = {i+1}, det(H_{i+1}) = {np.linalg.det(Hilbert_matr_vect[i])}')

    ########################### don't know what this other if does ################################
    if False:
        logging.basicConfig(level=logging.ERROR)
        n_max = 1000
        Hilbert_matr_vect = [scipy.linalg.hilbert(n) for n in range(1,n_max+1)]
        for i in range(len(Hilbert_matr_vect)):
            L, U, g, out = lufact(Hilbert_matr_vect[i])
            rel_back_err = relative_backward_error(Hilbert_matr_vect[i], L, U)
            print(f'n={i+1}, relative_back_err = {rel_back_err}, growth_factor = {g}')
            if out == 1:
                print(f'n={i+1}, out = {out}')
                ### Check on the matrix elements ###
                # u_nn = np.linalg.det(Hilbert_matr_vect[i])/np.linalg.det(Hilbert_matr_vect[i-1])
                # print(f'pivot for n = {i+1}: {u_nn}')
                # print(f'pivot for n = {i}: {np.linalg.det(Hilbert_matr_vect[i-1])/np.linalg.det(Hilbert_matr_vect[i-2])}')
                # print(f'pivot for n = {i-1}: {np.linalg.det(Hilbert_matr_vect[i-2])/np.linalg.det(Hilbert_matr_vect[i-3])}')
                # print(f'pivot for n = {i-2}: {np.linalg.det(Hilbert_matr_vect[i-3])/np.linalg.det(Hilbert_matr_vect[i-4])}')
                # print(f'U[n,n] = {U[i,i]}')
                break


                
        # for A in Hilbert_matr_vect:
        #     L, U, g, out = lufact(A)
        #     print(out)



