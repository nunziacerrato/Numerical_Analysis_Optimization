''' The program Project_1 serves as a library. It contains all the basic functions needed to solve
    the first Project of the "Numerical Analysis and Optimization" course. In particular, this file
    contains the functions used to create a dataset of square matrices of different types, and the
    functions used to compute the LU factorization and the relative backward error associated to it.
    It also contains functions used to create the Wilkinson matrix of a chosen dimension, and to
    compute the LU factorization, investigating when the algorithm breaks.
'''

import numpy as np
import pandas as pd
import scipy.linalg
import logging
import tenpy
import qutip

# Define global parameters
num_matr = 500
dim_matr_max = 50
common_path = "Project_1"

def lufact(A):
    r''' This function computes the LU factorization of a square matrix A
        without pivoting, giving as output the matrices L and U and the growth factor g,
        here defined as :math:`\frac{ max_{ij} (|L||U|)_{ij} }{ max_{ij} (|A|)_{ij} }`.

        Parameters
        ---------
        A : ndarray
            Input matrix of dimension :math:`(n\times n)`

        Returns
        ------
        L : ndarray
            Unit lower triagular matrix
        U : ndarray
            Upper triangular matrix
        g : float
            Growth factor
    '''

    # Compute the dimension of the input square matrix
    dim = A.shape
    n = dim[0]

    # Define the chosen precision
    precision = np.finfo(float).eps/2

    # Check that the input matrix is a square matrix
    assert (dim[0] == dim[1]), "The input matrix is not a square matrix"

    # Check if the determinant of the input matrix is less than the chosen precision
    if np.abs(np.linalg.det(A)) < precision:
        logging.warning("The determinant of the input matrix is less than the chosen precision")
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
    r''' This function computes the relative backward error :math:`\delta` of the LU factorization,
    defined as :math:`\delta = \frac{\lVert A -LU \rVert_{\infty}}{\lVert A \rVert_{\infty}}`.

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

def diagonally_dominant_matrix(n):
    r''' This function returns a diagonally dominant matrix of dimension :math:`(n \times n)`, whose
        non-diagonal entries are normally distributed.
        
        Parameters
        ----------
        n : int
            Dimension of the output matrix

        Returns
        -------
        out : ndarray
              Diagonally dominant matrix
    '''
    # The following steps are made to decide the sign of the diagonal element of the output matrix
    # Obtain n random numbers in [0,1) and apply the sign function to this values, shifted by 0.5
    diag_sign = np.random.rand(n)
    diag_sign = np.sign(diag_sign - 0.5)
    diag_sign[diag_sign == 0] = 1 # Set to 1 the (vary improbable) values equal to 0
    
    # Obtain a matrix of dimension (nxn) whose entries are normally distributed
    M = np.random.normal(loc=0.0, scale=1.0, size=(n,n))
    # Substitute all the diagonal elements in this matrix with the sum of the absolute values of all
    # the elements in the corresponding row
    for i in range(n):
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
    i = 0
    while i < num_matr:  
        matrix = np.array(qutip.rand_dm_ginibre((dim_matr), rank=None))
        # cond_numb = np.linalg.norm(matrix, ord = 2)*np.linalg.norm(np.linalg.inv(matrix), ord = 2)
        # if cond_numb > 1/precision_zero:
        #     print(f'i = {i} 1/cond_numb = {1/cond_numb}')
        #     pass
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

####################################################################################################
############################################ PROBLEM 2 #############################################

def wilkin(n):
    r''' This function computes the Wilkinson matrix of dimension :math:`(n \times n)`.

        Parameters
        -----------
        n : int
            Dimension of the Wilkinson matrix
        
        Returns
        --------
        W : ndarray
            Wilkinson matrix
    '''
    W = np.tril(-np.ones((n,n)),k=-1) + np.eye(n)
    W[:,n-1] = 1
    return W

def check_when_lufact_W_fails(n_max = 60, threshold = np.finfo(float).eps):
    r''' This function checks the failures of GEPP for a Wilkinson matrix W_n with dimension less or
        equal to n_max. We define a failure as a case in which the relative error between the 
        solution found by the algorithm and the expected solution is higher than a chosen threshold. 
        The default threshold is set equal to the machine epsilon. When an error is found, a warning 
        message is printed. The function returns the list of the dimensions less than or equal to
        n_max for which the GEPP algorithm fails.
        
        Parameters
        -----------
        n_max : int
                Maximum dimension of the Wilkinson matrix

        Returns
        --------
        fails : list
                List of dimensions for which the algorithm fails
    '''

    fails = []

    # Cycle on the dimension of the input
    for n in range(2,n_max+1):
        W = wilkin(n)
        logging.debug(f'W = {W}')

        # Define the vector b
        vect = np.ones(n)
        b = W @ vect

        # Solve the system with GEPP
        x = np.linalg.solve(W,b)

        # Compute the error in 1-norm between the computed solution and the exact solution,
        # and print a warning message when the erroor exceeds the chosen precision
        error = sum(np.abs(x-vect))
        if error <= threshold:
            logging.info(f'n = {n}, ||x - e||_1 = {error}')
        elif error > threshold:
            logging.warning(f'n = {n}, ||x - e||_1 = {error}')
            fails.append(n)
            # Cycle on the elements of the computed solution and print a warning message with the
            # wrong elements of the solution
            for i in range(n):
                x_i = x[i]
                if abs(x_i-1) > threshold:
                    logging.warning(f'x[{i}] = {x[i]}')
    return fails

def expected_LU_wilkin(n):
    r''' This function generates the exact LU factorization of the Wilkinson matrix of dimension 
        :math:`(n \times n)`.

        Parameters
        -----------
        n : int
            Dimension of the Wilkinson matrix

        Returns
        --------
        L : ndarray
            Unit lower triangular matrix
        U : ndarray
            Upper triangular matrix
    '''
    L = np.tril(-np.ones((n,n)),k=-1) + np.eye(n)
    U = np.eye(n) 
    U[:,n-1] = 2**np.array(range(n))
    return L, U

def compute_error_lufact_W(n):
    r''' This function compares the exact LU factorization of the Wilkinson matrix of dimension
        :math:`(n \times n)` with the LU factorization computed with the function lufact.

        Parameters
        -----------
        n : int
            Dimension of the Wilkinson matrix

        Returns
        --------
        err_L : float
                Error on the unit lower triangular matrix L
        err_U : float
                Error on the upper triangular matrix U
    '''
    
    # Generate the Wilkinson matrix of dimension n and compute the LU factorization
    W = wilkin(n)
    L, U, g = lufact(W)

    # Generate the matrices L and U in their expected form
    expected_L, expected_U = expected_LU_wilkin(n)

    # Compute the error on L and U as the maximum value of the modulus of the difference
    # between the computed and the expected L, U
    err_L = np.amax(np.abs(L-expected_L))
    err_U = np.amax(np.abs(U-expected_U))
    logging.info(f'error on L = {err_L}')
    logging.info(f'error on U = {err_U}')
    return err_L, err_U

def step_by_step_GEPP_W(n):
    r'''This function performs backward and forward substitutions to investigate when the GEPP 
        algorithm fails. When an error is found, a warning message is printed.
        The function returns a flag indicating if the solution of the system is exact or if there
        has been an error.

        Parameters
        -----------
        n : int
            Dimension of the Wilkinson matrix

        Returns
        --------
        out : int
              Error flag   
              0 : The solution of the system is exact
              1 : Error in the solution of the system
    '''

    out = 0

    # Define the machine error
    epsilon = np.finfo(float).eps

    # Define the Wilkinson matrix and perform the LU factorization
    W = wilkin(n)
    L, U, g = lufact(W)        
    vect = np.ones(n)
    b = W @ vect

    # Perform forward substitutions
    y = np.zeros(n)
    logging.info('We are solving Ly=b')
    for i in range(n):
        y[i] = b[i]
        logging.info(f'Compute y[{i}] = b^{i}_{i} = {y[i]}')
        # Check if the addition of 1 to 2^{i} has succeeded
        if (y[i] - 2**i) < epsilon and i < n-1:
            logging.warning(f'Expected 1: y[{i}] - 2^{i} = {y[i] - 2**i}')
        logging.info(f'Update b: b^{i+1} = b^{i} - y[{i}]*L[:,{i}] = {b} - {y[i]}*{L[:,i]}')
        
        b = b - y[i]*L[:,i]

    # Perform backward substitutions
    x = np.zeros(n)
    logging.info('We are solving Ux=y')
    for i in range(n):        
        x[n-1-i] = y[n-1-i]/U[n-1-i,n-1-i]        
        if np.abs(x[n-1-i] - 1) < epsilon:
            logging.info(f'Compute x[{n-1-i}] = y^{i}_{n-1-i}/U[{n-1-i},{n-1-i}] \
                           = {y[n-1-i]}/{U[n-1-i,n-1-i]} = {x[n-1-i]}')
        else:
            # Check if the solution is not a vector with all entries equal to one
            logging.warning(f'Compute x[{n-1-i}] = y^{i}_{n-1-i}/U[{n-1-i},{n-1-i}] \
                              = {y[n-1-i]}/{U[n-1-i,n-1-i]} = {x[n-1-i]}')
            out = 1
        logging.info(f'Update y: y^{i+1} = y^{i} - x[{n-1-i}]*U[:,{n-1-i}] \
                       = {y} - {x[n-1-i]}*{U[:,n-1-i]}')
        y = y - x[n-1-i]*U[:,n-1-i]
    
    return out

########################################à  MAIN PROGRAM  #########################################
#################################### SAVE DATA IN EXCEL FILES ####################################

if __name__ == '__main__' :

    logging.basicConfig(level=logging.ERROR)

    keys = create_dataset(1,2).keys()

    # Define a DataFrame to store all the failures of the LU factorization divided by matrx types.
    df_fails = pd.DataFrame(0, columns = keys, index = range(2,dim_matr_max+1))
    
    # Cycle on the different dimensions considered
    for dim_matr in range(2,dim_matr_max+1):
        logging.info(f'Dimension = {dim_matr}')

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
