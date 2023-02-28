''' docstring '''

import numpy as np
import pandas as pd
import scipy.linalg
import logging
import tenpy
import qutip
import matplotlib.pyplot as plt

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
        out : int
              flag meaning if there has been a pivot breakdown:
              out = 0 -> no pivot breakdown;
              out = 1 -> pivot breakdown
    '''
    out = 0
    dim = A.shape

    # Check that the input matrix is a square matrix
    if dim[0] != dim[1]:
        logging.warning("The input matrix is not a square matrix")
    if np.abs(np.linalg.det(A)) < np.finfo(float).tiny: ### CONTROLLARE ###
        logging.warning("The input matrix is singular")
        # for k in range(A.shape[0]):
        #     print(f'k={k}, det(A_{k}) = {np.linalg.det(A[:k+1,:k+1])}')

    # Compute the dimension of the input square matrix
    n = dim[0]

    # Create a copy of the input matrix to be modified
    B = np.copy(A)
    for k in range(0,n-1):
        for i in range(k+1,n):
            B_kk = B[k,k]
            if np.abs(B_kk) < np.finfo(float).tiny:
                out = 1
                logging.warning(f'Division by zero - B_kk = {B_kk}, B_kk_abs = {np.abs(B_kk)}')
            B[i,k] = B[i,k]/B_kk
        for j in range(k+1,n):
            for l in range(k+1,n):
                B[l,j]=B[l,j]-B[l,k]*B[k,j]

    # Obtain the matrices L and U
    L = np.tril(B,k=-1) + np.eye(n)
    U = np.triu(B,k=0)  

    # Compute the growth factor
    LU_abs = np.abs(L)@np.abs(U)
    g = np.amax(LU_abs)/np.amax(np.abs(A))
    return L, U, g, out

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
    ''' This function returns a diagonally dominant matrix of dimension :math:`(N\times N)`
        
        Parameters
        ----------
        N : int
            Dimension of the output matrix

        Returns
        -------
        out : ndarray
              Diagonally dominant matrix
    
    '''

    diag_sign = np.random.rand(N)
    diag_sign = np.sign(diag_sign - 0.5)
    diag_sign[diag_sign == 0] = 1
    M = np.random.normal(loc=0.0, scale=1.0, size=(N,N))
    for i in range(N):
        M[i,i] = sum(np.abs(M[i,:])) * diag_sign[i]
    return M

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
    
    # Set the seeds to have reproducibility of the results
    np.random.seed(1)


    Random = np.zeros((num_matr,dim_matr,dim_matr))
    Ginibre = np.zeros((num_matr,dim_matr,dim_matr), dtype=complex)
    GOE = np.zeros((num_matr,dim_matr,dim_matr))
    GUE = np.zeros((num_matr,dim_matr,dim_matr), dtype=complex)
    Wishart = np.zeros((num_matr,dim_matr,dim_matr), dtype=complex)
    Diag_dom = np.zeros((num_matr,dim_matr,dim_matr))

    dataset = {'Random':Random, 'Ginibre':Ginibre, 'GOE':GOE, 'GUE':GUE, 'Wishart':Wishart,\
               'Diag_dom':Diag_dom}
    
    # Random matrices
    i = 0
    while i < num_matr:  
        matrix = np.random.rand(dim_matr,dim_matr)
        if np.abs(np.linalg.det(matrix)) < np.finfo(float).tiny:
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
        if np.abs(np.linalg.det(matrix)) < np.finfo(float).tiny:
            pass
        else:
            dataset['Ginibre'][i,:,:] = matrix
            i = i + 1
    logging.info('Ginibre matrices generated')

    # GOE matrices 
    i = 0
    while i < num_matr:  
        matrix = tenpy.linalg.random_matrix.GOE((dim_matr,dim_matr))
        if np.abs(np.linalg.det(matrix)) < np.finfo(float).tiny:
            pass
        else:
            dataset['GOE'][i,:,:] = matrix
            i = i + 1
    logging.info('GOE matrices generated')

    # GUE matrices
    i = 0
    while i < num_matr:  
        matrix = tenpy.linalg.random_matrix.GUE((dim_matr,dim_matr))
        if np.abs(np.linalg.det(matrix)) < np.finfo(float).tiny:
            pass
        else:
            dataset['GUE'][i,:,:] = matrix
            i = i + 1
    logging.info('GUE matrices generated')

    # Wishart matrices
    i = 0
    while i < num_matr:  
        matrix = np.array(qutip.rand_dm_ginibre((dim_matr), rank=None))
        if np.abs(np.linalg.det(matrix)) < np.finfo(float).tiny:
            pass
        else:
            dataset['Wishart'][i,:,:] = matrix
            i = i + 1
    logging.info('Wishart matrices generated')

    # Diagonally dominant matrices
    i = 0
    while i < num_matr:  
        matrix = diagonally_dominant_matrix(dim_matr)
        if np.abs(np.linalg.det(matrix)) < np.finfo(float).tiny:
            pass
        else:
            dataset['Diag_dom'][i,:,:] = matrix
            i = i + 1
    logging.info('Diagonally dominant matrices generated')


    return dataset


#################################### SAVE DATA IN EXCEL FILES ####################################

if __name__ == '__main__' :

    if True:
        num_matr = 500
        dim_matr_max = 20

        common_path = "C:\\Users\\cerra\\Documents\\GitHub\\Numerical_Analysis_Optimization\\Project_1"

        for dim_matr in range(2,dim_matr_max+1):
            dataset = create_dataset(num_matr, dim_matr)

            keys = dataset.keys()

            g_dict = {}
            rel_back_err_dict = {}
            for key in keys:
                g_dict[key] = np.zeros(num_matr)
                rel_back_err_dict[key] = np.zeros(num_matr)


            for matrix_type in keys:
                
                for i in range(num_matr):
                    A = dataset[matrix_type][i,:,:]
                    L, U, g_dict[matrix_type][i], out = lufact(A)
                    rel_back_err_dict[matrix_type][i] = relative_backward_error(A, L, U)
            
            df_g = pd.DataFrame(data = g_dict)
            df_rel_back_err = pd.DataFrame(data = rel_back_err_dict)


            writer = pd.ExcelWriter(f'{common_path}\\Data\\'
                                    f'Statistics_for_{num_matr}_matrices_of_dim_{dim_matr}.xlsx')
            df_g.to_excel(writer, 'growth_factor', index = False)
            df_rel_back_err.to_excel(writer, 'rel_back_err', index = False)
            writer.save()


        # grafico di media, min, max, varianza di growth factor e di rel_back_err per ogni 
        # tipo di matrice al variare di N - saranno punti da fittare eventualmente 
        # box plot per i diversi tipi di matrice di growth factor e rel_back_err










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



