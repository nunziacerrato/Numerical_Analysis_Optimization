''' '''

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


def compute_A(points,order):
    ''' '''
    m = points.shape[0]
    n = order + 1
    A = np.zeros((m,n))
    for index in range(m):
        A[index,:]=(points[index])**range(n)

    return A

def Cholesky_factorization(A, b):
    r''' This function computes the Cholesky factorization of a matrix A, and solves the linear 
        system Ax = b, giving as output the solution of the system x.

        Parameters
        ---------
        A : ndarray
            Input matrix of dimension :math:`(m\times n)`
        b : ndarray
            Input vector of dimension :math:`(m)`
        
        Returns
        ------
        x : ndarray
            Solution of the linear system
    ''' 
    A_transpose = np.transpose(A)
    C = A_transpose@A
    d = A_transpose@b

    L, low = scipy.linalg.cho_factor(C)
    x = scipy.linalg.cho_solve((L, low),d)
    return x

def QR_fact(A, b):
    r''' This function computes the QR factorization of a matrix A, and solves the linear 
        system Ax = b, giving as output the solution of the system x.

        Parameters
        ---------
        A : ndarray
            Input matrix of dimension :math:`(m\times n)`
        b : ndarray
            Input vector of dimension :math:`(m)`
        
        Returns
        ------
        x : ndarray
            Solution of the linear system
    ''' 
    Q, R = np.linalg.qr(A)
    c = np.transpose(Q) @ b 
    x = scipy.linalg.solve_triangular(R, c, lower = False)
    return x

def Compute_residual(A,b,approx_solution):
    r''' This function takes an input a matrix A, a vector b and an approximate solution to the 
        system Ax = b and computes the residual :math:`\lVert A^T b - A^T A approx_solution \rVert_2`.

        Parameters
        ---------
        A : ndarray
            Input matrix of dimension :math:`(m\times n)`
        b : ndarray
            Input vector of dimension :math:`(m)`
        approx_solution : ndarray
                          Approximate solution to the system

        
        Returns
        ------
        r : float
            Residual
    ''' 
    A_transpose = np.transpose(A)
    C = A_transpose@A
    d = A_transpose@b

    residual = d - C @ approx_solution
    residual_norm_2 = np.linalg.norm(residual, ord=2)
    return residual, residual_norm_2

if __name__ == '__main__':
    pass
