''' '''

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


def compute_A(points,order):
    r'''This function computes the Vandermonde matrix A taking as inputs an array, cointaining
        the coordinates of the points, and an integer representing the order of the polynomial.

        Parameters
        ---------
        points : ndarray
                 Input points
        order : int
                Order of the polynomial
        
        Returns
        ------
        A : ndarray
            Vandermonde matrix
    ''' 
    m = points.shape[0]
    n = order + 1
    A = np.zeros((m,n))
    for index in range(m):
        A[index,:]=(points[index])**range(n)

    return A

def Least_Square_Cholesky(A, b):
    r'''This function computes the Cholesky factorization of a matrix :math:`(A^{T}A)`,
        and solves the linear system :math:`(A^{T}Ax = A^{T}b)`, giving as output the solution
        :math:`x` of the system.

        Parameters
        ----------
        A : ndarray
            Input matrix of dimension :math:`(m\times n)`
        b : ndarray
            Input vector of dimension :math:`(m)`

        Returns
        -------
        x : ndarray
            Solution of the linear system
    ''' 
    A_transpose = np.transpose(A)
    C = A_transpose@A
    d = A_transpose@b

    L, low = scipy.linalg.cho_factor(C)
    x = scipy.linalg.cho_solve((L, low),d)
    return x

def Least_Square_QR(A, b):
    r'''This function computes the QR factorization of a matrix :math:`A`, and solves the linear
        system :math:`Ax = b`, giving as output the solution of the system.

        Parameters
        ----------
        A : ndarray
            Input matrix of dimension :math:`(m\times n)`
        b : ndarray
            Input vector of dimension :math:`(m)`

        Returns
        -------
        x : ndarray
            Solution of the linear system
    ''' 
    Q, R = np.linalg.qr(A)
    c = np.transpose(Q) @ b 
    x = scipy.linalg.solve_triangular(R, c, lower = False)
    return x

# def Compute_residual(A,b,approx_solution):
#     r''' This function takes an input a matrix A, a vector b and an approximate solution to the 
#         system Ax = b and computes the residual :math:`\lVert A^T b - A^T A approx_solution \rVert_2`.

#         Parameters
#         ---------
#         A : ndarray
#             Input matrix of dimension :math:`(m\times n)`
#         b : ndarray
#             Input vector of dimension :math:`(m)`
#         approx_solution : ndarray
#                           Approximate solution to the linear system

        
#         Returns
#         ------
#         r : float
#             Residual
#     ''' 
#     A_transpose = np.transpose(A)
#     C = A_transpose@A
#     d = A_transpose@b

#     residual = d - C @ approx_solution
#     residual_norm_2 = np.linalg.norm(residual, ord=2)
#     return residual, residual_norm_2

# if __name__ == '__main__':
#     pass
