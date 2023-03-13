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
    '''
    ''' 
    A_transpose = np.transpose(A)
    C = A_transpose@A
    d = A_transpose@b

    L, low = scipy.linalg.cho_factor(C)
    x = scipy.linalg.cho_solve((L, low),d)
    return x

def QR_fact(A, b):
    '''
    '''
    Q, R = np.linalg.qr(A)
    c = np.transpose(Q) @ b 
    x = scipy.linalg.solve_triangular(R, c, lower = False)
    return x

if __name__ == '__main__':
    pass
