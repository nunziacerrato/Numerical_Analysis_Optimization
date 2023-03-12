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

    np.set_printoptions(precision=15, suppress=True)
    plot = True

    order = 2
    points = np.array([8,10,12,16,20,30,40,60,100])
    b = np.array([0.88,1.22,1.64,2.72,3.96,7.66,11.96,21.56,43.16])

    A = compute_A(points, order)

    x_chol = Cholesky_factorization(A, b)
    x_qr = QR_fact(A, b)
    print(f'Solution with Cholesky: x = {format(x_chol)}')
    print(f'Solution with QR: x = {format(x_qr)}')

    if plot == True:
        x_start = 0.9 * min(points)
        x_stop = 1.1 * max(points)
        x = np.linspace(x_start, x_stop, 1000)

        y = x_qr[0] + x_qr[1] * x + x_qr[2] * x**2

        fig, ax = plt.subplots()
        ax.plot(x, y, label='Best fit')  
        ax.scatter(points, b, marker='o', color='red', label='Data')  

        
        ax.set_title('Best fit and data')
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        plt.show()
