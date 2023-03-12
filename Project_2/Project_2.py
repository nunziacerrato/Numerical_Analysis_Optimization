''' '''

import numpy as np
import scipy.linalg


def compute_A(points,order):
    ''' '''
    m = points.shape[0]
    n = order + 1
    A = np.zeros((m,n))
    for index in range(m):
        A[index,:]=(points[index])**range(n)

    return A


if __name__ == '__main__':

    order = 2
    np.printoptions(precision=15)

    points = np.array([8,10,12,16,20,30,40,60,100])
    b = np.array([0.88,1.22,1.64,2.72,3.96,7.66,11.96,21.56,43.16])
    
    A = compute_A(points, order)
    A_transpose = np.transpose(A)
    C = A_transpose@A
    d = A_transpose@b


    L, low = scipy.linalg.cho_factor(C)
    x = scipy.linalg.cho_solve((L, low),d)
    # x = array(x)
    print(f'{x[0]:.15E}')
    print(format(x[0],'.15f'))
    # print(format(x,'.15f'))
    print(float(x[0]))