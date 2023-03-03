import numpy as np
import scipy.linalg
import logging
import qutip
from Project_1 import *


def wilkin(n):
    W = np.tril(-np.ones((n,n)),k=-1) + np.eye(n)
    W[:,n-1] = 1
    return W

def expected_LU_wilkin(n):
    L = np.tril(-np.ones((n,n)),k=-1) + np.eye(n)
    U = np.eye(n) 
    U[:,n-1] = 2**np.array(range(n))
    return L, U

def compute_error_lufact_W(n):
    W = wilkin(n)
    L, U, g = lufact(W)
    expected_L, expected_U = expected_LU_wilkin(n)
    err_L = np.amax(np.abs(L-expected_L))
    err_U = np.amax(np.abs(U-expected_U))
    print(f'error on L = {err_L}')
    print(f'error on U = {err_U}')
    return err_L, err_U

def check_when_lufact_W_fails(n_max = 60, treshold = np.finfo(float).eps):
    
    for n in range(2,n_max+1):
        W = wilkin(n)
        logging.debug(f'W = {W}')

        vect = np.ones(n)
        b = W @ vect
        x = np.linalg.solve(W,b)

        error = sum(np.abs(x-vect))
        if error <= treshold:
            logging.info(f'n = {n}')
            logging.info(f'||x - e||_1 = {error}')
        elif error > treshold:
            logging.warning(f'n = {n}')
            logging.warning(f'||x - e||_1 = {error}')
            for i in range(n):
                x_i = x[i]
                if abs(x_i-1) > treshold:
                    logging.warning(f'x[{i}] = {x[i]}')

def step_by_step_GEPP_W(n):
    W = wilkin(n)
    L, U, g = lufact(W)        
    vect = np.ones(n)
    b = W @ vect

    y = np.zeros(n)
    print('We want to solve Ly=b')
    for i in range(n):
        y[i] = b[i]
        print(f'Compute y[{i}] = b^{i}_{i} = {y[i]}')
        # print(f'expected 1: {y[i] - 2**i}')
        print(f'Update b: b^{i+1} = b^{i} - y[{i}]*L[:,{i}] = {b} - {y[i]}*{L[:,i]}')
        b = b - y[i]*L[:,i]

    x = np.zeros(n)
    print('We want to solve Ux=y')
    for i in range(n):
        x[n-1-i] = y[n-1-i]/U[n-1-i,n-1-i]
        print(f'Compute x[{n-1-i}] = y^{i}_{n-1-i}/U[{n-1-i},{n-1-i}] = {y[n-1-i]}/{U[n-1-i,n-1-i]} = {x[n-1-i]}')
        print(f'Update y: y^{i+1} = y^{i} - x[{n-1-i}]*U[:,{n-1-i}] = {y} - {x[n-1-i]}*{U[:,n-1-i]}')
        y = y - x[n-1-i]*U[:,n-1-i]

if __name__=='__main__':

    logging.basicConfig(level=logging.WARNING)
    
    step_by_step_GEPP_W(55)

    a = 2.**53
    print(a)
    b = a + 1
    print(b)
    c = b - a
    print(c)
    print(f'epsilon = {np.finfo(float).eps}')
    print(2**53 - np.finfo(float).eps**-1)
    print(f'2^53 = {2**53}')
    # print(54*np.log10(2))
    # print(10**(54*np.log10(2)-int(54*np.log10(2))))







