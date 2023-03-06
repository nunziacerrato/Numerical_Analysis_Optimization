import numpy as np
import scipy.linalg
import logging
import qutip
from Project_1 import *


def wilkin(n):
    r''' This function computes the Wilkinson matrix with dimension :math:'(n \times n)'.

        Parameters:
        -----------
        n : int
            Size of the Wilkinson matrix
        
        Returns:
        --------
        W : ndarray
            Wilkinson matrix
    '''
    W = np.tril(-np.ones((n,n)),k=-1) + np.eye(n)
    W[:,n-1] = 1
    return W


def check_when_lufact_W_fails(n_max = 60, treshold = np.finfo(float).eps):
    r''' This function checks the failures of GEPP for a Wilkinson matrix W_n with dimension less or equal than n_max. 
        We define a failure as the case in which the relative error between the solution found by the algorithm and 
        the expected solution is higher than a chosen treshold. The default treshold is set equal to the machine epsilon.
        When an error is found, a warning message is printed. The function returns the list of the dimensions less or 
        equal to n_max for which the GEPP algorithm fails.
        
        Parameters:
        -----------
        n_max : int
                Maximum size of the Wilkinson matrix

        Returns:
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

        # Compute the relative error in 1-norm and print a warning when the erroor exceedes the chosen precision
        error = sum(np.abs(x-vect))/n
        if error <= treshold:
            logging.info(f'n = {n}')
            logging.info(f'||x - e||_1 = {error}')
        elif error > treshold:
            logging.warning(f'n = {n}')
            logging.warning(f'||x - e||_1 = {error}')
            fails.append(n)
            for i in range(n):
                x_i = x[i]
                if abs(x_i-1) > treshold:
                    logging.warning(f'x[{i}] = {x[i]}')
    return fails

def expected_LU_wilkin(n):
    r''' This function generates the exact LU factorization of the Wilkinson matrix with dimension :math:'(n \times n)'.

        Parameters:
        -----------
        n : int
            Size of the Wilkinson matrix

        Returns:
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
    r''' This function compares the exact LU factorization of the Wilkinson matrix with dimension :math:'(n \times n)'
        with the LU factorization computed with the function lufact.

        Parameters:
        -----------
        n : int
            Size of the Wilkinson matrix

        Returns:
        --------
        err_L : float
                Error on L
        err_U : float
                Error on U
    '''
    
    # Generate the Wilkinson matrix with dimension n and compute the LU factorization
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
    r''' This function performs backward and forward substitutions to understand when the GEPP algorithm fails.
         When an error is found, a warning message is printed. The function returns a flag.

        Parameters:
        -----------
        n : int
            Size of the Wilkinson matrix

        Returns:
        --------
        out : int
              Error flag   
              0 : The solution of the system is exact
              1 : Error in the solution of the system
    '''

    out = 0

    # Define the machine error
    epsilon = np.finfo(float).eps

    # Define the Wilginson matrix and perform the LU factorization
    W = wilkin(n)
    L, U, g = lufact(W)        
    vect = np.ones(n)
    b = W @ vect

    # Perform forward substitutions
    y = np.zeros(n)
    print('We are solving Ly=b')
    for i in range(n):
        y[i] = b[i]
        logging.info(f'Compute y[{i}] = b^{i}_{i} = {y[i]}')
        if (y[i] - 2**i) < epsilon and i < n-1:
            logging.warning(f'expected 1: y[{i}] - 2^{i} = {y[i] - 2**i}')
        logging.info(f'Update b: b^{i+1} = b^{i} - y[{i}]*L[:,{i}] = {b} - {y[i]}*{L[:,i]}')
        b = b - y[i]*L[:,i]

    # Perform backward substitutions
    x = np.zeros(n)
    print('We are solving Ux=y')
    for i in range(n):
        x[n-1-i] = y[n-1-i]/U[n-1-i,n-1-i]
        if np.abs(x[n-1-i] - 1) < epsilon:
            logging.info(f'Compute x[{n-1-i}] = y^{i}_{n-1-i}/U[{n-1-i},{n-1-i}] = {y[n-1-i]}/{U[n-1-i,n-1-i]} = {x[n-1-i]}')
        else:
            logging.warning(f'Compute x[{n-1-i}] = y^{i}_{n-1-i}/U[{n-1-i},{n-1-i}] = {y[n-1-i]}/{U[n-1-i,n-1-i]} = {x[n-1-i]}')
            out = 1
        logging.info(f'Update y: y^{i+1} = y^{i} - x[{n-1-i}]*U[:,{n-1-i}] = {y} - {x[n-1-i]}*{U[:,n-1-i]}')
        y = y - x[n-1-i]*U[:,n-1-i]
    
    return out

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







