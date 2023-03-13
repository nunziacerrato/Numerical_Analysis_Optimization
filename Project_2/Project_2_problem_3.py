''' '''

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from Project_2 import *




if __name__ == '__main__':

    A = np.array([[-4.,-2.,-4.,-2.],[2.,-2.,2.,1.],[-800,200,-800,-401]])
    singular_values = np.linalg.svd(A, compute_uv=False)
    pseudoinverse = np.linalg.pinv(A)
    spectral_cond_num = max(singular_values)/min(singular_values)
    print(f'singular values of A = {singular_values}')
    print(f'pseudoinverse of A = {pseudoinverse}')
    print(f'spectral condition number of A = {spectral_cond_num}')