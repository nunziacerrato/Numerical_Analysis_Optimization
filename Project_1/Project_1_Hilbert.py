''' In this program we study the growth factor and the relative backward error related to
    the LU factorization algorithm performed on Hilbert matrices. The aim of this code is to
    investigate what is the critical dimension of the input matrix which cause the algorthm to
    break, having fixed a certain precision.
'''

import scipy.linalg
import matplotlib.pyplot as plt
from Project_1 import *

# Set size parameters for the plots
tickparams_size = 15
xylabel_size = 16
title_size = 18
legend_size = 15

logging.basicConfig(level=logging.ERROR)

# Choose a maximum dimension for the input matrix
n_max = 30

# Obtain a list of Hilbert matrices, where the index of the list represent the increasing
# dimension of the matrix.
Hilbert_matr_vect = [scipy.linalg.hilbert(n) for n in range(1,n_max+1)]
g_list = []
rel_back_err_list = []
for i in range(len(Hilbert_matr_vect)):
    try:
        L, U, g = lufact(Hilbert_matr_vect[i])
        g_list.append(g)
        rel_back_err_list.append(relative_backward_error(Hilbert_matr_vect[i], L, U))
    except ValueError:
        print(f'LU factorization breaks for N={i+1}')

# Obtain the plot of the growth factor and the relative backward error as a function of the
# dimension of the input matrix.
plt.scatter(range(1, len(g_list)+1), g_list, label = r'$\gamma$' )
plt.scatter(range(1, len(g_list)+1), rel_back_err_list, label = r'$\delta$')
plt.title('Scatterplot of the growth factor and relative backward error for the Hilbert matrices',
           fontsize = title_size)
plt.xlabel('N', fontsize = xylabel_size)
plt.tick_params(labelsize = tickparams_size)
plt.legend(fontsize = legend_size)
plt.show()