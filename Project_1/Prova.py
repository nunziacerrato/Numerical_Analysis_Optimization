import numpy as np
import scipy.linalg
import logging
import tenpy
import qutip

np.random.seed(0)

num_matr = 5
dim_matr = 2

Random = np.zeros((num_matr,dim_matr,dim_matr))
Ginibre = np.zeros((num_matr,dim_matr,dim_matr), dtype=complex)
GOE = np.zeros((num_matr,dim_matr,dim_matr))
GUE = np.zeros((num_matr,dim_matr,dim_matr), dtype=complex)
Wishart = np.zeros((num_matr,dim_matr,dim_matr), dtype=complex)

dataset = {'Random':Random, 'Ginibre':Ginibre, 'GOE':GOE, 'GUE':GUE, 'Wishart':Wishart}
 
for i in range(num_matr):
    dataset['Random'][i,:,:] = np.random.rand(dim_matr,dim_matr)
    dataset['Ginibre'][i,:,:] = np.random.normal(loc=0.0, scale=1.0, size=(dim_matr,dim_matr)) + \
                        1j*np.random.normal(loc=0.0, scale=1.0, size=(dim_matr,dim_matr))
    dataset['GOE'][i,:,:] = tenpy.linalg.random_matrix.GOE((dim_matr,dim_matr))
    dataset['GUE'][i,:,:] = tenpy.linalg.random_matrix.GUE((dim_matr,dim_matr))
    dataset['Wishart'][i,:,:] = np.array(qutip.rand_dm_ginibre((dim_matr), rank=None))

print(dataset)