import numpy as np
import itertools
import matplotlib.pyplot as plt
from Project_5 import *

func_b = lambda x : 2*x[0] - x[1]**2
grad_b = lambda x : np.array([2, -2*x[1]])
hess_b = lambda x : np.array([[0, 0],[0, -2]])
# func_b = lambda x : 2*x[1] - x[0]**2
# grad_b = lambda x : np.array([-2*x[0], 2])
# hess_b = lambda x : np.array([[-2, 0],[0, 0]])

c_b = lambda x : np.array([1 - x[0]**2 - x[1]**2 , x[0], x[1]])
grad_c_b = lambda x : np.array([[-2*x[0],-2*x[1]],[1,0],[0,1]])
def hess_c_b(x):
    hess_c = np.zeros((2,3,2))
    hess_c[:,0,:] = np.array([[-2, 0],[0, -2]])
    return hess_c

x0 = np.array([0.3,0.7])

method = 'basic'
mu = 1e-5
tol = 1e-12
seed = 1
common_path = "Project_5"

max_val = 10
start = 1e-12
step = 0.1
l_array = list(np.arange(start,max_val+1e-10,step))
z_array = list(np.arange(start,max_val+1e-10,step))
dim_l = len(l_array)
dim_z = len(z_array)
M = np.zeros([dim_l,dim_z])
tol_x = 1e-1
x0_list = [np.array([0.1,0.1]),np.array([0.1,0.9]),np.array([0.9,0.1]),np.array([0.3,0.7]),
           np.array([0.7,0.3]),np.array([0.5,0.5]),np.array([0,0]),np.array([1,0]),np.array([0,1])]

for x0 in x0_list:
    print(f'x0 = {x0}')
    for l_int in l_array:
        for z_int in z_array:
            l0 = l_int*np.ones(3)
            z0 = z_int*np.ones(3)
            # print(f'iter = {l_int,z_int}')
            results = int_point(func_b, grad_b, hess_b, c_b, grad_c_b, hess_c_b, x0, method=method, alpha=1., beta=1.,
                        gamma=1., mu=mu, tol=tol, maxit=100, l0=l0, z0=z0, seed=seed)
            k = results['n_iter']
            min_val = results['x_min']
            min_exact = np.array([0.,1.])
            min_err = np.array([0.,0.])
            if np.linalg.norm(min_val-min_exact) > tol_x and np.linalg.norm(min_val-min_err) < tol_x:
                k = 150
            l_index = l_array.index(l_int)
            z_index = z_array.index(z_int)
            M[l_index,z_index] = k

    fig = plt.figure()
    # plt.imshow(M,cmap="inferno")
    X, Y = np.meshgrid(l_array, z_array)
    plt.pcolormesh(X, Y, M, cmap='jet')
    plt.title(fr'Convergence iteration - $\mu$ = {mu}, $x_{{0}}$={x0}')
    plt.xlabel(r'$\lambda$')
    plt.ylabel("z")
    # plt.xticks(np.arange(start,max_val+1))
    # plt.yticks(np.arange(start,max_val+1))
    plt.colorbar()
    fig.savefig(f'{common_path}_latex\\Plot\\func_a_method={method}_x0={x0}_mu={mu}.png', bbox_inches='tight', dpi = 500)
    # plt.show()
# conv = results['convergence']
# # x0 = results['x0']
# min_point = results['x_min']
# min_value = results['f_min']
# mu = results['mu']
# x_interm = results['x_interm']
# lambda_interm = results['lambda_interm']
# z_interm = results['z_interm']
# print(f'convergence = {conv}, with {k} steps')
# print(f'starting point = {x0}, mu = {mu}')
# print(f'min point = {min_point}')
# print(f'min value = {min_value}')
# print(f'x_interm = {x_interm}')
# print(f'lambda_interm = {lambda_interm}')
# print(f'z_interm = {z_interm}')