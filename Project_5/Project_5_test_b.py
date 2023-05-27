import numpy as np
from Project_5 import *

func_b = lambda x : 2*x[0] - x[1]**2
grad_b = lambda x : np.array([2, -2*x[1]])
hess_b = lambda x : np.array([[0, 0],[0, -2]])

c_b = lambda x : np.array([1 - x[0]**2 - x[1]**2 , x[0], x[1]])
grad_c_b = lambda x : np.array([[-2*x[0],-2*x[1]],[1,0],[0,1]])

x0 = np.array([0.,0.99])

method = 'basic'
mu = 0
tol = 1e-12
seed = 1

results = int_point(func_b, grad_b, hess_b, c_b, grad_c_b, x0, method=method, alpha=1., beta=1.,
                   gamma=1., mu=mu, tol=tol, maxit=100, seed=seed)
k = results['n_iter']
conv = results['convergence']
# x0 = results['x0']
min_point = results['x_min']
min_value = results['f_min']
mu = results['mu']
print(f'convergence = {conv}, with {k} steps')
print(f'starting point = {x0}, mu = {mu}')
print(f'min point = {min_point}')
print(f'min value = {min_value}')