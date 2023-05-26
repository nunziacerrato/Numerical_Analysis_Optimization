from Project_5 import *

func_a = lambda x : (x[0] - 4)**2 + x[1]**2
grad_a = lambda x : np.array([2*(x[0]-4), 2*x[1]])
hess_a = lambda x : np.array([[2, 0],[0, 2]])

c_a = lambda x : np.array([x[0] + x[1] -2, -x[0], -x[1]])
grad_c_a = lambda x : np.array([[1,1],[-1,0],[0,-1]])

x0 = np.array([0.1,0.1])

result = int_point(func_a, grad_a, hess_a, c_a, grad_c_a, x0, method='full', alpha=1., beta=1., gamma=1., tol=1e-12, maxit=100, seed=1)
print(f'result={result}')