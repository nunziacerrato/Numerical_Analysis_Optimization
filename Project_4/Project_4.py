''' This script implements the unconstrained optimization of a function using the Newton algorithm.
'''
import numpy as np
import numpy.linalg as np_lin

########### a ##############
func_a = lambda x: (x[0]-2)**4 + ((x[0]-2)**2)*x[1]**2 + (x[1]+1)**2
grad_a = lambda x: np.array([4*(x[0]-2)**3 + 2*(x[0]-2)*(x[1]**2), 2*((x[0]-2)**2)*x[1] + 2*(x[1]+1)])
hess_a = lambda x: np.array([[12*(x[0]-2)**2 + 2*x[1]**2, 4*(x[0]-2)*x[1]],
                                  [4*(x[0]-2)*x[1], 2*(x[0]-2)**2 + 2]])
x0_a_1 = np.array([1,1])
x0_a_2 = np.array([2,-1])

########### b ##############
b = np.array([5.04, -59.4, 146.4, -96.6])
H = np.array([[0.16,-1.2,2.4,-1.4],
              [-1.2,12.0,-27.0,16.8],
              [2.4,-27.0,64.8,-42.0],
              [-1.4,16.8,-42.0,28.0]])
func_b = lambda x: b @ x + 0.5*(x @ H @ x)
grad_b = lambda x: b + H @ x
hess_b = lambda x: H
x0_b = np.array([-1,3,3,0])


old_point = x0_b
gradient = grad_b(old_point)
hessian = hess_b(old_point)
eigval, eigvect = np_lin.eigh(hess_b(old_point))

mu = abs(min(eigvect.any(), 0)) + 1e-15
print(mu)
print(eigvect)
print(eigvect.T)
coeff_vect_1 = eigvect@grad_b(old_point)/(eigval + mu)
coeff_vect_2 = np.transpose(eigvect)@grad_b(old_point)/(eigval + mu)

print(coeff_vect_1)
print(coeff_vect_2)
print('-------------------')
p = - (eigvect.T @ grad_b(old_point))/(eigval+mu) @ eigvect
p2 = - eigvect /(eigval+mu) @ eigvect.T @ grad_b(old_point)


coeff = eigvect.T @ grad_b(old_point)
print(coeff)
print('--------------')
print(f'eigvect = {eigvect}')
shift_eigval = eigval + mu
print(f'eigval + mu = {shift_eigval}')
num = eigvect/(eigval + mu)
print(f'num={num}')
print('--------------')
print(p)
print(p2)
print('--------------')
vect = np.zeros(len(eigval))
print(vect)
for i in range(len(eigval)):
    c = eigvect[:,i]@grad_b(old_point)
    denom = eigval[i] + mu 
    frac = -(c/denom)*eigvect[:,i]
    vect = vect + frac
print(vect)

coeff_vect = eigvect.T @ grad_b(old_point)/(eigval + mu)
ccc = eigvect@coeff_vect
print(ccc)