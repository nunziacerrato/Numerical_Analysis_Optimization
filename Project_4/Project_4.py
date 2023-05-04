''' This script implements the unconstrained optimization of a function using the Newton algorithm.
'''
import numpy as np

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
x_0_b = np.array([-1,3,3,0])


def Netwon(func, grad, hess, tol, maxit, start_point):
    ''' '''
    func_0 = func(start_point)
    old_point = start_point

    for k in range(maxit):
        gradient = grad(old_point)
        hessian = hess(old_point)
        
        c, low = cho_factor(hessian)
        p = cho_solve((c, low), -gradient)

        new_point = old_point + p

        diff_x = new_point - old_point
        norm_grad = np.linalg.norm(gradient, ord='fro')
        norm_diff_x = np.linalg.norm(diff_x, ord='fro')
        if norm_grad <= tol and norm_diff_x <=tol*(1 + np.linalg.norm(new_point, ord='fro')):
            
            break

    return k, 