''' This script implements the unconstrained optimization of a function using the Newton algorithm.
'''
import numpy as np
import scipy.linalg

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
sol_x_b = np.array([1,0,-1,2])
sol_f_b = -167.28


def Newton(func, grad, hess, tol, maxit, start_point, sol_x, sol_f):
    ''' '''
    func_0 = func(start_point)
    old_point = start_point

    for k in range(maxit):
        gradient = grad(old_point)
        hessian = hess(old_point)
        
        c, low = scipy.linalg.cho_factor(hessian)
        p = scipy.linalg.cho_solve((c, low), -gradient)

        new_point = old_point + p

        # Compute norms for the stopping criterion
        diff_x = new_point - old_point
        norm_grad = np.linalg.norm(gradient)
        norm_diff_x = np.linalg.norm(diff_x)

        # Check if the stopping criterion is satisfied
        if norm_grad <= tol and norm_diff_x <=tol*(1 + np.linalg.norm(new_point)):
            # Compute the 2norm of the difference between the new point and the exact solution
            diff_sol_x = new_point - sol_x
            norm_diff_sol_x = np.linalg.norm(diff_sol_x)
            # Compute the absolute error of the function evaluated in the new point
            diff_sol_f = func(new_point) - sol_f
            grad_new_point = grad(new_point)
            # Compute the following scalar quantity
            scalar = (-grad_new_point)@(np.linalg.inv(hess(new_point)))@(grad_new_point)
            break
        # Compute the 2norm of the difference between the new point and the exact solution
        diff_sol_x = new_point - sol_x
        norm_diff_sol_x = np.linalg.norm(diff_sol_x)
        # Compute the absolute error of the function evaluated in the new point
        diff_sol_f = func(new_point) - sol_f
        grad_new_point = grad(new_point)
        # Compute the following scalar quantity
        scalar = (-grad_new_point)@(np.linalg.inv(hess(new_point)))@(grad_new_point)

    return k, norm_diff_sol_x, diff_sol_f, scalar

k, norm_diff_sol_x, diff_sol_f, scalar = Newton(func_b, grad_b, hess_b, 1e-4, 10000, x_0_b, sol_x_b, sol_f_b)
print(k, norm_diff_sol_x, diff_sol_f, scalar)