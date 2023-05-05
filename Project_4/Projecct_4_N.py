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
sol_x_a = (2,-1)
sol_f_a = 0

########### b ##############
b = np.array([5.04, -59.4, 146.4, -96.6])
H = np.array([[0.16,-1.2,2.4,-1.4],
              [-1.2,12.0,-27.0,16.8],
              [2.4,-27.0,64.8,-42.0],
              [-1.4,16.8,-42.0,28.0]])
func_b = lambda x: b @ x + 0.5*(x @ H @ x)
grad_b = lambda x: b + H @ x
hess_b = lambda x: H
x0_b_1 = np.array([-1,3,3,0])
sol_x_b = np.array([1,0,-1,2])
sol_f_b = -167.28

########### c ##############
func_c = lambda x: (1.5 - x[0] * (1 - x[1]))**2 + (2.25 - x[0] * (1 - x[1]**2))**2 +\
                    (2.625 - x[0] * (1 - x[1]**3))**2
grad_c = lambda x: np.array([2 * (1.5 - x[0] * (1 - x[1])) * (x[1] - 1) + \
                             2 * (2.25 - x[0] * (1 - x[1]**2)) * (x[1]**2 - 1) + \
                             2 * (2.625 - x[0] * (1 - x[1]**3)) * (x[1]**3 - 1) ,
                             2 * x[0] * (1.5 - x[0] * (1 - x[1])) + \
                             4 * x[0] * x[1] * (2.25 - x[0] * (1 - x[1]**2)) + \
                             6 * x[0] * (x[1]**2) * (2.625 - x[0] * (1 - x[1]**3)) ])
hess_c = lambda x: np.array([[2 * (1 - x[1])**2 + 2 * (1 - x[1]**2)**2 + 2 * (1 - x[1]**3)**2,
                              4 * x[0] * (x[1] - 1) + 8 * x[0] * x[1] * (x[1]**2 - 1) + 12 * x[0] * x[1]**2 * (x[1]**3 - 1) ],
                             [4 * x[0] * (x[1] - 1) + 8 * x[0] * x[1] * (x[1]**2 - 1) + 12 * x[0] * x[1]**2 * (x[1]**3 - 1),
                              2 * x[0]**2 + 12 * x[0]**2 * x[1]**2 + 30 * x[0]**2 * x[1]**4 ]])
x0_c_1 = np.array([5,0.2])
x0_c_2 = np.array([8,0.8])
sol_x_c = np.array([3,0.5])
sol_f_c = 0


########### d ##############
func_d = lambda x: x[0]**4 + x[0] * x[1] + (1 + x[1])**2
grad_d = lambda x: np.array([4 * x[0]**3 + x[1], x[0] + 2 * (1 + x[1])])
hess_d = lambda x: np.array([[ 12 * x[0]**2, 1], [1 , 2]])
x0_d_1 = np.array([0.75,-1.25])
x0_d_2 = np.array([0,0])
sol_x_d = np.array([0.695884386,-1.34794219])
sol_f_d = -0.582445174



def Newton(func, grad, hess, tol, maxit, start_point, sol_x, sol_f, alpha=1, sigma=0.1, rho=0.99, backtracking=False):
    ''' '''
    func_0 = func(start_point)
    old_point = start_point
    alpha_0 = alpha

    for k in range(maxit):
        gradient = grad(old_point)
        hessian = hess(old_point)
        
        try:
            c, low = scipy.linalg.cho_factor(hessian)
            p = scipy.linalg.cho_solve((c, low), -gradient)
        except:
            print('hessian non positive')
            lu, piv = scipy.linalg.lu_factor(hessian)
            p = scipy.linalg.lu_solve((lu, piv), -gradient)
            # alpha=0.1*alpha

        # Backtracking
        if backtracking == True:
            alpha = alpha_0
            iteraz_backtracking = 0 
            while func(old_point + alpha*p) > func(old_point) + (sigma*alpha)*(p @ gradient) and iteraz_backtracking < 100:
                alpha = rho*alpha   
                iteraz_backtracking += 1         

        new_point = old_point + alpha*p

        # Compute norms for the stopping criterion
        diff_x = new_point - old_point
        norm_grad = np.linalg.norm(gradient)
        norm_diff_x = np.linalg.norm(diff_x)

        # Check if the stopping criterion is satisfied
        if norm_grad <= tol and norm_diff_x <=tol*(1 + np.linalg.norm(new_point)):
            break

        old_point = new_point
    
    # Compute the 2norm of the difference between the new point and the exact solution
    diff_sol_x = new_point - sol_x
    norm_diff_sol_x = np.linalg.norm(diff_sol_x)
    # Compute the absolute error of the function evaluated in the new point
    diff_sol_f = func(new_point) - sol_f
    grad_new_point = grad(new_point)
    # Compute the following scalar quantity
    scalar = (-grad_new_point)@(np.linalg.inv(hess(new_point)))@(grad_new_point)
        

    return k, norm_diff_sol_x, diff_sol_f, scalar

k, norm_diff_sol_x, diff_sol_f, scalar = Newton(func_c, grad_c, hess_c, 1e-5, 10000, x0_c_1, sol_x_c, sol_f_c, 1, rho=0.99, backtracking=True)
print(k, norm_diff_sol_x, diff_sol_f, scalar)