''' This script implements the unconstrained optimization of a function using the Newton algorithm.
'''
import numpy as np
import numpy.linalg as np_lin
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm

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

hess_c = lambda x: np.array([[2 * (x[1] - 1)**2 + 2 * (x[1]**2 - 1)**2 + 2 * (x[1]**3 - 1)**2,
                              4 * x[0] * (x[1] - 1) + 8 * x[0] * x[1] * (x[1]**2 - 1) + 12 * x[0] * x[1]**2 * (x[1]**3 - 1) ],
                             [4 * x[0] * (x[1] - 1) + 8 * x[0] * x[1] * (x[1]**2 - 1) + 12 * x[0] * x[1]**2 * (x[1]**3 - 1),
                              2 * x[0]**2 + 12 * x[0]**2 * x[1]**2 + 30 * x[0]**2 * x[1]**4 ]])


x0_c_1 = np.array([5,0.2])
x0_c_2 = np.array([8,0.2])
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

    # Create a list to save intermediate points
    interm_points = [old_point]
    # Create a list to save the scalar product between the gradient and the descent direction
    scalar_prod = []

    # Cycle on the number of iterations
    for k in range(maxit):
        gradient = grad(old_point)
        hessian = hess(old_point)
        
        # Compute the descent direction by solving a linear system
        p = np_lin.solve(hessian, -gradient)
        scalar_prod.append(gradient@p)
        #### alpha=0.1*alpha

        # Implement backtracking if backtracking == True is passed to the function
        if backtracking == True:
            alpha = alpha_0
            iteraz_backtracking = 0 
            while func(old_point + alpha*p) > func(old_point) + (sigma*alpha)*(p @ gradient) and iteraz_backtracking < 100:
                alpha = rho*alpha   
                iteraz_backtracking += 1         

        # Compute the new point and add it to the list of intermediate points
        new_point = old_point + alpha*p
        interm_points.append(new_point)

        # Compute norms for the stopping criterion
        norm_grad = np_lin.norm(gradient)
        norm_diff_x = np_lin.norm(new_point - old_point)

        # Check if the stopping criterion is satisfied
        if norm_grad <= tol and norm_diff_x <=tol*(1 + np_lin.norm(new_point)):
            min_value = func(new_point)
            scalar_prod.append(gradient@p)
            print(k)
            break


        old_point = new_point
    
    # Compute the 2norm of the difference between the new point and the exact solution
    error_x = [np_lin.norm(interm_x - sol_x) for interm_x in interm_points]
    
    # Compute the absolute error of the function evaluated in the new point with respect
    # to its value in the exact minimum point
    error_f = [func(interm_x) - sol_f for interm_x in interm_points]
    
    # grad_new_point = grad(new_point)
    # # Compute the scalar product between the gradient and the descend direction
    # scalar_prod = (-grad_new_point)@(np_lin.inv(hess(new_point)))@(grad_new_point)

    results = {'final_iter' : k+1, 'min_point' : new_point, 'min_value' : min_value ,
               'interm_point' : interm_points, 'error_x' : error_x, 'error_f' : error_f,
               'scalar_product' : scalar_prod}

    return results


if __name__ == '__main__':

    compute = 'a'

    # (a)
    if compute == 'a':
        results = Newton(func_a, grad_a, hess_a, 1e-5, 100, x0_a_1, sol_x_a, sol_f_a, 1, backtracking=False)
        print(results['final_iter'])
        print(results['error_x'])
        print(results['error_f'])
        print(results['scalar_product'])

        delta = 0.025
        X = np.arange(0., 3.0, delta)
        Y = np.arange(-2.0, 2.0, delta)
        X, Y = np.meshgrid(X,Y)

        Z = (X-2)**4 + ((X-2)**2)*(Y**2) + (Y+1)**2

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True, alpha=0.7)
        
        # Set colors and format
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        interm_points = results['interm_point']
        print(interm_points)
        zz = func_a(interm_points[-1])
        print(zz)
        interm_x = np.array([interm_points[i][0] for i in range(0,len(interm_points))])
        interm_y = np.array([interm_points[i][1] for i in range(0,len(interm_points))])
        zz = func_a([interm_x, interm_y])
        scatter = ax.plot(interm_x, interm_y, zz, '-o', markersize=4, color='black')
        plt.show()

    if compute == 'b':
        results = Newton(func_b, grad_b, hess_b, 1e-5, 100, x0_b_1, sol_x_b, sol_f_b, 1, backtracking=False)
        print(results['final_iter'])

    if compute == 'c':
        results = Newton(func_c, grad_c, hess_c, 1e-5, 100, x0_c_2, sol_x_c, sol_f_c, 1, backtracking=False)
        print(results['final_iter'])

        delta = 0.025
        X = np.arange(0., 3.0, delta)
        Y = np.arange(-2.0, 2.0, delta)
        X, Y = np.meshgrid(X,Y)
        
        Z = (1.5 - X * (1 - Y))**2 + (2.25 - X * (1 - Y**2))**2 +\
                    (2.625 - X * (1 - Y**3))**2

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True, alpha=0.7)
        
        # Set colors and format
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        interm_points = results['interm_point']
        print(interm_points)
        print(interm_points[0][0])
        zz = func_a(interm_points)
        print(zz)
        interm_x = np.array([interm_points[i][0] for i in range(0,len(interm_points))])
        interm_y = np.array([interm_points[i][1] for i in range(0,len(interm_points))])
        zz = func_a([interm_x, interm_y])
        scatter = ax.plot(interm_x, interm_y, zz, '-o', markersize=4, color='black')
        plt.show()

