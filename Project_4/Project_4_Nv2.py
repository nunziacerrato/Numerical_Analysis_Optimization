''' This script implements the unconstrained optimization of a function using the Newton algorithm.
'''
import numpy as np
import numpy.linalg as np_lin
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import seaborn as sns

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
                              2*(1.5) + 4 * x[0] * (x[1] - 1) + 4*(2.25)*x[1] + 8 * x[0] * x[1] * (x[1]**2 - 1) + 6*(2.625)*(x[1]**2) + 12 * x[0] * x[1]**2 * (x[1]**3 - 1) ],
                             [2*(1.5) +  4 * x[0] * (x[1] - 1) + 4*(2.25)*x[1] + 8 * x[0] * x[1] * (x[1]**2 - 1) + 6*(2.625)*(x[1]**2) + 12 * x[0] * x[1]**2 * (x[1]**3 - 1),
                              2*(x[0])**2 + 2*(2*x[0]*x[1])**2 + 2*(3*x[0]*(x[1]**2))**2 + 
                              4*x[0]*(2.25 - x[0]*(1-x[1]**2)) + 12*x[0]*x[1]*(2.625 -x[0]*(1-x[1]**3))]])
                              
                              
                            #   + 2 * x[0]**2 + 12 * x[0]**2 * x[1]**2 + 30 * x[0]**2 * x[1]**4 ]])


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



def Newton(func, grad, hess, tol, maxit, x_0, sol_x, sol_f, alpha=1, sigma=0.0001, rho=0.5, backtracking=False):
    ''' '''
    old_point = x_0
    # func_0 = func(start_point)
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
        # alpha=0.1*alpha

        # # Implement backtracking if backtracking == True is passed to the function
        if backtracking == True:
            alpha = alpha_0
            iteraz_backtracking = 0 
            while func(old_point + alpha*p) > func(old_point) + (sigma*alpha)*(p @ gradient) and iteraz_backtracking < 100:
                # print('a')
                alpha = rho*alpha
                # new_point = old_point + alpha*p
                # old_point = new_point
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
            print(k)
            break

        old_point = new_point
    
    gradient = grad(new_point)
    hessian = hess(new_point)
    p = np_lin.solve(hessian, -gradient)
    scalar_prod.append(gradient@p)

    # Compute the 2norm of the difference between the new point and the exact solution
    error_x = [np_lin.norm(interm_x - sol_x) for interm_x in interm_points]
    
    # Compute the absolute error of the function evaluated in the new point with respect
    # to its value in the exact minimum point
    error_f = [func(interm_x) - sol_f for interm_x in interm_points]
    
    # grad_new_point = grad(new_point)
    # # Compute the scalar product between the gradient and the descend direction
    # scalar_prod = (-grad_new_point)@(np_lin.inv(hess(new_point)))@(grad_new_point)

    results = {'k' : k, 'final_iter' : k+2, 'min_point' : new_point, 'min_value' : min_value ,
               'interm_point' : interm_points, 'error_x' : error_x, 'error_f' : error_f,
               'scalar_product' : scalar_prod}

    return results



def Newton_Backtracking(func, grad, hess, tol, maxit, x_0, sol_x, sol_f, alpha=1, sigma=0.1, rho=0.99):
    ''' '''
    old_point = x_0
    # func_0 = func(start_point)
    # alpha_0 = alpha

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

        # # Implement backtracking if backtracking == True is passed to the function
        # if backtracking == True:
        #     # alpha = alpha_0
        #     iteraz_backtracking = 0 
        #     while func(old_point + alpha*p) > func(old_point) + (sigma*alpha)*(p @ gradient) and iteraz_backtracking < 100:
        #         # print('a')
        #         alpha = rho*alpha
        #         new_point = old_point + alpha*p
        #         old_point = new_point
        #         iteraz_backtracking += 1         

        # Compute the new point and add it to the list of intermediate points
        new_point = old_point + alpha*p
        interm_points.append(new_point)

        # Compute norms for the stopping criterion
        norm_grad = np_lin.norm(gradient)
        norm_diff_x = np_lin.norm(new_point - old_point)

        # Check if the stopping criterion is satisfied
        if norm_grad <= tol and norm_diff_x <=tol*(1 + np_lin.norm(new_point)):
            min_value = func(new_point)
            print(k)
            break

        old_point = new_point
    
    gradient = grad(new_point)
    hessian = hess(new_point)
    p = np_lin.solve(hessian, -gradient)
    scalar_prod.append(gradient@p)

    # Compute the 2norm of the difference between the new point and the exact solution
    error_x = [np_lin.norm(interm_x - sol_x) for interm_x in interm_points]
    
    # Compute the absolute error of the function evaluated in the new point with respect
    # to its value in the exact minimum point
    error_f = [func(interm_x) - sol_f for interm_x in interm_points]
    
    # grad_new_point = grad(new_point)
    # # Compute the scalar product between the gradient and the descend direction
    # scalar_prod = (-grad_new_point)@(np_lin.inv(hess(new_point)))@(grad_new_point)

    results = {'k' : k, 'final_iter' : k+2, 'min_point' : new_point, 'min_value' : min_value ,
               'interm_point' : interm_points, 'error_x' : error_x, 'error_f' : error_f,
               'scalar_product' : scalar_prod}

    return results


def Trust_Region(func, grad, hess, tol, maxit, x_0, sol_x, sol_f, alpha=1, eta=0.01):
    ''' '''

    # Evaluate the gradient and the hessian in the starting point
    gradient = grad(x_0)
    hessian = hess(x_0)
    
    # Compute the fist descent direction and the first delta value
    p = np_lin.solve(hessian, -gradient)
    delta = np_lin.norm(p)
    interm_radius = [delta]

    interm_points = [x_0]
    scalar_prod = []
    old_point = x_0
    # Cycle on the number of iterations
    for k in range(maxit):
        
        # Evaluate the gradient and the hessian at the current point
        gradient = grad(old_point)
        hessian = hess(old_point)

        # Diagonalize the hessian computed at the current point
        eigval, eigvect = np_lin.eigh(hessian)
        # Choose an initial mu value (to be adjusted)
        mu = abs(min(min(eigval), 0)) + 1e-12

        coeff_vect = eigvect.T @ gradient/(eigval + mu)

        # Choose the optimal mu value which respects the condition on the 2-norm of p
        while sum([coeff**2 for coeff in coeff_vect]) > delta**2:
            mu = mu*2
            coeff_vect = eigvect.T @ gradient/(eigval + mu)
        # print(mu)
        # Compute the descent direction
        # p = - (eigvect.T @ grad(old_point))/(eigval+mu) @ eigvect
        p = - eigvect /(eigval+mu) @ eigvect.T @ gradient
        # p = - eigvect @ coeff_vect
        scalar_prod.append(- gradient @ eigvect @ np.diag(1/eigval) @ eigvect.T @ gradient)

        new_point = old_point + alpha*p

        # Choose the value of delta for the successive iteration
        rho = (func(new_point)-func(old_point))/(p @ gradient + 0.5*p @ hessian @ p)
        print(f'k={k}')
        print(f'rho={rho}')

        if 0 < rho < 0.25:
            delta = delta/4
            interm_radius.append(delta)
        elif rho > 0.75: # and np_lin.norm(p)==delta:
            print(f'k={k}')
            print(f'rho={rho}')
            delta = 2*delta
            interm_radius.append(delta)
        elif 0 < rho < eta:
            new_point = old_point
            interm_radius.append(delta)

        interm_points.append(new_point)

        # Compute norms for the stopping criterion
        norm_grad = np_lin.norm(gradient)
        norm_diff_x = np_lin.norm(new_point - old_point)

        # Check if the stopping criterion is satisfied
        if norm_grad <= tol and norm_diff_x <=tol*(1 + np_lin.norm(new_point)):
            min_value = func(new_point)
            print(k)
            break

        old_point = new_point
    
    gradient = grad(new_point)
    hessian = hess(new_point)
    p = np_lin.solve(hessian, -gradient)
    scalar_prod.append(- gradient @ eigvect @ np.diag(1/eigval) @ eigvect.T @ gradient)

    # Compute the 2norm of the difference between the new point and the exact solution
    error_x = [np_lin.norm(interm_x - sol_x) for interm_x in interm_points]
    
    # Compute the absolute error of the function evaluated in the new point with respect
    # to its value in the exact minimum point
    error_f = [func(interm_x) - sol_f for interm_x in interm_points]
    
    # grad_new_point = grad(new_point)
    # # Compute the scalar product between the gradient and the descend direction
    # scalar_prod = (-grad_new_point)@(np_lin.inv(hess(new_point)))@(grad_new_point)

    results = {'k' : k, 'final_iter' : k+2, 'min_point' : new_point, 'min_value' : min_value,
               'interm_point' : interm_points, 'interm_radius' : interm_radius, 'error_x' : error_x, 'error_f' : error_f,
               'scalar_product' : scalar_prod}

    return results



if __name__ == '__main__':

    compute = 'd'

    # Set size parameters for the plots
    tickparams_size = 16
    xylabel_size = 12
    suptitle_size = 35
    title_size = 22
    legend_size = 19

    # (a)
    if compute == 'a':
        results = Newton(func_a, grad_a, hess_a, 1e-5, 100, x0_a_1, sol_x_a, sol_f_a, 1, backtracking=False)
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
        ax.zaxis.set_major_formatter('{x:.0f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect= 10)

        interm_points = results['interm_point']
        print(interm_points)
        zz = func_a(interm_points[-1])
        print(zz)
        interm_x = np.array([interm_points[i][0] for i in range(0,len(interm_points))])
        interm_y = np.array([interm_points[i][1] for i in range(0,len(interm_points))])
        zz = func_a([interm_x, interm_y])
        scatter = ax.plot(interm_x, interm_y, zz, '-o', markersize=4, color='black')
        scatter = ax.plot(interm_x[-1], interm_y[-1], zz[-1], '-o', markersize=4, color='red')
        ax.set_xlabel(r'$x_{1}$', fontsize=xylabel_size)
        ax.set_ylabel(r'$x_{2}$', fontsize=xylabel_size)
        plt.show()

    if compute == 'b':
        results = Newton(func_b, grad_b, hess_b, 1e-5, 100, x0_b_1, sol_x_b, sol_f_b, 1, backtracking=False)
        print(results['final_iter'])
        interm_points = results['interm_point']
        print(interm_points)
        zz = func_b(interm_points[-1])
        print(zz)

    if compute == 'c':
        results = Newton(func_c, grad_c, hess_c, 1e-5, 100, x0_c_2, sol_x_c, sol_f_c, 1, backtracking=True)
        print(results['final_iter'])

        delta = 0.025
        X = np.arange(0., 9.0, delta) # x in [0,9] per backtracking
        Y = np.arange(0., 1., delta) # y in [0,1] per backtracking
        X, Y = np.meshgrid(X,Y)
        
        
        Z = (1.5 - X * (1 - Y))**2 + (2.25 - X * (1 - Y**2))**2 +\
                    (2.625 - X * (1 - Y**3))**2

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True, alpha=0.7)
        
        # Set colors and format
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.0f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=10)

        interm_points = results['interm_point']
        print(interm_points)
        # print(interm_points[0][0])
        zz = func_c(interm_points[-1])
        print(zz)
        interm_x = np.array([interm_points[i][0] for i in range(0,len(interm_points))])
        interm_y = np.array([interm_points[i][1] for i in range(0,len(interm_points))])
        zz = func_c([interm_x, interm_y])
        scatter = ax.plot(interm_x, interm_y, zz, '-o', markersize=4, color='black')
        scatter = ax.plot(interm_x[-1], interm_y[-1], zz[-1], '-o', markersize=4, color='red')
        ax.set_xlabel(r'$x_{1}$', fontsize=xylabel_size)
        ax.set_ylabel(r'$x_{2}$', fontsize=xylabel_size)
        plt.show()

    if compute == 'd':
        # results = Newton(func_d, grad_d, hess_d, 1e-5, 100, x0_d_2, sol_x_d, sol_f_d, 0.9, backtracking=False)
        results = Trust_Region(func_d, grad_d, hess_d, 1e-5, 100, x0_d_2, sol_x_d, sol_f_d)
        print(results['interm_radius'])

        delta = 0.0025
        X = np.arange(-0.25, 1.6, delta) # x in [0,9] per backtracking
        Y = np.arange(-1.6, 0.25, delta) # y in [0,1] per backtracking
        X, Y = np.meshgrid(X,Y)
        
        Z = X**4 + X * Y + (1 + Y)**2

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig_contour, ax_contour = plt.subplots()
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True, alpha=0.7)
        
        delta = 0.0025
        Xc = np.arange(-0.5, 2, delta) # x in [0,9] per backtracking
        Yc = np.arange(-2, 0.5, delta) # y in [0,1] per backtracking
        Xc, Yc = np.meshgrid(Xc,Yc)      
        Zc = Xc**4 + Xc * Yc + (1 + Yc)**2

        
        cp = ax_contour.contour(Xc, Yc, Zc, levels = 80)
        fig_contour.colorbar(cp, shrink=0.5, aspect=10)

        # Set colors and format
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=10)

        interm_points = results['interm_point']
        print(interm_points)
        # print(interm_points[0][0])
        zz = func_d(interm_points[-1])
        print(zz)
        interm_x = np.array([interm_points[i][0] for i in range(0,len(interm_points))])
        interm_y = np.array([interm_points[i][1] for i in range(0,len(interm_points))])
        zz = func_d([interm_x, interm_y])
        scatter = ax.plot(interm_x, interm_y, zz, '-o', markersize=4, color='black')
        scatter = ax.plot(interm_x[-1], interm_y[-1], zz[-1], '-o', markersize=4, color='red')
        ax.set_xlabel(r'$x_{1}$', fontsize=xylabel_size)
        ax.set_ylabel(r'$x_{2}$', fontsize=xylabel_size)
        
        scatter_c = ax_contour.plot(interm_x, interm_y, '-o', markersize=4, color='black')
        scatter_c = ax_contour.plot(interm_x[-1], interm_y[-1], '-o', markersize=4, color='red')
        # interm_rad = results['interm_radius']
        # for i in range(1):
        #     circle = plt.Circle([interm_x[i],interm_y[i]], interm_rad[i])
        #     ax_contour.add_patch(circle)
        
        
        
        
        plt.show()

