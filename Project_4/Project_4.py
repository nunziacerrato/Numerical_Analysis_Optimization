''' This file serves as a library. It containes two functions which implement, respectively, the
    standard Newton method, with the possibility of using the backtracking approach by passing 
    the variable backtracking=True to this function, and the Newton method with the trust region
    approach. These functions are used to perform the unconstrained optimization of a function.
'''
import numpy as np
import numpy.linalg as np_lin


def Newton(func, grad, hess, tol, maxit, x_0, sol_x, sol_f, alpha=1, sigma=0.0001, rho=0.5, backtracking=False):
    ''' This function implements the standard Newton method for unconstrained optimization.
    
    Parameters
    ----------
    func : func
        Function to be minimized. It must be :math:`f : \mathbb{R}^{n} \rightarrow \mathbb{R}`.
    grad : func
        Gradient of the function. It returns a 1d-array (vector)
    hess : func
        Hessian of the function. It returns a 2d-array (matrix)
    tol : float
        Tolerance parameter for the stopping criterion
    maxit : int
        Maximum number of iterations
    x_0 : ndarray
        Starting point
    sol_x : ndarray
        Exact solution to the minimization problem
    sol_f : float
        Value of the function in correspondence with the exact solution
    alpha : float
        Step lenght
    sigma : float
        Constant parameter in :math:`(0,1)`. Used if backtracking = True. Default value sigma=0.0001
    rho : float
        Reduction parameter in :math:`(0,1)`. Used if backtracking = True. Default value rho=0.5

    Results
    -------
    results : dict
        Dictonary of the results given by the function. It contains the following items:
        - 'k': (int) final iteration at which convergence is reached
        - 'min_point' : (ndaray) computed point at which minimum is reached
        - 'min_value' : (float) computed minimum value of the function
        - 'interm_point' : (list) list of the intermediate points
        - 'error_x' : (float) 2-norm of the difference between 'min_point' and the exact solution
        - 'error_f' : (float) absolute error of the function evaluated at 'min_point' with respect
                      to its value in the exact minimum point
        - 'scalar_product' : (list) scalar product between the discent direction and the gradient

    '''
    old_point = x_0
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

        # Implement backtracking if backtracking == True is passed to the function
        if backtracking == True:
            alpha = alpha_0
            iteraz_backtracking = 0 
            while func(old_point + alpha*p) > func(old_point) + (sigma*alpha)*(p @ gradient) \
                  and iteraz_backtracking < 100:
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
    
    results = {'k' : k, 'min_point' : new_point, 'min_value' : min_value ,
               'interm_point' : interm_points, 'error_x' : error_x, 'error_f' : error_f,
               'scalar_product' : scalar_prod}

    return results



def Newton_trust_region(func, grad, hess, tol, maxit, x_0, sol_x, sol_f, alpha=1, eta=0.01):
    ''' This function implements the Newton method with the trust region approach for unconstrained 
        optimization.
    
    Parameters
    ----------
    func : func
        Function to be minimized. It must be :math:`f : \mathbb{R}^{n} \rightarrow \mathbb{R}`.
    grad : func
        Gradient of the function. It returns a 1d-array (vector)
    hess : func
        Hessian of the function. It returns a 2d-array (matrix)
    tol : float
        Tolerance parameter for the stopping criterion
    maxit : int
        Maximum number of iterations
    x_0 : ndarray
        Starting point
    sol_x : ndarray
        Exact solution to the minimization problem
    sol_f : float
        Value of the function in correspondence with the exact solution
    alpha : float
        Step lenght
    eta : float
        Constant parameter in :math:`(0,0.25)`. Default value sigma=0.01
    

    Results
    -------
    results : dict
        Dictonary of the results given by the function. It contains the following items:
        - 'k': (int) final iteration at which convergence is reached
        - 'min_point' : (ndaray) computed point at which minimum is reached
        - 'min_value' : (float) computed minimum value of the function
        - 'interm_point' : (list) list of the intermediate points
        - 'interm_radius' : (list) list of the intermediate radii
        - 'error_x' : (float) 2-norm of the difference between 'min_point' and the exact solution
        - 'error_f' : (float) absolute error of the function evaluated at 'min_point' with respect
                      to its value in the exact minimum point
        - 'scalar_product' : (list) scalar product between the discent direction and the gradient

    '''

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
        # Choose an initial mu value
        mu = abs(min(min(eigval), 0)) + 1e-12
        coeff_vect = eigvect.T @ gradient/(eigval + mu)

        # Choose the optimal mu value which respects the condition on the 2-norm of p
        while sum([coeff**2 for coeff in coeff_vect]) > delta**2:
            mu = mu*2
            coeff_vect = eigvect.T @ gradient/(eigval + mu)
        
        # Compute the descent direction
        p = - eigvect @ coeff_vect
        scalar_prod.append(- gradient @ eigvect @ np.diag(1/eigval) @ eigvect.T @ gradient)

        new_point = old_point + alpha*p

        # Choose the value of delta for the successive iteration
        rho = (func(new_point)-func(old_point))/(p @ gradient + 0.5*p @ hessian @ p)

        if 0 < rho < 0.25:
            delta = delta/4
            interm_radius.append(delta)
        elif rho > 0.75:
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
    
    results = {'k' : k, 'min_point' : new_point, 'min_value' : min_value,
               'interm_point' : interm_points, 'interm_radius' : interm_radius, 'error_x' : error_x,
               'error_f' : error_f, 'scalar_product' : scalar_prod}

    return results
