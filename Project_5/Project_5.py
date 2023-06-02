''' This file serves as a library. It contains the function int_point which implements the interior
    point method for constrained minimization problems.'''
import numpy as np
import scipy.linalg


def int_point(func, grad_func, hess_func, constr, grad_constr, hess_constr, x0, method='basic',
              alpha=1., beta=1., gamma=1., mu=1e-12, tol=1e-12, maxit=100, l0='random', z0='random',
              curv=False, seed=1):
    '''This function implements the interior point method for constrained minimization problems.
    Parameters
    ----------
    func : function
        Function to be minimized. It must be :math:`f : \mathbb{R}^{n} \rightarrow \mathbb{R}`
    grad_func : function
        Gradient of the function. It returns a 1d-array (vector)
    hess_func : function
        Hessian of the function. It returns a 2d-array (matrix)
    constr : function
        Function of the constraints. It must be :math:`c : \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}`
        and it must return a 1d-array (vector).
    grad_constr : function
        Gradient of the function of the constraints. It returns a 2d-array (matrix)
    hess_constr : function
        Hessian of the function of the constraints. It returns a 3d-array
    x_0 : ndarray
        Starting point
    method : str
        String to choose the method to perform the Newton step. Possible values are
        - 'basic': to solve the full linear system
        - 'first': to solve the first reduced linear system
        - 'full': to solve the fully reduced linear system
        Default value method='basic'.
    alpha : float
        Starting step-lenght for the parameter :math:`x`. Default value alpha=1.
    beta : float
        Starting step-lenght for the parameter :math:`\lambda`. Default value beta=1.
    gamma : float
        Starting step-lenght for the parameter :math:`z`. Default value gamma=1.
    mu : float
        Coefficient of the log barrier term
    tol : float
        Tolerance parameter for the stopping criterion. Default value tol=1e-12
    maxit : int
        Maximum number of iterations. Default value maxit=100
    l0 : str or ndarray
        Starting values of the Lagrange multipliers. Deafult value l0='random', to generate
        uniformly distributed random values in the interval [1e-16,10).
    z0 : str or ndarray
        Starting values of the slack variable. Deafult value z0='random', to generate
        uniformly distributed random values in the interval [1e-16,10).
    curv : bool
        Boolean value to choose whether to discard the curvature term in the expression of the
        Jacobian of the system. Default value curv=False.
    seed : int
        Parameter to fix the seed. Default value seed=1
    
    Results
    -------
    results : dict
    Dictionary of the results given by the function. It contains the following items:
    - 'convergence' : (bool) True if the algorithm converges, False if it doesn't converge
    - 'n_iter' : (int) progressive number of final iteration
    - 'x_min' : (ndarray) computed point at which the minimum of the function is reached
    - 'f_min' : (float) computed minimum value of the function
    - 'x_interm' : (list) list of the intermediate points
    - 'lambda_interm' : (list) list of the intermediate Lagrange multipliers
    - 'z_interm' : (list) list of the intermediate slack variables
'''

    # Check if the starting point is in the feasible set
    if any(constr(x0) < 0):
        print(f'Starting point x0 = {x0} is not feasible')
        return False
    
    # Assign initial values to the the variables x_old, lambda_old and z_old
    x_old = x0
    lambda_old, z_old = l0, z0
    m, n = len(constr(x_old)), len(x_old)

    # Fix the seed and initialize lambda_old and z_old by using the uniform random distribution if required 
    np.random.seed(seed=seed)
    if type(l0) == str and l0 == 'random':
        lambda_old = np.random.uniform(low=1e-16, high= 10., size=m)
    if type(z0) == str and z0 == 'random':
        z_old = np.random.uniform(low=1e-16, high= 10., size=m)

    # Compute the vector R
    r1 = grad_func(x_old) - lambda_old @ grad_constr(x_old)
    r2 = constr(x_old) - z_old
    r3 = z_old*lambda_old - mu
    R = np.array([*r1, *r2, *r3])

    # Append the starting values of the variables to the corresponding lists
    x_interm = [x0]
    lambda_interm = [lambda_old]
    z_interm = [z_old]

    # Cycle until the stopping criterion is satisfied or k reaches the maximum number of iterations
    k = 0
    while(np.linalg.norm(R)>tol and k < maxit):
        a0, b0, g0 = alpha, beta, gamma

        # Compute matrices and vectors entering the expression of the linear system
        Z = np.diag(z_old)
        Z_inv = np.linalg.inv(Z)
        grad_c = grad_constr(x_old)
        Lambda = np.diag(lambda_old)
        hess_f = hess_func(x_old)
        K = 0
        if curv == True:
            hess_c = hess_constr(x_old)
            K = - lambda_old @ hess_c

        # Choose the method to perform the Newton step and compute dx, dl, dz
        # Solve the full system
        if method == 'basic':            
            Jacobian = np.block([[ hess_f + K, - grad_c.T, np.zeros((n,m)) ],
                                 [ grad_c , np.zeros((m,m)), - np.eye(m)],
                                 [ np.zeros((m,n)), Z, Lambda ]])
            p = np.linalg.solve(Jacobian, -R)
            dx = p[:n]
            dl = p[n:n+m]
            dz = p[n+m:]

        # Solve the first reduced system
        if method == 'first':
            Lambda_inv = np.linalg.inv(Lambda)
            matrix = np.block([[hess_f + K, -(grad_c).T],
                               [grad_c, Lambda_inv @ Z]])
            vector = np.array([*r1, *(r2 + Lambda_inv @ r3)])

            p = np.linalg.solve(matrix,-vector)
            dx = p[:n]
            dl = p[n:]
            dz = - Lambda_inv @ (r3 + Z @ dl)

        # Solve the fully reduced system
        if method == 'full':            
            matrix = hess_f + K + (grad_c).T @ (Z_inv @ Lambda @ grad_c )
            vect = - r1 - (grad_c).T @ Z_inv @ (r3 + Lambda @ r2)

            dx = np.linalg.solve(matrix,vect)
            dl = - Z_inv @ Lambda @ r2 - Z_inv @ r3 -Z_inv @ Lambda @ grad_c @ dx
            dz = - np.linalg.inv(Lambda) @ (r3 + Z @ dl)

        # Update the values of the variables
        x_new = x_old + a0*dx
        lambda_new = lambda_old + b0*dl
        z_new = z_old + g0*dz

        # Check if the updated values satisfy the required conditions and
        # re-update them until necessary
        while any(constr(x_new) < 0 ):
            a0 = a0/2
            x_new = x_old + a0*dx

        while any(lambda_new < 0 ):
            b0 = b0/2
            lambda_new = lambda_old + b0*dl

        while any(z_new <= 0 ):
            g0 = g0/2
            z_new = z_old + g0*dz

        # Append the new values of the variables to the corresponding lists
        x_interm.append(x_new)
        lambda_interm.append(lambda_new)
        z_interm.append(z_new)

        # Compute the new values of R
        r1 = grad_func(x_new) - lambda_new @ grad_constr(x_new)
        r2 = constr(x_new) - z_new
        r3 = z_new*lambda_new - mu
        R = np.array([*r1, *r2, *r3])

        x_old = x_new
        lambda_old = lambda_new
        z_old = z_new
        k = k + 1
    
    # Check if the convergence is reached
    conv = True
    if k == maxit:
        conv = False

    f_min = func(x_new)
    
    results = {'convergence' : conv, 'n_iter' : k , 'x_min' : x_new, 'f_min' : f_min,
            'x_interm' : x_interm, 'lambda_interm' : lambda_interm, 'z_interm' : z_interm}
    return results

if __name__ == '__main__':
    pass
