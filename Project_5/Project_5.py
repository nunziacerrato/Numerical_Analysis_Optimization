import numpy as np



def int_point(func, grad_func, hess_func, constr, grad_constr, hess_constr, x0, method='basic', alpha=1., beta=1.,
              gamma=1., mu=1e-12, tol=1e-12, maxit=100, l0='random', z0='random', seed=1):
    ''' aaa '''

    # Check if the starting point is in the feasible set
    if any(constr(x0) < 0):
        print(f'Starting point x0 = {x0} is not feasible')
        return False
    
    x_old = x0
    m = len(constr(x_old))
    n = len(x_old)

    # Fix the seed to ensure reproducibility
    np.random.seed(seed=seed)
    # Initialize the parameter mu and the vectors lambda, z by using the uniform random distribution in [1e-16,1)
    if l0 == 'random':
        lambda_old = np.random.uniform(low=1e-16, high= 10., size=m)
    else:
        lambda_old = l0
    if z0 == 'random':
        z_old = np.random.uniform(low=1e-16, high= 10., size=m)
    else:
        z_old = z0

    r1 = grad_func(x_old) - lambda_old @ grad_constr(x_old)
    r2 = constr(x_old) - z_old
    r3 = z_old*lambda_old - mu

    R = np.array([*r1, *r2, *r3])
    k = 0
    x_interm = [x0]
    lambda_interm = []
    z_interm = []
    while(np.linalg.norm(R)>tol and k < maxit):

        a0 = alpha
        b0 = beta
        g0 = gamma

        # Compute matrices and vectors entering the expression of the linear system
        Z = np.diag(z_old)
        Z_inv = np.linalg.inv(Z)
        grad_c = grad_constr(x_old)
        Lambda = np.diag(lambda_old)
        hess_f = hess_func(x_old)


        # Choose the method to compute dx, dl, dz
        if method == 'basic':
            hess_c = hess_constr(x_old)
            K=0
            # K = sum([lambda_old[i] * hess_c[:,i,:] for i in range(m)])
            Jacobian = np.block([[ hess_f + K, - grad_c.T, np.zeros((n,m)) ],
                                 [ grad_c , np.zeros((m,m)), - np.eye(m)],
                                 [ np.zeros((m,n)), Z, Lambda ]])
            p = np.linalg.solve(Jacobian, -R)
            dx = p[:n]
            dl = p[n:n+m]
            dz = p[n+m:]

        if method == 'first':
            Lambda_inv = np.linalg.inv(Lambda)
            matrix = np.block([[hess_f, -(grad_c).T],
                               [grad_c, Lambda_inv @ Z]])
            vector = np.array([*r1, *(r2 + Lambda_inv @ r3)])

            p = np.linalg.solve(matrix,-vector)
            dx = p[:n]
            dl = p[n:]
            dz = - Lambda_inv @ (r3 + Z @ dl)

        if method == 'full':
            
            matrix = hess_f + (grad_c).T @ (Z_inv @ Lambda @ grad_c )
            vect = - r1 - (grad_c).T @ Z_inv @ (r3 + Lambda @ r2)
            
            # Solve the linear system to obtain dx
            dx = np.linalg.solve(matrix,vect)

            # Compute dl and dz
            dl = - Z_inv @ Lambda @ r2 - Z_inv @ r3 -Z_inv @ Lambda @ grad_c @ dx
            dz = - np.linalg.inv(Lambda) @ (r3 + Z @ dl)

        x_new = x_old + a0*dx
        lambda_new = lambda_old + b0*dl
        z_new = z_old + g0*dz

        while any(constr(x_new) < 0 ):
            a0 = a0/2
            x_new = x_old + a0*dx

        while any(lambda_new < 0 ):
            b0 = b0/2
            lambda_new = lambda_old + b0*dl

        while any(z_new <= 0 ):
            g0 = g0/2
            z_new = z_old + g0*dz

        x_interm.append(x_new)
        lambda_interm.append(lambda_new)
        z_interm.append(z_new)

        r1 = grad_func(x_new) - lambda_new @ grad_constr(x_new)
        r2 = constr(x_new) - z_new
        r3 = z_new*lambda_new - mu
        
        R = np.array([*r1, *r2, *r3])

        x_old = x_new
        lambda_old = lambda_new
        z_old = z_new
        k = k + 1
        # print(x_new,lambda_new, z_new)
    
    conv = True
    if k == maxit:
        conv = False

    f_min = func(x_new)
    
    results = {'convergence' : conv, 'n_iter' : k , 'x_min' : x_new, 'f_min' : f_min,
            'x_interm' : x_interm, 'lambda_interm' : lambda_interm, 'z_interm' : z_interm, 'mu' : mu}
    return results

if __name__ == '__main__':
    constr = np.array([1,2,3])
    z = np.array([2,-2,2])
    ZZ = np.linalg.inv(np.diag(z))
    vect = np.array([*constr, *z])
    print(vect)
    # if (z >0).any:
    #     print(z>0)
