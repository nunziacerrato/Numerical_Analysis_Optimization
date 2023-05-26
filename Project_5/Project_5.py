import numpy as np



def int_point(func,grad_func,hess_func,constr,grad_constr,x0,method='full',alpha=1.,beta=1.,gamma=1.,tol=1e-12,maxit=100,seed=1):
    ''' '''
    # Check if the starting point is in the feasible set
    if (constr(x0) > 0).any:
        print(f'Starting point x0 = {x0} is not feasible')
        return False
    
    x_old = x0
    m = len(constr(x_old))

    # Fix the seed to ensure reproducibility
    np.random.seed(seed=seed)
    # Initialize the parameter mu and the vectors lambda, z by using the uniform random distribution in [1e-16,1)
    mu = np.random.uniform(low=1e-16, high=1.)
    z_old = np.random.uniform(low=1e-16, high=1., size=m)
    lambda_old = np.random.uniform(low=1e-16, high=1., size=m)

    r1 = grad_func(x_old) - lambda_old @ grad_constr(x_old)
    r2 = constr(x_old) - z_old
    r3 = z_old*lambda_old - mu

    R = np.array([*r1, *r2, *r3])
    k = 0
    while(np.linalg.norm(R)>tol and k < maxit):

        a0 = alpha
        b0 = beta
        g0 = gamma

        # Choose the method to compute dx, dl, dz
        if method == 'full':
            # Compute matrices and vectors entering the expression of the linear system
            Z_inv = np.linalg.inv(np.diag(z_old))
            grad_c = grad_constr(x_old)
            Lambda = np.diag(lambda_old)
            
            matrix = hess_func(x_old) + (grad_c).T @ (Z_inv @ Lambda @ grad_c )
            vect = - r1 - (grad_c).T @ Z_inv @ (r3 + Lambda @ r2)
            
            # Solve the linear system to obtain dx
            dx = np.linalg.solve(matrix,vect)

            # Compute dl and dz
            dl = - Z_inv @ Lambda @ r2 - Z_inv @ r3 -Z_inv @ Lambda @ grad_c @ dx
            dz = - np.linalg.inv(Lambda) @ (r3 + np.diag(z_old) @ dl)
        
        if method == 'first':
            pass
        if method == 'basic':
            pass

        x_new = x_old + alpha*dx
        lambda_new = lambda_old + beta*dl
        z_new = z_old + gamma*dz

        while( (constr(x_new) >0).any ):
            a0 = a0/2
            x_new = x_old + a0*dx

        while(lambda_new < 0 ):
            b0 = b0/2
            lambda_new = lambda_old + b0*dl

        while(z_new <= 0 ):
            g0 = g0/2
            z_new = z_old + g0*dz

        r1 = grad_func(x_new) - lambda_new @ grad_constr(x_new)
        r2 = constr(x_new) - z_new
        r3 = z_new*lambda_new - mu
        
        R = np.array([*r1, *r2, *r3])

        x_old = x_new
        lambda_old = lambda_new
        z_old = z_new
        k = k + 1
    
    conv = True
    if k == maxit:
        conv = False
    
    results = {'conv': conv, }

    return results

if __name__ == '__main__':
    constr = np.array([1,2,3])
    z = np.array([2,-2,2])
    ZZ = np.linalg.inv(np.diag(z))
    vect = np.array([*constr, *z])
    print(vect)
    # if (z >0).any:
    #     print(z>0)
