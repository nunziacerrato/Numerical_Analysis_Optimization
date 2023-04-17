'''
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy

# Set initial parameters
N = 500
lambda_5 = 25*np.pi**2
initial_guess = np.random.random(N)
initial_guess = initial_guess/np.linalg.norm(initial_guess, ord=2)
count = 0

# Set size parameters for the plots
tickparams_size = 16
xylabel_size = 18
suptitle_size = 35
title_size = 22
legend_size = 19

# Build T_N
T_N = scipy.sparse.diags([-1,2,-1],[-1,0,1],shape=(N,N)).toarray()

tol = 1e-12 * np.linalg.norm((N+1)**2 * T_N, ord = np.inf)
diff = 10 * tol

# LU factorization of h^(-2)T_N
lu, piv = scipy.linalg.lu_factor(T_N*(N+1)**2 - lambda_5 * np.eye(N))

vect_old = initial_guess

# Cycle until the stopping criterion is satisfied - condition on the residual
while diff >= tol:
    # Compute and normalize the new vector by using the LU factorization to solve a linear system
    vect_new = scipy.linalg.lu_solve((lu, piv), vect_old)  
    vect_new = vect_new/np.linalg.norm(vect_new, ord=2)
    # Compute the approximated eigenvalue
    approx_eig = vect_new @ T_N @ vect_new * (N+1)**2
    diff = np.linalg.norm(( T_N * (N+1)**2 - approx_eig * np.eye(N) ) @ vect_new, ord=np.inf)
    count += 1
    vect_old = vect_new

print(f'Iterations performed = {count}')
exact_eigenvalue_laplacian = 25 * np.pi**2
print(f'Error on lambda_exact = {abs(approx_eig - exact_eigenvalue_laplacian)}')
exact_eigenvalue_T_N = 2 * (N+1)**2 * (1-np.cos(5*np.pi/(N+1)))
print(f'Error on lambda(h^(-2)T_N) = {abs(approx_eig - exact_eigenvalue_T_N)}')
index = np.array(range(1,N+1))
exact_eigenvector = np.sqrt(2/(N+1)) * np.sin(index*np.pi*5/(N+1))
print(f'Error on eigenvector = {np.linalg.norm(vect_new - exact_eigenvector, ord=2)}')

# Plot
fig_eigvect, ax_eigvect = plt.subplots()
fig_eigvect.suptitle(fr'Fifth eigenvector of $T_N$ with N={N}',fontsize=title_size)
ax_eigvect.set_xlabel('k', fontsize=xylabel_size)
ax_eigvect.set_ylabel(r'$u_5(k)$',fontsize=xylabel_size)
ax_eigvect.tick_params(labelsize = tickparams_size)
ax_eigvect.scatter(index, vect_new)
plt.show()