'''
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy

# Set initial parameters
N = 500
tol = 1e-8
initial_guess = np.random.random(N)
initial_guess = initial_guess/np.linalg.norm(initial_guess, ord=2)
count = 0
diff = 10 * tol

# Set size parameters for the plots
tickparams_size = 16
xylabel_size = 18
suptitle_size = 35
title_size = 22
legend_size = 19

# Build T_N
T_N = scipy.sparse.diags([-1,2,-1],[-1,0,1],shape=(N,N)).toarray()

# Cholesky factorization of h^(-2)T_N
L, low = scipy.linalg.cho_factor(T_N)
L = (N+1) * L

vect_old = initial_guess

# Cycle until the difference in the 2-norm between two successive (approximated)
# eigenvectors is less than the chosen tolerance
while diff >= tol:
    # Compute and normalize the new vector by using the Choleski factorization of T_N
    vect_new = scipy.linalg.cho_solve((L, low),vect_old)
    vect_new = vect_new/np.linalg.norm(vect_new, ord=2)
    diff = np.linalg.norm(vect_new - vect_old, ord=2)
    count += 1
    vect_old = vect_new
    
# Compute the approximated eigenvalue
approx_eig = vect_new @ T_N @ vect_new * (N+1)**2

# print(f'Iterations performed = {count}')
# exact_eigenvalue_laplacian = np.pi**2
# print(f'Error on lambda_exact = {abs(approx_eig - exact_eigenvalue_laplacian)}')
# exact_eigenvalue_T_N = 2 * (N+1)**2 * (1-np.cos(np.pi/(N+1)))
# print(f'Error on lambda(h^(-2)T_N) = {abs(approx_eig - exact_eigenvalue_T_N)}')
# index = np.array(range(1,N+1))
# exact_eigenvector = np.sqrt(2/(N+1)) * np.sin(index*np.pi/(N+1))
# print(f'Error on eigenvector = {np.linalg.norm(vect_new - exact_eigenvector, ord=2)}')

print(f'Iterations performed = {count}')
exact_eigval_T_N = 2 * (N+1)**2 * (1-np.cos(np.pi/(N+1)))
print(f'Absolute error eigenvalue) = {abs(approx_eig - exact_eigval_T_N)}')
index = np.array(range(1,N+1))
exact_eigvect = np.sqrt(2/(N+1)) * np.sin(index*np.pi/(N+1))
print(f'2-norm error eigenvector = {np.linalg.norm(vect_new - exact_eigvect, ord=2)}')

# Plot
fig_eigvect, ax_eigvect = plt.subplots()
fig_eigvect.suptitle(fr'First eigenvector of $T_N$ with N={N}', fontsize=title_size)
ax_eigvect.tick_params(labelsize = tickparams_size)
ax_eigvect.set_xlabel('k',fontsize=20)
ax_eigvect.set_ylabel(r'$u_1(k)$',fontsize=20)
ax_eigvect.scatter(index, vect_new)

plt.show()