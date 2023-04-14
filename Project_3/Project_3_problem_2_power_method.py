'''
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy

N = 500
tol = 1e-8
initial_guess = np.random.random(N)
initial_guess = initial_guess/np.linalg.norm(initial_guess, ord=2)
count = 0
diff = 10 * tol
# Build T_N
T_N = scipy.sparse.diags([-1,2,-1],[-1,0,1],shape=(N,N)).toarray()

# Cholesky factorization of h^(-2)T_N
L, low = scipy.linalg.cho_factor(T_N)
L = (N+1) * L

vect_old = initial_guess

while diff >= tol:
    vect_new = scipy.linalg.cho_solve((L, low),vect_old)
    vect_new = vect_new/np.linalg.norm(vect_new, ord=2)
    diff = np.linalg.norm(vect_new - vect_old, ord=2)
    approx_eig = vect_new @ T_N @ vect_new * (N+1)**2
    count += 1
    vect_old = vect_new
    print(diff)

print(f'Iterations performed = {count}')
exact_eigenvalue_laplacian = np.pi**2
print(f'Error on lambda_exact = {abs(approx_eig - exact_eigenvalue_laplacian)}')
exact_eigenvalue_T_N = 2 * (N+1)**2 * (1-np.cos(np.pi/(N+1)))
print(f'Error on lambda(h^(-2)T_N) = {abs(approx_eig - exact_eigenvalue_T_N)}')
index = np.array(range(1,N+1))
exact_eigenvector = np.sqrt(2/(N+1)) * np.sin(index*np.pi/(N+1))
print(f'Error on eigenvector = {np.linalg.norm(vect_new - exact_eigenvector, ord=2)}')

fig_eigvect, ax_eigvect = plt.subplots()
fig_eigvect.suptitle(fr'First eigenvector of $T_N$ with N={N}')

ax_eigvect.scatter(index, vect_new)
plt.show()