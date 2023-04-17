''' Exercise 1 - Project 3.
'''

import numpy as np
import matplotlib.pyplot as plt

# Set the initial values
tol = 1e-5
counter = 0
t = 1 # Initial exponent for the tolerance

# Set the number of significant digits
np.set_printoptions(precision=15, suppress=True)

# Construct the matrix A
A = np.array([[4,3,2,1],[3,4,3,2],[2,3,4,3],[1,2,3,4]])
# Compute the exact eigenvalues
exact_eigenvalues = np.linalg.eigvals(A)

# Initialize an empty list to store the number of iterations at each t
t_counter_list = []

# Cycle until the stopping criterion is not satisfied
while np.amax(np.abs(A - np.diag(np.diag(A)))) >= tol:
    # Obtain the QR factorization of A and perform the QR iteration
    Q, R = np.linalg.qr(A)
    A = R@Q

    # Obtain the current approximation of the eigenvalues
    computed_eigvals = np.diag(A)

    # Compute and store the maximum absolute error on the eigenvalues
    abs_error = max(np.abs(exact_eigenvalues - computed_eigvals))

    counter += 1

    if abs_error < 10**(-t):
        t_counter_list.append(counter)
        t += 1

    if counter in [1,5,10,15]:
        print(f'k={counter}')
        print(A)
        print('--------------------------')
print(f'Final k = {counter}')
print(A)
computed_eigvals = np.diag(A)

print('------------------------')
print(f'Exact eigenvalues = {format(exact_eigenvalues)}')
print(f'Computed eigenvalues = {format(computed_eigvals)}')
diff_eigvals = np.abs(exact_eigenvalues - computed_eigvals)
print(f'Absolute error = {format(diff_eigvals)}')

# Plot

# Set size parameters for the plots
tickparams_size = 16
xylabel_size = 22
suptitle_size = 35
title_size = 22
legend_size = 19

t = min(t, 9)
fig_t_list, ax_t_list = plt.subplots()
ax_t_list.scatter(range(1,t),t_counter_list[:8], label = 'Simple QR', s=65)
ax_t_list.legend(fontsize=legend_size)
ax_t_list.tick_params(labelsize = tickparams_size)
ax_t_list.set_xlabel('t', fontsize = xylabel_size)
ax_t_list.set_ylabel('Number of iterations', fontsize = xylabel_size)

# plt.show()

######### QR factorization with Rayleigh quotient shift and deflation #########

# Initialize values
t = 9
tol_list = 10.**(-np.array(range(1,t)))
counter_list = []

# Construct the matrix A
A = np.array([[4,3,2,1],[3,4,3,2],[2,3,4,3],[1,2,3,4]])
# Compute the exact eigenvalues
exact_eigenvalues = np.linalg.eigvals(A)
exact_eigenvalues_copy = exact_eigenvalues.copy()

# Cycle on the list of tolerances
for tol in tol_list:
    counter = 0
    A = np.array([[4,3,2,1],[3,4,3,2],[2,3,4,3],[1,2,3,4]])
    exact_eigenvalues = exact_eigenvalues_copy.copy()
    dim = np.shape(A)[0]

    # Perform QR iteration until the matrix is reduced to a single value by deflation
    while dim > 0:
        approx_lambda = A[dim-1,dim-1]
        eig_error = abs(exact_eigenvalues-approx_lambda)
        # If error on an eigenvalue is less than the tolerance, perform deflation
        if min(eig_error) < tol:
            index = np.argmin(eig_error)
            exact_eigenvalues = np.delete(exact_eigenvalues,index)
            A = A[:dim-1,:dim-1]
            dim = dim - 1 
            continue
        
        # Compute the shift and perform QR iteration
        shift = A[dim-1,dim-1]
        Q, R = np.linalg.qr(A - shift * np.eye(dim))
        A = R@Q + shift * np.eye(dim)
        counter = counter + 1
 
    counter_list.append(counter)

# Plot
ax_t_list.scatter(range(1,9),counter_list[:8], label = 'Shifted QR', s=65)
ax_t_list.legend(fontsize=legend_size)

plt.show()