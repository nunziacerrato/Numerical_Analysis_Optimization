'''
'''

import numpy as np
import matplotlib.pyplot as plt

tol = 1e-8
counter = 0
# Set the number of significant digits
np.set_printoptions(precision=15, suppress=True)

A = np.array([[4,3,2,1],[3,4,3,2],[2,3,4,3],[1,2,3,4]])
exact_eigenvalues = np.linalg.eigvals(A)
abs_error_list = []

t_counter_list = []
t = 1

while np.amax(np.abs(A - np.diag(np.diag(A)))) >= tol:
    Q, R = np.linalg.qr(A)
    A = R@Q

    computed_eigvals = np.diag(A)
    abs_error = max(np.abs(exact_eigenvalues - computed_eigvals))
    abs_error_list.append(abs_error)

    counter += 1

    if abs_error < 10**(-t):
        t_counter_list.append(counter)
        t += 1

    if counter in [1,5,10,15]:
        print(A)
        print('--------------------------')
print(f'counter = {counter}')
print(A)
computed_eigvals = np.diag(A)

print('------------------------')
print(f'exact eigenvalues = {format(exact_eigenvalues)}')
print(f'computed eigenvalues = {format(computed_eigvals)}')
diff_eigvals = np.abs(exact_eigenvalues - computed_eigvals)
print(f'difference eigvals = {format(diff_eigvals)}')

fig_total, ax_total = plt.subplots()
ax_total.scatter(range(1,counter+1), np.log10(abs_error_list), label = 'total errors')
ax_total.legend()
ax_total.set_xlabel('iterations')
ax_total.set_ylabel('log10(absolute error)')

fig_t_list, ax_t_list = plt.subplots()
ax_t_list.scatter(range(1,t),t_counter_list, label = 'simple QR')
ax_t_list.legend()
ax_t_list.set_xlabel('-log10(tolerance)')
ax_t_list.set_ylabel('number of iterations')



############# QR factrorization with Rayleigh quotient shift and deflation ########################

t = 9
tol_list = 10.**(-np.array(range(1,t)))
counter_list = []

A = np.array([[4,3,2,1],[3,4,3,2],[2,3,4,3],[1,2,3,4]])
exact_eigenvalues = np.linalg.eigvals(A)
exact_eigenvalues_copy = exact_eigenvalues.copy()
# computed_eigenvalues = []


for tol in tol_list:
    counter = 0
    A = np.array([[4,3,2,1],[3,4,3,2],[2,3,4,3],[1,2,3,4]])
    exact_eigenvalues = exact_eigenvalues_copy.copy()
    dim = np.shape(A)[0]

    while dim > 0:
        approx_lambda = A[dim-1,dim-1]
        eig_error = abs(exact_eigenvalues-approx_lambda)
        if min(eig_error) < tol:
            # computed_eigenvalues.append(approx_lambda)
            index = np.argmin(eig_error)
            exact_eigenvalues = np.delete(exact_eigenvalues,index)
            A = A[:dim-1,:dim-1]
            dim = dim - 1 
            continue
        
        shift = A[dim-1,dim-1]
        Q, R = np.linalg.qr(A - shift * np.eye(dim))
        A = R@Q + shift * np.eye(dim)
        counter = counter + 1
 
    counter_list.append(counter)

# fig_t_list, ax_t_list = plt.subplots()
ax_t_list.scatter(range(1,t),counter_list, label = 'shifted')
ax_t_list.legend()
ax_t_list.set_xlabel('-log10(tolerance)')
ax_t_list.set_ylabel('number of iterations')

plt.show()