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
ax_t_list.scatter(range(1,t),t_counter_list, label = 't_list errors')
ax_t_list.legend()
ax_t_list.set_xlabel('-log10(tolerance)')
ax_t_list.set_ylabel('number of iterations')

plt.show()

############# QR factrorization with Rayleigh quotient shift and deflation ########################
# if True:
#     while np.amax(np.abs(A - np.diag(np.diag(A)))) >= tol:
#     Q, R = np.linalg.qr(A)
#     A = R@Q

#     computed_eigvals = np.diag(A)
#     abs_error = max(np.abs(exact_eigenvalues - computed_eigvals))
#     abs_error_list.append(abs_error)

#     counter += 1

#     if abs_error < 10**(-t):
#         t_counter_list.append(counter)
#         t += 1

#     if counter in [1,5,10,15]:
#         print(A)
#         print('--------------------------')
# print(f'counter = {counter}')
# print(A)
# computed_eigvals = np.diag(A)

# print('------------------------')
# print(f'exact eigenvalues = {format(exact_eigenvalues)}')
# print(f'computed eigenvalues = {format(computed_eigvals)}')
# diff_eigvals = np.abs(exact_eigenvalues - computed_eigvals)
# print(f'difference eigvals = {format(diff_eigvals)}')

# fig_total, ax_total = plt.subplots()
# ax_total.scatter(range(1,counter+1), np.log10(abs_error_list), label = 'total errors')
# ax_total.legend()
# ax_total.set_xlabel('iterations')
# ax_total.set_ylabel('log10(absolute error)')

# fig_t_list, ax_t_list = plt.subplots()
# ax_t_list.scatter(range(1,t),t_counter_list, label = 't_list errors')
# ax_t_list.legend()
# ax_t_list.set_xlabel('-log10(tolerance)')
# ax_t_list.set_ylabel('number of iterations')

# plt.show()
