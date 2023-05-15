''' This script implements the unconstrained optimization of the test function (b)
    using the Newton algorithms reported in the library Project_4.py.
'''
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
from decimal import Decimal
from Project_4 import *

#  Define the test function (b) and its gradient and hessian as lambda functions.
b = np.array([5.04, -59.4, 146.4, -96.6])
H = np.array([[0.16,-1.2,2.4,-1.4],
              [-1.2,12.0,-27.0,16.8],
              [2.4,-27.0,64.8,-42.0],
              [-1.4,16.8,-42.0,28.0]])
func_b = lambda x: b @ x + 0.5*(x @ H @ x)
grad_b = lambda x: b + H @ x
hess_b = lambda x: H
x0_b_1 = np.array([-1,3,3,0])
sol_x_b = np.array([1,0,-1,2])
sol_f_b = -167.28

results = Newton(func_b, grad_b, hess_b, 1e-12, 100, x0_b_1, sol_x_b, sol_f_b, 1, backtracking=False)
k = results['k']
conv = results['convergence']
min_point = results['min_point']
min_value = results['min_value']
print(f'convergence = {conv}, with {k} steps')
print(f'min point = {min_point}')
print(f'min value = {min_value}')

# Initialize LateX code for creating a table
table = "\\begin{table} \n \\centering \n \\begin{tabular}{|c|c|c|c|} \n \hline \n"
table += "k & $\| \\textbf{x}_{k} - \\textbf{x}^*\|_{2} $ & \
$f_{1}(\\textbf{x}_{k}) - f_{1}(\\textbf{x}^{*}) $ & $-\\nabla f_{1}(\\textbf{x}_{k})^{T}\
[\\nabla^{2}f_{1}(\\textbf{x}_{k})]^{-1} \\nabla f_{1}(\\textbf{x}_{k})$ \\\\ \n \hline \n"

# Extract data from the dictionary "results"
for i in range(len(results['error_x'])):
    list_ = []
    row = ""
    list_.append(results['error_x'][i])
    list_.append(results['error_f'][i])
    list_.append(results['scalar_product'][i])
    row += f"${i}$ & "
    # Report data in scientific notation with three digits after the dot
    for value in list_:
        exp = 0
        if value != 0:
            exp = int(math.floor(math.log10(abs(value))))
        coeff = round(value / (10 ** exp), 3)
        if exp == 0:
            notation = f"{Decimal(coeff):.3f}"
        else:
            notation = f"{coeff}\\times10^{{{exp}}}"
        # Add the value to the table row
        row += f"${notation}$ & "
    # Remone the last character from the row of the table
    row = row[:-2]
    # Add the last row to the table
    table += f"{row}\\\\ \n \hline \n"

# Close the table
table += "\\end{tabular} \n \\end{table}"
print(table)