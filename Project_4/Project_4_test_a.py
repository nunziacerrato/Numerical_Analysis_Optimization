''' This script implements the unconstrained optimization of the test function (a)
    using the Newton algorithms reported in the library Project_4.py.
'''
from Project_4 import *
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import pandas as pd
from decimal import Decimal
import math

common_path = "Project_4"

#  Define the test function (a) and its gradient and hessian as lambda functions.
func_a = lambda x: (x[0]-2)**4 + ((x[0]-2)**2)*x[1]**2 + (x[1]+1)**2
grad_a = lambda x: np.array([4*(x[0]-2)**3 + 2*(x[0]-2)*(x[1]**2), 2*((x[0]-2)**2)*x[1] + 2*(x[1]+1)])
hess_a = lambda x: np.array([[12*(x[0]-2)**2 + 2*x[1]**2, 4*(x[0]-2)*x[1]],
                                  [4*(x[0]-2)*x[1], 2*(x[0]-2)**2 + 2]])
x0_a_1 = np.array([1,1])
x0_a_2 = np.array([2,-1])
sol_x_a = (2,-1)
sol_f_a = 0


# Compute the minimum value of the function using the standard Newton algorithm
results = Newton(func_a, grad_a, hess_a, 1e-12, 100, x0_a_1, sol_x_a, sol_f_a, 1, backtracking=False)
# results = Newton_trust_region(func_a, grad_a, hess_a, 1e-5, 100, x0_a_1, sol_x_a, sol_f_a, alpha=1, eta=0.01)
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
    row += f"{i} & "
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
        row += f"{notation} & "
    # Remone the last character from the row of the table
    row = row[:-2]
    # Add the last row to the table
    table += f"{row}\\\\ \n \hline \n"

# Close the table
table += "\\end{tabular} \n \\end{table}"
print(table)


plot = False
if plot == True:
    ##################### PLOT #####################
    # Set size parameters for the plots
    tickparams_size = 16
    xylabel_size = 20
    suptitle_size = 35
    title_size = 22
    legend_size = 19

    ########### 3D plot ###########
    delta = 0.025
    X = np.arange(0., 3.0, delta)
    Y = np.arange(-2.0, 2.0, delta)
    X, Y = np.meshgrid(X,Y)
    Z = (X-2)**4 + ((X-2)**2)*(Y**2) + (Y+1)**2

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(14,9))
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                    linewidth=0, antialiased=True, alpha=0.7)

    # Add a color bar which maps values to colors and choose the format
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.0f}')
    cb = fig.colorbar(surf, shrink=0.5, aspect= 10)
    cb.ax.tick_params(labelsize=12) 

    # Obtain the scatterplot of the intermediate points
    interm_points = results['interm_point']
    interm_x = np.array([interm_points[i][0] for i in range(0,len(interm_points))])
    interm_y = np.array([interm_points[i][1] for i in range(0,len(interm_points))])
    zz = func_a([interm_x, interm_y])
    scatter = ax.plot(interm_x, interm_y, zz, '-o', markersize=4, color='black')
    scatter = ax.plot(interm_x[-1], interm_y[-1], zz[-1], '-o', markersize=4, color='red')

    ax.tick_params(labelsize = 12)
    ax.set_xlabel(r'$x_{1}$', fontsize=xylabel_size, labelpad=10)
    ax.set_ylabel(r'$x_{2}$', fontsize=xylabel_size, labelpad=10)
    ax.set_zlabel(r'$f\,(x_{1},x_{2})$', fontsize=xylabel_size)

    fig.savefig(f'{common_path}_latex\\Plot\\func_a_standard_newton_3d', bbox_inches='tight')

    ########### Contour plot ###########
    delta = 0.0025
    Xc = np.arange(0, 3, delta)
    Yc = np.arange(-2, 2, delta)
    Xc, Yc = np.meshgrid(Xc,Yc)
    Zc = (Xc-2)**4 + ((Xc-2)**2)*(Yc**2) + (Yc+1)**2

    fig_contour, ax_contour = plt.subplots(figsize=(14,9))
    cp = ax_contour.contour(Xc, Yc, Zc, levels = 200)
    cb = fig_contour.colorbar(cp, shrink=1, aspect=15, format='%.0f')
    cb.set_label(label = r'$f\,(x_{1},x_{2})$', size = 25)
    cb.ax.tick_params(labelsize=14) 
    scatter_c = ax_contour.plot(interm_x, interm_y, '-o', markersize=4, color='black')
    scatter_c = ax_contour.plot(interm_x[-1], interm_y[-1], '-o', markersize=4, color='red')

    ax_contour.tick_params(labelsize = tickparams_size)
    ax_contour.set_xlabel(r'$x_{1}$', fontsize=25)
    ax_contour.set_ylabel(r'$x_{2}$', fontsize=25)

    fig_contour.savefig(f'{common_path}_latex\\Plot\\func_a_standard_newton_contour', bbox_inches='tight')

    # plt.show()